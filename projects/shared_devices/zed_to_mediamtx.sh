#!/bin/bash
# ZED Camera to MediaMTX streaming script
# Streams left and right ZED camera feeds to MediaMTX via RTSP with NVENC encoding.
#
# Automatically selects the correct NVENC encoder based on the detected GPU:
#   - Blackwell (RTX 50-series): nvcudah264enc with the new preset/tune API.
#     The legacy preset names ("low-latency-hq", etc.) are rejected by NVENC
#     on Blackwell even though GStreamer still exposes them.
#   - Pre-Blackwell (RTX 40/30/20-series): nvh264enc with the legacy named
#     presets, which is what older driver branches support natively.
#
# Both code paths target low-latency, high-quality streaming at the configured
# resolution and bitrate.

set -u

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MEDIAMTX_HOST="localhost"
MEDIAMTX_PORT="8554"
# Resolution options: 0=HD2K(2208x1242), 1=HD1080(1920x1080), 2=HD1200(1920x1200), 3=HD720(1280x720), 5=VGA(672x376)
# use "ZED_Explorer" app to test different resolutions and frame rates
# HD1200 (RESOLUTION=2) uses the full 1920x1200 native sensor on ZED X / GMSL2 cameras.
# HD1080 (RESOLUTION=1) is a center crop of the same sensor, losing ~120 rows top/bottom.
RESOLUTION="3"
FRAMERATE="60"   # HD720@60 for XR teleop (low latency / smooth); ZED Mini supports 60fps only at HD720 or lower
BITRATE="16000" # Per-eye kbps. The router is wired directly to the PC (dedicated
                 # single-hop 6GHz to one headset), so bandwidth is NOT the limit and
                 # bitrate is not the cause of lag/ghosting. Kept generous for crisp
                 # motion. Sync-safe; MediaMTX writeQueueSize stays 512 for eye sync.
# ~0.5s keyframe interval. MediaMTX can't relay a WebRTC client's keyframe request
# (PLI) to this encoder, so a lost reference only heals on the next scheduled IDR.
# 0.5s recovers ghosting fast without the big once-per-second I-frame burst that a
# long GOP creates (and at 8 Mbps these IDRs are small). Symmetric -> sync-safe.
GOP_SIZE=$(( FRAMERATE / 2 ))
QUEUE_SIZE="1"   # Minimize buffering for lowest latency (1-3 buffers)

# SBS=1 streams ONE stitched side-by-side [left|right] frame to a single path /zed_stereo, instead
# of two separate left/right paths. zedsrc stream-type=2 is ALREADY a side-by-side frame (that's why
# zeddemux can split it), so SBS just skips the demux and encodes the whole frame. The XR client then
# decodes ONE H.264 stream and splits L/R in-shader (MediaMTXReceiver.singleStreamSbs + the _SBS
# shader mode): one decoder instead of two -> less decode contention, guaranteed L/R sync, one WebRTC
# connection -> lower latency. This is how the native ZED SDK streams. Point the XR client's base
# address at ".../zed_stereo".
# DEFAULT is now SBS (1) — matches the XR client (MediaMTXReceiver.singleStreamSbs defaults ON). Run
# with SBS=0 to fall back to the original two-path (left/right) pipeline.
SBS="${SBS:-1}"
if [[ "${SBS}" == "1" ]]; then
    BITRATE=24000   # one 2560x720 frame instead of two 1280x720 -> bump the combined budget
fi

# Optional ZED calibration/settings folder. Use this on machines where you lack
# write access to the system path /usr/local/zed/settings (no sudo / not in the
# 'zed' group): point it at a user-owned dir containing SN<serial>.conf, e.g.
#   ZED_SETTINGS_PATH="$HOME/zed_settings" ./zed_to_mediamtx.sh
# Leave empty to use the SDK default (/usr/local/zed/settings).
ZED_SETTINGS_PATH="${ZED_SETTINGS_PATH:-}"
if [[ -n "${ZED_SETTINGS_PATH}" ]]; then
    ZED_SETTINGS_OPT="optional-settings-path=${ZED_SETTINGS_PATH}"
else
    ZED_SETTINGS_OPT=""
fi

# Optional cap on the auto-exposure shutter time (microseconds) to kill motion
# blur / smearing on fast motion. At 60fps the camera can expose up to ~16000us
# (16ms); capping lower forces a faster shutter = crisper motion, at the cost of a
# darker/noisier image (auto-gain compensates). Try 4000, drop toward 2000 if it
# still smears, raise if too dark/noisy. Empty = SDK default (no cap). e.g.:
#   MAX_EXPOSURE_US=4000 ./zed_to_mediamtx.sh
MAX_EXPOSURE_US="${MAX_EXPOSURE_US:-}"
if [[ -n "${MAX_EXPOSURE_US}" ]]; then
    EXPOSURE_OPT="ctrl-aec-agc=true ctrl-exposure-range-max=${MAX_EXPOSURE_US}"
else
    EXPOSURE_OPT=""
fi

# ----------------------------------------------------------------------------
# GPU detection and encoder selection
# ----------------------------------------------------------------------------
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

if [[ -z "${GPU_NAME}" ]]; then
    echo "ERROR: Could not query GPU via nvidia-smi. Is the NVIDIA driver loaded?" >&2
    exit 1
fi

# Encoder selection is by capability, not GPU name. The legacy nvh264enc named
# presets ("low-latency-hq", etc.) are rejected by NVENC on newer driver/NVENC
# builds (CUDA 12+, as shipped on Ada/Blackwell and on older cards once the
# driver is updated) even though GStreamer still exposes them in the enum. The
# modern nvcudah264enc element uses the P1-P7 preset + tune API those drivers
# require. Keying off the GPU name is brittle (e.g. an RTX 20/40 card on a new
# driver also needs the modern API), so instead we probe: run a tiny test encode
# through nvcudah264enc and use it if it succeeds, falling back to nvh264enc.
probe_modern_encoder() {
    gst-launch-1.0 -e videotestsrc num-buffers=2 \
        ! video/x-raw,width=320,height=240,framerate=30/1 \
        ! videoconvert ! video/x-raw,format=NV12 ! cudaupload \
        ! "video/x-raw(memory:CUDAMemory),format=NV12" \
        ! nvcudah264enc preset=p5 tune=low-latency rate-control=cbr bitrate=4000 \
        ! h264parse ! fakesink >/dev/null 2>&1
}

if gst-inspect-1.0 nvcudah264enc >/dev/null 2>&1 && probe_modern_encoder; then
    ENCODER_NAME="nvcudah264enc"
    # b-frames=0 + repeat-sequence-header=true: no reordering (lower latency) and
    # SPS/PPS on every IDR so a client resyncs on the next keyframe after loss.
    # Both are symmetric per-stream -> do NOT affect left/right eye sync.
    # Quality-maxed but still low-latency tune (idle RTX 5080 has the headroom):
    #   preset=p7        highest-quality preset (best motion estimation)
    #   multi-pass=two-pass  better bit allocation on motion -> less smearing
    #   qos=false        NEVER skip-encode frames for "lateness" (skips look like
    #                    motion smear/judder vs ZED_Explorer's smooth 60fps)
    ENCODER_OPTS="preset=p7 tune=low-latency rate-control=cbr bitrate=${BITRATE} gop-size=${GOP_SIZE} multi-pass=two-pass b-frames=0 repeat-sequence-header=true zero-reorder-delay=true spatial-aq=true qos=false"
    ENCODER_API="modern (nvcudah264enc, p7 two-pass, low-latency)"
    # Modern CUDA encoder needs frames already in CUDA memory as NV12.
    PRE_ENCODER="videoconvert ! video/x-raw,format=NV12 ! cudaupload ! video/x-raw(memory:CUDAMemory),format=NV12"
else
    ENCODER_NAME="nvh264enc"
    ENCODER_OPTS="preset=low-latency-hq rc-mode=cbr bitrate=${BITRATE} gop-size=${GOP_SIZE} zerolatency=true bframes=0 spatial-aq=true aq-strength=8 qos=true"
    ENCODER_API="legacy (nvh264enc, low-latency-hq)"
    # Legacy encoder accepts system-memory raw frames; pre-encoder is just a CPU convert.
    PRE_ENCODER="videoconvert ! video/x-raw,format=I420"
fi

# ----------------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------------
echo "Starting ZED stereo camera pipeline with NVIDIA GPU encoding..."
echo "  GPU:        ${GPU_NAME}"
echo "  Encoder:    ${ENCODER_API}"
echo "  Resolution: ${RESOLUTION} @ ${FRAMERATE}fps"
echo "  Bitrate:    ${BITRATE} kbps"
echo "Streaming to:"
if [[ "${SBS}" == "1" ]]; then
    echo "  SBS:   rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed_stereo  (single side-by-side; default)"
else
    echo "  Left:  rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/left"
    echo "  Right: rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/right"
fi
echo ""

# ----------------------------------------------------------------------------
# Stop any pipelines left over from a prior run
# ----------------------------------------------------------------------------
pkill -f "gst-launch.*zedsrc" 2>/dev/null || true

# ----------------------------------------------------------------------------
# Run the pipeline
# Single zedsrc, demuxed into left/right, each encoded and sent via RTSP.
# ----------------------------------------------------------------------------
if [[ "${SBS}" == "1" ]]; then
    # Single stitched side-by-side stream: encode the whole stream-type=2 frame as ONE H.264 stream.
    echo "  Mode:       SINGLE side-by-side -> rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed_stereo"
    echo ""
    /usr/bin/gst-launch-1.0 -e \
        zedsrc ${ZED_SETTINGS_OPT} ${EXPOSURE_OPT} stream-type=2 camera-resolution=${RESOLUTION} camera-fps=${FRAMERATE} ! \
        video/x-raw,format=BGRA,framerate=${FRAMERATE}/1 ! \
        queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
        ${PRE_ENCODER} ! \
        ${ENCODER_NAME} ${ENCODER_OPTS} ! \
        video/x-h264,profile=main,stream-format=byte-stream ! \
        h264parse ! \
        queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
        rtspclientsink location=rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed_stereo protocols=tcp latency=0
else
    /usr/bin/gst-launch-1.0 -e \
        zedsrc ${ZED_SETTINGS_OPT} ${EXPOSURE_OPT} stream-type=2 camera-resolution=${RESOLUTION} camera-fps=${FRAMERATE} ! \
        video/x-raw,format=BGRA,framerate=${FRAMERATE}/1 ! \
        zeddemux name=demux \
        demux.src_left ! \
            queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
            ${PRE_ENCODER} ! \
            ${ENCODER_NAME} ${ENCODER_OPTS} ! \
            video/x-h264,profile=main,stream-format=byte-stream ! \
            h264parse ! \
            queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
            rtspclientsink location=rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/left protocols=tcp latency=0 \
        demux.src_aux ! \
            queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
            ${PRE_ENCODER} ! \
            ${ENCODER_NAME} ${ENCODER_OPTS} ! \
            video/x-h264,profile=main,stream-format=byte-stream ! \
            h264parse ! \
            queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
            rtspclientsink location=rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/right protocols=tcp latency=0
fi

echo ""
echo "Pipeline stopped."