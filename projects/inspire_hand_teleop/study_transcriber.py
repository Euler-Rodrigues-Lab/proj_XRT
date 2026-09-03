#!/usr/bin/env python3
"""study_transcriber.py — live speech transcription for the user study (own process).

Captures the CONFERENCE ROOM microphone, segments speech with silero VAD, transcribes
each utterance with faster-whisper (GPU), identifies the speaker against enrolled voice
profiles (resemblyzer embeddings), and writes an epoch-timestamped TRANSCRIPT CSV into
the session folder. RAW AUDIO IS NEVER SAVED — only text + embeddings-in-memory (the
IRB-friendly design: no voice recordings are retained).

Runs as the manager's SUBPROCESS (``--serve --fd N`` over a socketpair, length-prefixed
JSON — the same pattern as apriltag_tracker/head_cam_localize): all audio + ML work
lives on its own GIL so teleop can never feel it. Audio capture uses ``arecord`` (ALSA)
directly — no PortAudio system dependency.

Transcript CSV (one per session, in the session root — the room has ONE mic, so this is
room-level data, not per-robot):

    t_start_epoch, t_end_epoch, duration_s, speaker, confidence, take, text

``take`` is the manager-reported take label active when the utterance STARTED (or
"between" outside takes), so utterances align with attempts; all timestamps are
``time.time()`` — the same clock as every other recorded stream.

Speaker identification is ENROLLMENT-BASED: each person (participant1, participant2,
coordinator) reads ~10 s at calibration time; utterances are labeled by cosine
similarity to the enrolled embeddings, with a confidence column (below --min-speaker-sim
the label is "unknown" for easy post-hoc fixes).

Standalone test:  python study_transcriber.py --test           # live mic -> console
Protocol (serve): init / start_session / set_take / stop_session / enroll /
                  enroll_clear / shutdown ; status @1 Hz.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

SR = 16000                      # capture sample rate (silero + whisper native)
FRAME = 512                     # silero streaming frame (exactly 512 samples @16 kHz)
ENROLL_MAX_S = 600.0            # safety cap for open-ended enrollment (a forgotten Stop
                                # can't grow the capture buffer without bound)
PRE_ROLL_S = 0.30               # audio kept before speech onset
END_SILENCE_S = 0.80            # utterance ends after this much silence
MAX_UTTER_S = 28.0              # hard cap (whisper window is 30 s)
MIN_UTTER_S = 0.35              # discard blips shorter than this


# ── IPC helpers (numpy-safe, length-prefixed JSON — mirrors apriltag_tracker) ──
def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _send_msg(sock, obj):
    blob = json.dumps(obj, default=_json_default).encode("utf-8")
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("transcriber IPC closed")
        buf += chunk
    return buf


def _recv_msg(sock):
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


# ── microphone discovery ─────────────────────────────────────────────────────
def find_mic_device(hint: str | None = None) -> str | None:
    """Pick the ALSA capture device: an explicit hint ('plughw:...'), else the first
    USB capture card that is NOT the laptop's built-in array (sof-soundwire)."""
    if hint and hint != "auto":
        return hint
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return None
    for m in re.finditer(r"card \d+: (\w+) \[([^\]]+)\], device (\d+):", out):
        card_id, card_name, dev = m.group(1), m.group(2), m.group(3)
        if "sofsoundwire" in card_id.lower() or "sof-soundwire" in card_name.lower():
            continue                                   # built-in mic array
        return f"plughw:CARD={card_id},DEV={dev}"
    return None


# ── PipeWire-aware capture routing ───────────────────────────────────────────
# On a PipeWire system, PipeWire OWNS the ALSA cards, so raw hw access
# (arecord -D plughw:...) fails with "device busy". Capture must instead go THROUGH
# PipeWire's shared 'pipewire' PCM, targeting the specific source node for that card.
def _alsa_card_number(plughw: str | None):
    """Card NUMBER for a 'plughw:CARD=NAME,…' string (via /proc/asound/NAME → cardN)."""
    m = re.search(r"CARD=([^,]+)", plughw or "")
    if not m:
        return None
    try:
        real = os.path.realpath(f"/proc/asound/{m.group(1)}")
        cm = re.search(r"card(\d+)", real)
        return int(cm.group(1)) if cm else None
    except Exception:
        return None


def _pipewire_source_for_card(card_no):
    """The PipeWire source node.name backing ALSA card ``card_no`` (via pw-dump), or None if
    PipeWire/pw-dump aren't present. Matches on the node's ``alsa.card`` property."""
    if card_no is None:
        return None
    try:
        objs = json.loads(subprocess.run(["pw-dump"], capture_output=True,
                                         text=True, timeout=5).stdout)
    except Exception:
        return None
    for o in objs:
        p = (o.get("info") or {}).get("props") or {}
        if (p.get("media.class") == "Audio/Source"
                and str(p.get("node.name", "")).startswith("alsa_input")
                and str(p.get("alsa.card")) == str(card_no)):
            return p.get("node.name")
    return None


def resolve_capture(detected: str | None):
    """Map a detected raw device ('plughw:CARD=…') to what arecord should actually use.
    Returns (arecord_device, env_dict_or_None): on PipeWire → ('pipewire', {PIPEWIRE_NODE:…})
    so capture shares the device (no 'busy') and targets the SPECIFIC card; otherwise the
    raw device unchanged (env None)."""
    node = _pipewire_source_for_card(_alsa_card_number(detected))
    if node:
        return "pipewire", {"PIPEWIRE_NODE": node}
    return detected, None


# ── the transcription engine ─────────────────────────────────────────────────
class Transcriber:
    """Capture → VAD segmentation → whisper → speaker-ID → CSV rows."""

    def __init__(self, device=None, model_size="small.en", enroll_path=None,
                 min_speaker_sim=0.45, min_speaker_margin=0.06, stereo=False, log=print):
        self.log = log
        self.device = find_mic_device(device)
        self.model_size = model_size
        # Capture mode. "mono": ONE conference mic → VAD → whisper → voice-ID (needs
        # Calibrate Voices). "stereo": a 2-channel device (e.g. Hollyland Lark M1 in
        # dual-channel mode) → LEFT = operator1, RIGHT = operator2; each channel is
        # segmented + transcribed independently and attributed BY CHANNEL — no voice-ID or
        # enrollment, and talking over each other doesn't corrupt attribution.
        self.stereo = bool(stereo)
        self.mode = "stereo" if self.stereo else "mono"
        self.channels = 2 if self.stereo else 1
        # Fixed speaker per channel in stereo; None in mono (→ run voice-ID).
        self._channel_speaker = ["operator1", "operator2"] if self.stereo else [None]
        # Speaker-ID decision: label the nearest enrolled voice only if its cosine clears
        # ``min_speaker_sim`` (absolute floor, rejects noise/wrong-room) AND beats the
        # runner-up by ``min_speaker_margin`` (relative separation — the reliable signal for
        # WHICH operator it is when far-field/short-utterance embeddings all sit ~0.4-0.6).
        self.min_speaker_sim = float(min_speaker_sim)
        self.min_speaker_margin = float(min_speaker_margin)
        self.enroll_path = str(enroll_path or (_HERE / "voice_profiles.json"))

        self._stop = threading.Event()
        # One capture queue PER channel (the interleaved stereo stream is deinterleaved in
        # _capture_loop). Mono = a single-element list, so the mono path is unchanged.
        self._chunks = [queue.Queue(maxsize=400) for _ in range(self.channels)]
        # UNBOUNDED on purpose: with an accuracy-first (slower-than-realtime) model, finished
        # utterances buffer here and are transcribed in order — NEVER dropped. Each entry is
        # (audio, t0, t1, speaker) where speaker is the fixed channel operator (stereo) or
        # None → run voice-ID (mono). Backlog is exposed in status(); memory is bounded by
        # talk-time × ~64 KB/s per channel; audio never touches disk.
        self._utts: "queue.Queue[tuple[np.ndarray, float, float, object]]" = queue.Queue()
        self._arec: subprocess.Popen | None = None

        # session / take state (set over IPC; guarded by _lock)
        self._lock = threading.Lock()
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        self._take = "between"
        # Only utterances spoken WHILE an attempt is recording are saved. The manager sets
        # this True at attempt-record start and False at attempt stop; utterances are tagged
        # with it at speech onset (below), so whisper's lag can't misattribute a row.
        self._recording = False

        # live status
        self.rms = 0.0
        self.n_segments = 0
        self.last_text = ""
        self.last_speaker = ""
        self.listening = False
        self.error = None
        # Serve mode sets this to stream each FINISHED transcript row to the manager UI
        # (the live-transcription window); None = no consumer (e.g. the standalone --test).
        self.on_transcript = None

        # enrollment: {name: 256-d embedding}
        self._profiles: dict[str, np.ndarray] = {}
        self._load_profiles()
        # one-shot enrollment capture request: (name, seconds, reply_cb). ``seconds`` is
        # None for OPEN-ENDED capture that runs until ``_enroll_stop`` is set (the
        # operator's Stop button); a number keeps the old fixed-length grab.
        self._enroll_req = None
        self._enroll_stop = False

        # models (loaded in start(): heavy)
        self._whisper = None
        self._vad_iter = None
        self._encoder = None
        self._preprocess = None    # resemblyzer.preprocess_wav (loudness-norm + VAD trim)

    @staticmethod
    def _preload_cuda12_libs():
        """CTranslate2 dlopens libcublas.so.12 + libcudnn*.so.9 lazily at FIRST
        inference — with a cu13 torch those aren't on the loader path, so GPU whisper
        loads fine then dies at runtime. Preload the pip-provided cu12 libs
        (nvidia-cublas-cu12 / nvidia-cudnn-cu12) with RTLD_GLOBAL so the lazy dlopen
        resolves."""
        import ctypes
        import site
        roots = []
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            roots.append(Path(sp) / "nvidia")
        roots.append(Path(sys.prefix) / "lib"
                     / ("python%d.%d" % sys.version_info[:2])
                     / "site-packages" / "nvidia")
        seen = set()
        for root in roots:
            if not root.is_dir() or str(root) in seen:
                continue
            seen.add(str(root))
            for pat in ("cublas/lib/libcublas.so.12", "cublas/lib/libcublasLt.so.12",
                        "cudnn/lib/libcudnn.so.9"):
                so = root / pat
                if so.exists():
                    try:
                        ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        if self.device is None:
            self.error = "no microphone found (is the conference mic plugged in?)"
            self.log(f"[transcribe] {self.error}")
            return False
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad, VADIterator
        from resemblyzer import VoiceEncoder, preprocess_wav
        t0 = time.time()
        self._preload_cuda12_libs()
        try:
            self._whisper = WhisperModel(self.model_size, device="cuda",
                                         compute_type="float16")
            # WARM-UP PROBE: CT2's CUDA libs load lazily at first inference — force
            # that NOW so a broken GPU stack fails here and the CPU fallback engages
            # (without this, load "succeeds" and every real utterance errors).
            list(self._whisper.transcribe(np.zeros(SR // 2, np.float32),
                                          language="en", beam_size=1)[0])
            dev = "cuda/fp16"
        except Exception as exc:                         # Blackwell/driver fallback
            self.log(f"[transcribe] CUDA whisper unavailable ({exc}) — CPU int8")
            self._whisper = WhisperModel(self.model_size, device="cpu",
                                         compute_type="int8")
            dev = "cpu/int8"
        # One silero VAD instance PER channel — each keeps its own streaming state, loaded
        # fresh so the two channels' hidden states never collide.
        self._vad_iter = [VADIterator(load_silero_vad(), sampling_rate=SR)
                          for _ in range(self.channels)]
        self._encoder = VoiceEncoder("cpu", verbose=False)
        self._preprocess = preprocess_wav
        self.log(f"[transcribe] models ready in {time.time()-t0:.1f}s "
                 f"(whisper {self.model_size} on {dev}; {self.mode} {self.channels}ch); "
                 f"mic = {self.device}")
        threading.Thread(target=self._capture_loop, daemon=True,
                         name="transcribe-capture").start()
        for ch in range(self.channels):
            threading.Thread(target=self._segment_loop, args=(ch,), daemon=True,
                             name=f"transcribe-vad{ch}").start()
        threading.Thread(target=self._transcribe_loop, daemon=True,
                         name="transcribe-whisper").start()
        return True

    def stop(self):
        self._stop.set()
        try:
            if self._arec is not None:
                self._arec.terminate()
        except Exception:
            pass
        self.close_session()

    # -- audio capture (arecord → 512-sample chunks per channel) ---------------
    def _capture_loop(self):
        nch = self.channels
        bytes_per_frame = FRAME * 2 * nch                 # S16_LE, nch interleaved channels
        while not self._stop.is_set():
            # Route through PipeWire when it owns the card (raw hw access = 'busy'); env
            # carries PIPEWIRE_NODE so it targets THIS card, not just the default source.
            cap_dev, cap_env = resolve_capture(self.device)
            env = {**os.environ, **cap_env} if cap_env else None
            cmd = ["arecord", "-D", cap_dev, "-q", "-f", "S16_LE",
                   "-r", str(SR), "-c", str(nch), "-t", "raw"]
            try:
                self._arec = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                              stderr=subprocess.DEVNULL, env=env)
                self.listening = True
                self.error = None
                buf = b""
                while not self._stop.is_set():
                    data = self._arec.stdout.read(bytes_per_frame - len(buf))
                    if not data:
                        raise IOError("arecord stream ended")
                    buf += data
                    if len(buf) < bytes_per_frame:
                        continue
                    samples = np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0
                    buf = b""
                    now = time.time()                     # epoch of the chunk's END
                    if nch == 1:
                        chans = [samples]
                    else:                                 # de-interleave L R L R … → [L, R]
                        s = samples.reshape(-1, nch)
                        chans = [np.ascontiguousarray(s[:, c]) for c in range(nch)]
                    self.rms = float(np.sqrt(np.mean(samples ** 2)))
                    for c, chunk in enumerate(chans):
                        q = self._chunks[c]
                        try:
                            q.put_nowait((chunk, now))
                        except queue.Full:                # VAD stalled — drop oldest
                            try:
                                q.get_nowait()
                                q.put_nowait((chunk, now))
                            except Exception:
                                pass
            except Exception as exc:                      # mic unplugged etc.
                self.listening = False
                self.error = f"capture: {exc}"
                if self._stop.is_set():
                    return
                time.sleep(2.0)                           # retry (mic may come back)

    # -- VAD segmentation → utterances (one instance per channel) --------------
    def _segment_loop(self, ch=0):
        """VAD-segment ONE channel into utterances. ``ch`` indexes the per-channel chunk
        queue + VAD instance. Each utterance carries the channel's fixed speaker (stereo);
        in mono (speaker=None) this channel also runs voice enrollment and the transcribe
        loop does voice-ID instead."""
        chunks_q = self._chunks[ch]
        vad = self._vad_iter[ch]
        speaker = self._channel_speaker[ch]
        do_enroll = speaker is None                       # enrollment/voice-ID is mono-only
        pre_frames = int(PRE_ROLL_S * SR / FRAME)
        end_frames = int(END_SILENCE_S * SR / FRAME)
        max_frames = int(MAX_UTTER_S * SR / FRAME)
        ring: list[np.ndarray] = []
        utter: list[np.ndarray] = []
        utter_t0 = 0.0        # audio-segment start (incl. pre-roll) — for the padded clip
        speech_t0 = 0.0       # true speech onset (VAD start) — reported as the line's t0
        speech_t1 = 0.0       # true speech offset (last voiced frame) — reported as t1
        silence = 0
        in_speech = False
        speaking = False      # silero's speech/silence state, tracked from start/end events
        utter_rec = False     # was an attempt recording when THIS utterance began? (save gate)
        utter_take = "between"  # take label AT ONSET → the attempt/practice this row belongs to
        # Enrollment capture (mono only; bypasses VAD). Two modes, chosen by request[1]:
        #   • a number → fixed-length grab (legacy).  • None → OPEN-ENDED until stop_enroll().
        enroll_buf: list[np.ndarray] = []
        enroll_left = 0            # >0: fixed frames remaining
        enroll_open = False        # True: recording until stop / safety cap
        enroll_cap = int(ENROLL_MAX_S * SR / FRAME)

        while not self._stop.is_set():
            try:
                chunk, t_end = chunks_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if do_enroll:
                req = self._enroll_req
                if req is not None and enroll_left == 0 and not enroll_open and not enroll_buf:
                    if req[1] is None:
                        enroll_open = True                        # start open-ended capture
                    else:
                        enroll_left = int(float(req[1]) * SR / FRAME)   # fixed-length capture
                if enroll_open:
                    enroll_buf.append(chunk)
                    if self._enroll_stop or len(enroll_buf) >= enroll_cap:
                        capped = not self._enroll_stop
                        audio = (np.concatenate(enroll_buf) if enroll_buf
                                 else np.zeros(0, np.float32))
                        enroll_buf = []
                        enroll_open = False
                        self._finish_enroll(audio, capped=capped)
                    continue                               # enrollment audio is separate
                if enroll_left > 0:
                    enroll_buf.append(chunk)
                    enroll_left -= 1
                    if enroll_left == 0:
                        audio = np.concatenate(enroll_buf)
                        enroll_buf = []
                        self._finish_enroll(audio)
                    continue                               # enrollment audio is separate

            try:
                res = vad(chunk)                           # {'start':..}/{'end':..}/None
            except Exception:
                res = None
            is_start = bool(res and "start" in res)
            is_end = bool(res and "end" in res)

            if not in_speech:
                ring.append(chunk)
                if len(ring) > pre_frames:
                    ring.pop(0)
                if is_start:
                    in_speech = True
                    speaking = True
                    utter = list(ring)
                    ring = []
                    utter_t0 = t_end - len(utter) * FRAME / SR   # audio start (incl. pre-roll)
                    speech_t0 = t_end                            # true speech onset (VAD start)
                    speech_t1 = t_end                            # advances with each voiced frame
                    utter_rec = self._recording                  # save only if recording at onset
                    utter_take = self._take                       # label by the take at onset
                    silence = 0
            else:
                utter.append(chunk)
                # silero's VADIterator emits a dict ONLY at a start/end TRANSITION and returns
                # None on every frame in between — for ongoing SPEECH and ongoing silence alike.
                # So the speech/silence state must be tracked from the transitions: we're
                # "speaking" from a 'start' until the next 'end'. (The old code counted every
                # `res is None` frame as silence — i.e. treated ongoing speech as silence — so
                # `silence` grew to ~the whole utterance and speech_t1 = t_end - silence*dt
                # collapsed back to speech_t0, reporting a ~0 s window.)
                if is_start:
                    speaking = True
                elif is_end:
                    speaking = False
                if speaking:
                    speech_t1 = t_end                            # last voiced frame → speech offset
                    silence = 0
                else:
                    silence += 1                                 # count only REAL trailing silence
                # Close on silero's end-of-speech, an over-long buffer, or a long silent tail
                # (the tail guards a missed 'end'). t0/t1 now bracket exactly the voiced span, so
                # t1-t0 is how long the phrase took; the clip to Whisper keeps its pre-roll pad.
                if is_end or len(utter) >= max_frames or silence >= end_frames:
                    audio = np.concatenate(utter)
                    if len(audio) / SR >= MIN_UTTER_S:
                        # Unbounded queue → never drop; speaker fixed by channel (stereo) or
                        # None → voice-ID in the transcribe loop (mono). utter_rec gates saving;
                        # utter_take pins the row to the attempt/practice it was spoken in.
                        self._utts.put((audio, speech_t0, max(speech_t0, speech_t1),
                                        speaker, utter_rec, utter_take))
                    in_speech = False
                    speaking = False
                    utter = []
                    silence = 0

    # -- whisper + speaker-ID → CSV --------------------------------------------
    def _transcribe_loop(self):
        while not self._stop.is_set():
            try:
                audio, t0, t1, ch_speaker, rec, take = self._utts.get(timeout=0.5)
            except queue.Empty:
                continue
            # NOTE: every utterance is transcribed and shown LIVE (below), whether or not an
            # attempt is recording — the operator always sees the room transcript. `rec` (the
            # recording state captured at onset) gates ONLY whether the row is persisted to the
            # session CSV; setup / between-attempt speech is shown but not saved.
            try:
                segs, _info = self._whisper.transcribe(
                    audio, language="en", beam_size=5, vad_filter=False,
                    condition_on_previous_text=False)
                text = " ".join(s.text.strip() for s in segs).strip()
            except Exception as exc:                       # noqa: BLE001
                self.log(f"[transcribe] whisper error: {exc}")
                continue
            if not text or text in ("Thank you.",):        # whisper noise hallucination
                continue
            if ch_speaker is not None:                     # stereo → attributed BY CHANNEL
                speaker, conf = ch_speaker, 1.0
            else:                                          # mono → voice-ID
                speaker, conf = self._identify(audio)
            with self._lock:
                # SAVE gate: persist to the CSV only when an attempt/practice was recording at
                # onset (rec). `take` is the label captured AT ONSET (passed through _utts) — NOT
                # the live self._take — so a saved row is pinned to the attempt/practice it was
                # spoken in even if whisper finished after that take ended.
                w = self._csv_writer
                if w is not None and rec:
                    w.writerow([round(t0, 3), round(t1, 3), round(t1 - t0, 2),
                                speaker, round(conf, 3), take, text])
                    self._csv_file.flush()
            self.n_segments += 1
            self.last_text = text[:120]
            self.last_speaker = speaker
            # Transcript rows go ONLY to the Live Transcription window (on_transcript) and
            # the session CSV — never to the console/manager log, which stays clean.
            if self.on_transcript is not None:
                try:
                    self.on_transcript({"t0": round(t0, 3), "t1": round(t1, 3),
                                        "speaker": speaker, "conf": round(conf, 3),
                                        "take": take, "saved": bool(rec), "text": text})
                except Exception:
                    pass

    # -- speaker enrollment / identification ------------------------------------
    def _load_profiles(self):
        try:
            with open(self.enroll_path) as f:
                raw = json.load(f)
            self._profiles = {k: np.asarray(v, np.float32) for k, v in raw.items()}
        except Exception:
            self._profiles = {}

    def _save_profiles(self):
        with open(self.enroll_path, "w") as f:
            json.dump({k: v.tolist() for k, v in self._profiles.items()}, f)

    def _embed(self, audio):
        """Compute a speaker embedding the resemblyzer-recommended way: loudness-normalize
        (to -30 dBFS) + VAD-trim silence via preprocess_wav BEFORE embed_utterance. Skipping
        this made the embedding capture the CHANNEL (mic/room/loudness) instead of the voice
        — two DIFFERENT speakers scored ~0.69, so a re-recorded voice never cleared the 0.70
        threshold and everyone came out 'unknown'. Both enrollment and identification MUST
        use the same path. Falls back to raw audio if preprocessing trims it to almost
        nothing (very short utterance)."""
        wav = np.asarray(audio, np.float32)
        if self._preprocess is not None:
            try:
                pp = self._preprocess(wav, source_sr=SR)
                if pp is not None and len(pp) >= int(0.4 * SR):   # enough voiced audio
                    wav = pp
            except Exception:
                pass
        return np.asarray(self._encoder.embed_utterance(wav), np.float32)

    def request_enroll(self, name, seconds, done_cb):
        """Queue a voice-enrollment capture. ``seconds=None`` records OPEN-ENDED until
        ``stop_enroll()`` (the operator's Stop button); a number keeps a fixed grab."""
        self._enroll_stop = False
        self._enroll_req = (str(name), (None if seconds is None else float(seconds)),
                            done_cb)

    def stop_enroll(self):
        """End an in-progress OPEN-ENDED enrollment (the operator finished the script)."""
        self._enroll_stop = True

    def _finish_enroll(self, audio, capped=False):
        name, _secs, cb = self._enroll_req
        self._enroll_req = None
        self._enroll_stop = False
        dur = len(audio) / SR
        if len(audio) < int(0.5 * SR):                     # Stop pressed too early
            cb(False, f"too short ({dur:.1f}s) — record again and let them read a few "
                      "seconds before pressing Stop.")
            return
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 5e-4:
            cb(False, f"too quiet (rms {rms:.5f}) — is anyone speaking near the mic?")
            return
        try:
            self._profiles[name] = self._embed(audio)
            self._save_profiles()
            note = "  [hit the safety cap — pressed Stop?]" if capped else ""
            cb(True, f"enrolled '{name}' ({dur:.0f}s, rms {rms:.4f}){note}; "
                     f"profiles: {sorted(self._profiles)}")
        except Exception as exc:                           # noqa: BLE001
            cb(False, f"embedding failed: {exc}")

    def clear_profiles(self):
        self._profiles = {}
        try:
            os.remove(self.enroll_path)
        except Exception:
            pass

    def _identify(self, audio):
        if not self._profiles or self._encoder is None:
            return "unknown", 0.0
        try:
            emb = self._embed(audio)
        except Exception:
            return "unknown", 0.0
        sims = sorted(
            ((float(np.dot(emb, ref)
                    / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-9)), name)
             for name, ref in self._profiles.items()),
            reverse=True)
        best_sim, best = sims[0]
        second_sim = sims[1][0] if len(sims) > 1 else -1.0
        # Floor rejects noise / totally-wrong matches; margin ensures the winner is clearly
        # closer than the other operator (robust when all scores are low). With one enrolled
        # voice the margin is trivially met, so the floor alone decides.
        if best_sim < self.min_speaker_sim or (best_sim - second_sim) < self.min_speaker_margin:
            return "unknown", best_sim
        return best, best_sim

    # -- session / take plumbing -------------------------------------------------
    _FIELDS = ["t_start_epoch", "t_end_epoch", "duration_s",
               "speaker", "confidence", "take", "text"]

    def open_session(self, session_dir, session_id):
        self.close_session()
        os.makedirs(session_dir, exist_ok=True)
        path = os.path.join(session_dir, f"transcript_{session_id}.csv")
        new = not os.path.exists(path)
        with self._lock:
            self._csv_file = open(path, "a", newline="")   # append → resume-safe
            self._csv_writer = csv.writer(self._csv_file)
            if new:
                self._csv_writer.writerow(self._FIELDS)
                self._csv_file.flush()
            self._csv_path = path
            self._take = "between"
            self._recording = False       # armed, not recording — nothing saved until an attempt
        self.log(f"[transcribe] session transcript → {os.path.basename(path)}")

    def set_take(self, label):
        with self._lock:
            self._take = str(label or "between")

    def set_recording(self, on):
        """Gate saving: only speech captured while this is True (an attempt is recording) is
        transcribed + written to the session CSV."""
        with self._lock:
            self._recording = bool(on)

    def close_session(self):
        with self._lock:
            if self._csv_file is not None:
                try:
                    self._csv_file.close()
                except Exception:
                    pass
            self._csv_file = None
            self._csv_writer = None
            self._csv_path = None
            self._take = "between"
            self._recording = False

    def status(self):
        return {
            "listening": self.listening, "device": self.device,
            "model": self.model_size, "rms": round(self.rms, 5),
            "n_segments": self.n_segments, "last_text": self.last_text,
            "last_speaker": self.last_speaker,
            "enrolled": sorted(self._profiles),
            "session_open": self._csv_path is not None,
            "csv": self._csv_path, "error": self.error,
            # utterances still queued for whisper — how far processing is behind realtime.
            "backlog": self._utts.qsize(),
            "mode": self.mode, "channels": self.channels,
        }


# ── serve mode (the manager's subprocess) ────────────────────────────────────
def serve(fd):
    sock = socket.socket(fileno=fd)
    send_lock = threading.Lock()

    def safe_send(obj):
        with send_lock:
            try:
                _send_msg(sock, obj)
            except Exception:
                pass

    def log(msg):
        safe_send({"log": str(msg)})

    try:
        init = _recv_msg(sock).get("init", {})
    except Exception:
        return
    tr = Transcriber(device=init.get("device", "auto"),
                     model_size=init.get("model", "small.en"),
                     enroll_path=init.get("enroll_path"),
                     min_speaker_sim=float(init.get("min_speaker_sim", 0.45)),
                     min_speaker_margin=float(init.get("min_speaker_margin", 0.06)),
                     stereo=bool(init.get("stereo", False)),
                     log=log)
    # Stream each finished utterance to the manager for its live-transcription window.
    tr.on_transcript = lambda row: safe_send({"transcript": row})
    ok = tr.start()
    safe_send({"hello": {"ok": ok, "device": tr.device, "error": tr.error}})

    stop = threading.Event()

    def _status_loop():
        while not stop.is_set():
            safe_send({"status": tr.status()})
            stop.wait(1.0)
    threading.Thread(target=_status_loop, daemon=True).start()

    try:
        while True:
            msg = _recv_msg(sock)
            cmd = msg.get("cmd")
            rid = msg.get("id")
            try:
                if cmd == "shutdown":
                    break
                elif cmd == "start_session":
                    tr.open_session(msg["dir"], msg["session_id"])
                    safe_send({"reply": rid, "ok": True})
                elif cmd == "set_take":
                    tr.set_take(msg.get("label"))
                elif cmd == "set_recording":
                    tr.set_recording(msg.get("on"))
                elif cmd == "stop_session":
                    tr.close_session()
                    safe_send({"reply": rid, "ok": True})
                elif cmd == "enroll":
                    def _done(okk, text, rid=rid):
                        safe_send({"reply": rid, "ok": bool(okk), "msg": str(text)})
                    tr.request_enroll(msg.get("name", "speaker"),
                                      float(msg.get("seconds", 12.0)), _done)
                elif cmd == "enroll_start":
                    # OPEN-ENDED capture: records until an enroll_stop arrives. The reply
                    # (embedding result) is sent then, tagged with THIS start's id.
                    def _done(okk, text, rid=rid):
                        safe_send({"reply": rid, "ok": bool(okk), "msg": str(text)})
                    tr.request_enroll(msg.get("name", "speaker"), None, _done)
                elif cmd == "enroll_stop":
                    tr.stop_enroll()               # finish → replies on the start's id
                elif cmd == "enroll_clear":
                    tr.clear_profiles()
                    safe_send({"reply": rid, "ok": True})
            except Exception as exc:                       # keep serving on error
                if rid is not None:
                    safe_send({"reply": rid, "ok": False, "error": str(exc)})
                log(f"[transcribe] command {cmd!r} failed: {exc}")
    except Exception:
        pass
    finally:
        stop.set()
        tr.stop()
        try:
            sock.close()
        except Exception:
            pass


# ── standalone test mode ─────────────────────────────────────────────────────
def _test(args):
    tr = Transcriber(device=args.device, model_size=args.model)
    if not tr.start():
        sys.exit(1)
    out = Path(args.out or ".").resolve()
    tr.open_session(str(out), "mic_test")
    print(f"listening on {tr.device} — speak; transcript → {tr._csv_path}; Ctrl+C to stop")
    try:
        while True:
            time.sleep(2)
            s = tr.status()
            print(f"  rms={s['rms']:.4f} segments={s['n_segments']} "
                  f"last=[{s['last_speaker']}] {s['last_text'][:60]}")
    except KeyboardInterrupt:
        tr.stop()


def main():
    p = argparse.ArgumentParser(description="Study transcriber (whisper + speaker-ID).")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--fd", type=int, default=None)
    p.add_argument("--test", action="store_true", help="standalone live-mic test")
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default="small.en")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.serve:
        serve(args.fd)
    elif args.test:
        _test(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
