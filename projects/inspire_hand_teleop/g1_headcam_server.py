#!/usr/bin/env python3
"""
g1_headcam_server.py — RUN THIS ON THE UNITREE G1's ONBOARD COMPUTER (the Jetson).

The G1's head Intel RealSense D435i is wired to the Jetson, not your laptop, and
the robot does NOT expose it through the SDK VideoClient (GetImageSample returns
error 3102). So this thin server opens the head camera ON the Jetson (where it's a
real USB device, with factory intrinsics) and streams its color frames + the
intrinsics to the laptop over TCP. The laptop then does the AprilTag detection and
base self-localization — see head_cam_localize.py.

Wire protocol (length-prefixed: big-endian uint32 length, then payload):
  message 1   : JSON intrinsics {fx,fy,ppx,ppy,width,height,model,coeffs,serial}
  message 2..N: a JPEG-encoded BGR color frame

Only reads/streams while a client is connected, so it's idle otherwise.

Deploy:
  scp g1_headcam_server.py unitree@<jetson-ip>:~/
  # on the Jetson:
  python3 g1_headcam_server.py                      # 1280x720@30, port 5005
  python3 g1_headcam_server.py --width 848 --height 480 --fps 30 --port 5005

Python 3.8 compatible. Needs pyrealsense2 + opencv (cv2) on the Jetson.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not found on the Jetson.")
try:
    import cv2
except ImportError:
    sys.exit("opencv (cv2) not found on the Jetson — needed for JPEG encoding. "
             "Try: pip3 install opencv-python-headless")


def send_msg(conn, payload):
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def main():
    p = argparse.ArgumentParser(description="G1 head-camera TCP frame server (Jetson).")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default all interfaces)")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", default=None, help="RealSense serial (default: the only one)")
    p.add_argument("--jpeg-quality", type=int, default=85)
    args = p.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(cfg)
    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    intrinsics = {
        "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
        "width": intr.width, "height": intr.height,
        "model": str(intr.model), "coeffs": list(intr.coeffs),
        "serial": args.serial or "",
    }
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    print(f"RealSense color {args.width}x{args.height}@{args.fps}  "
          f"fx={intr.fx:.1f} fy={intr.fy:.1f} ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    intr_blob = json.dumps(intrinsics).encode("utf-8")

    # Multi-client: one capture loop grabs each frame once and broadcasts it to
    # every connected laptop (the localizer AND the crate tracker both connect).
    clients = []   # [{sock, addr, need_intr}]
    clients_lock = threading.Lock()

    def accept_loop():
        while True:
            try:
                conn, addr = srv.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with clients_lock:
                clients.append({"sock": conn, "addr": addr, "need_intr": True})
                total = len(clients)
            print(f"Client connected: {addr}  ({total} total)")

    threading.Thread(target=accept_loop, daemon=True).start()
    print(f"Serving head camera on {args.host}:{args.port} — multi-client, "
          "waiting for laptop(s)...")

    n = 0
    t0 = time.time()
    try:
        while True:
            with clients_lock:
                have_clients = bool(clients)
            if not have_clients:
                time.sleep(0.1)        # idle when nobody is connected
                continue
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            ok, jpg = cv2.imencode(".jpg", img, enc)
            if not ok:
                continue
            payload = jpg.tobytes()
            with clients_lock:
                dead = []
                for c in clients:
                    try:
                        if c["need_intr"]:
                            send_msg(c["sock"], intr_blob)
                            c["need_intr"] = False
                        send_msg(c["sock"], payload)
                    except Exception as exc:
                        print(f"Client {c['addr']} dropped ({exc})")
                        dead.append(c)
                for c in dead:
                    clients.remove(c)
                    try:
                        c["sock"].close()
                    except Exception:
                        pass
                n_clients = len(clients)
            n += 1
            if n % 300 == 0:
                print(f"  streaming… {n / (time.time() - t0):.1f} fps to {n_clients} client(s)")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        srv.close()
        with clients_lock:
            for c in clients:
                try:
                    c["sock"].close()
                except Exception:
                    pass
        print("Stopped.")


if __name__ == "__main__":
    main()
