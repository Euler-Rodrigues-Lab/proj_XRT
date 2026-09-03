#!/usr/bin/env python3
"""Diagnose and clear a latched Dynamixel hardware error on the Ostrich neck.

When a neck servo trips a hardware fault (overload, overheating, input-voltage,
encoder, or electrical-shock), it sets a bit in its Hardware Error Status
register (addr 70), auto-disables torque, and then REFUSES every torque-enable
until the latch is cleared. Restarting XRT_user_study_manager.py does not clear
it because the latch lives in the motor firmware, not the host -- you see:

    Neck unavailable: Torque enable error for ID 11: [RxPacketError] Hardware
    error occurred. Check the error at Control Table (Hardware Error Status)!

This tool reads the latched fault for each neck motor, prints present voltage and
temperature (to point at the root cause), issues a software REBOOT to clear the
latch, then re-reads the status and (optionally) re-enables torque to confirm
recovery.

Usage:
    python neck_recover.py                 # diagnose + reboot motors 11 and 12
    python neck_recover.py --no-reboot     # diagnose only, do not clear/reboot
    python neck_recover.py --ids 11        # only the yaw motor
    python neck_recover.py --port /dev/ttyUSB0 --baud 4000000
"""
from __future__ import annotations

import argparse
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# X-series control table (Protocol 2.0) -- matches DynamixelNeckController.
PROTOCOL_VERSION = 2.0
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_PRESENT_VOLTAGE = 144   # 2 bytes, units of 0.1 V
ADDR_PRESENT_TEMPERATURE = 146  # 1 byte, deg C

# Hardware Error Status bit meanings (addr 70) for X-series servos.
HW_ERROR_BITS = [
    (0x01, "Input Voltage Error  -> supply too low/high; check the power rail & connector"),
    (0x04, "Overheating Error    -> motor ran too hot; let it cool, reduce hold torque/duty"),
    (0x08, "Motor Encoder Error  -> encoder fault; check wiring/seating, may be failing motor"),
    (0x10, "Electrical Shock     -> over-current spike; check for shorts/loose power"),
    (0x20, "Overload Error       -> stalled or fought a hard stop / cable tension too long"),
]


def decode_hw_error(status: int) -> list[str]:
    if status == 0:
        return []
    hits = [desc for bit, desc in HW_ERROR_BITS if status & bit]
    if not hits:
        hits.append(f"Unknown error bits set (0x{status:02X})")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=4_000_000)
    ap.add_argument("--ids", type=int, nargs="+", default=[11, 12],
                    help="motor IDs to check (default: 11 yaw, 12 pitch)")
    ap.add_argument("--no-reboot", action="store_true",
                    help="diagnose only; do not reboot/clear the latch")
    ap.add_argument("--enable-after", action="store_true",
                    help="after reboot, re-enable torque to confirm full recovery")
    args = ap.parse_args()

    port = PortHandler(args.port)
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        print(f"ERROR: failed to open {args.port}", file=sys.stderr)
        return 1
    if not port.setBaudRate(args.baud):
        print(f"ERROR: failed to set baud {args.baud}", file=sys.stderr)
        port.closePort()
        return 1

    print(f"Opened {args.port} @ {args.baud} baud, protocol {PROTOCOL_VERSION}\n")
    any_unreachable = False

    try:
        for mid in args.ids:
            print(f"=== Motor ID {mid} ===")

            status, comm, derr = packet.read1ByteTxRx(port, mid, ADDR_HARDWARE_ERROR_STATUS)
            if comm != COMM_SUCCESS:
                any_unreachable = True
                print(f"  UNREACHABLE: {packet.getTxRxResult(comm)}")
                print("  -> wrong port/baud, bad cable, or motor unpowered.\n")
                continue

            volt, _, _ = packet.read2ByteTxRx(port, mid, ADDR_PRESENT_VOLTAGE)
            temp, _, _ = packet.read1ByteTxRx(port, mid, ADDR_PRESENT_TEMPERATURE)
            print(f"  Present voltage: {volt / 10.0:.1f} V")
            print(f"  Present temp:    {temp} C")
            print(f"  Hardware Error Status (addr 70): 0x{status:02X}")

            faults = decode_hw_error(status)
            if faults:
                print("  LATCHED FAULT(S):")
                for f in faults:
                    print(f"    - {f}")
            else:
                print("  No latched hardware fault.")

            if args.no_reboot:
                print()
                continue

            if status == 0 and not args.enable_after:
                print("  Nothing to clear.\n")
                continue

            print("  Rebooting motor to clear the latch...")
            comm, derr = packet.reboot(port, mid)
            if comm != COMM_SUCCESS:
                print(f"    reboot failed: {packet.getTxRxResult(comm)}\n")
                continue
            time.sleep(0.5)  # firmware needs a moment to come back up

            status2, comm2, _ = packet.read1ByteTxRx(port, mid, ADDR_HARDWARE_ERROR_STATUS)
            if comm2 != COMM_SUCCESS:
                print(f"    post-reboot read failed: {packet.getTxRxResult(comm2)}\n")
                continue
            if status2 == 0:
                print("    Latch CLEARED (status 0x00).")
            else:
                print(f"    STILL FAULTED (0x{status2:02X}) -> the fault condition is still "
                      "present; fix the root cause (load/voltage/heat) before retrying.")
                for f in decode_hw_error(status2):
                    print(f"      - {f}")

            if args.enable_after:
                comm3, derr3 = packet.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 1)
                if comm3 == COMM_SUCCESS and derr3 == 0:
                    print("    Torque enable OK -- motor is ready.")
                    packet.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, 0)  # leave it as we found it
                else:
                    msg = packet.getRxPacketError(derr3) if derr3 else packet.getTxRxResult(comm3)
                    print(f"    Torque enable still failing: {msg}")
            print()
    finally:
        port.closePort()

    if any_unreachable:
        print("One or more motors were unreachable -- check that the neck power supply "
              "is on and the U2D2/USB cable is seated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
