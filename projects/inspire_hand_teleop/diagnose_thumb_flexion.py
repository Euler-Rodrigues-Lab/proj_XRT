#!/usr/bin/env python3
"""Thumb flexion/extension diagnostic for the Inspire hands.

WHY: teleop thumb flexion (bend / curl toward the palm) stopped working. This isolates HARDWARE
(a dead/stalled/decoupled thumb actuator) from SOFTWARE (retargeting or index mapping) by talking
to the hands DIRECTLY over Modbus — bypassing the manager and the whole IK/retargeting stack.

The Inspire thumb has TWO actuators, and they are easy to confuse by eye:
    idx 4  THUMB-BEND (flexion)    — curls the thumb toward the palm/fingers
    idx 5  THUMB-ROTATION          — swings the thumb across the palm (looks like "back/forth")
(full actuator order: 0 pinky  1 ring  2 middle  3 index  4 thumb-bend  5 thumb-rotation)

This sweeps EACH thumb actuator on its own, holding everything else open, and for each reports
whether the measured driver position actually tracked the command, the current draw, and any
latched ERROR. Watch which sweep makes the thumb CURL INWARD.

  • BEND sweep curls the thumb in + position tracks + normal current -> actuator HEALTHY; the
    teleop problem is upstream SOFTWARE (thumb bend not being detected/retargeted).
  • BEND sweep: motor position tracks but the thumb does NOT curl -> HARDWARE mechanical
    decoupling (slipped/snapped tendon or stripped gear) — classic wear failure.
  • BEND sweep: position does NOT track -> HARDWARE motor (current high+stuck = jam/stall;
    current ~0 = open circuit / no drive). ERROR non-zero = latched fault.
  • The ROTATION sweep (not the bend one) is what curls the thumb in -> SOFTWARE index mismatch.

SAFETY: this MOVES the thumbs. Clear the hands, keep fingers clear. Run with the manager STOPPED
(two Modbus clients on one hand fight). Hands/robot must be powered.

    python projects/inspire_hand_teleop/diagnose_thumb_flexion.py            # both hands
    python projects/inspire_hand_teleop/diagnose_thumb_flexion.py --hand r   # right only
    python projects/inspire_hand_teleop/diagnose_thumb_flexion.py --no-sweep # read registers only
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from inspire_hand_hw_controller import InspireHandController  # noqa: E402

BEND = 4                      # thumb flexion/extension actuator index
ROT = 5                       # thumb rotation actuator index
NAMES = ["pinky", "ring", "middle", "index", "THUMB-BEND(flex)", "THUMB-ROTATION"]
MOVE_THRESH = 150             # min actual-position travel (counts) to call an actuator "tracking"


def _client_for(ctrl, side):
    return ctrl.client_left if side == "Left" else ctrl.client_right


def _read(ctrl, side):
    """Read one hand's joint-state registers; returns the dict or None if unreachable."""
    return ctrl._read_joint_states_once(_client_for(ctrl, side))


def _print_state(side, st):
    if not st:
        print(f"  [{side}] NO RESPONSE — hand unreachable (powered off? wrong IP? manager still "
              "running and holding the connection?).")
        return False
    pos, cur = st.get("POS_ACT", [0] * 6), st.get("CURRENT", [0] * 6)
    err, sts, tmp = st.get("ERROR", [0] * 6), st.get("STATUS", [0] * 6), st.get("TEMP", [0] * 6)
    print(f"  [{side}]  idx name              POS_ACT  CURRENT  ERROR  STATUS  TEMP")
    for i in range(6):
        mark = "  <-- flexion" if i == BEND else ("  <-- rotation" if i == ROT else "")
        print(f"        {i}  {NAMES[i]:<16} {pos[i]:>7} {cur[i]:>8} {err[i]:>6} {sts[i]:>7} "
              f"{tmp[i]:>5}{mark}")
    return True


def sweep_actuator(ctrl, side, idx, hold_pose):
    """Sweep actuator `idx` across its range (a few cycles) while holding the rest at hold_pose.
    Returns (travel, cur_max, err_max) from the measured registers."""
    hand = "l" if side == "Left" else "r"
    mode = ctrl.ActStates.POSITION
    print(f"\n  {'▶' * 3} [{side}] MOVING {NAMES[idx]} (actuator {idx}) — WATCH THIS THUMB NOW "
          f"{'◀' * 3}")
    print("        (does the thumb CURL toward the palm, or just swing/wag to the side?)")
    observed, currents, errs = [], [], []
    for target in (0, 2000, 0, 2000, 0):
        cmd = list(hold_pose)
        cmd[idx] = target
        ctrl._send_command(mode, hand=hand, pos_set=cmd)
        time.sleep(1.0)                                  # let it drive + settle before reading
        st = _read(ctrl, side)
        pa = st.get("POS_ACT", [0] * 6)[idx] if st else None
        cu = abs(st.get("CURRENT", [0] * 6)[idx]) if st else None
        er = st.get("ERROR", [0] * 6)[idx] if st else None
        if pa is not None:
            observed.append(pa)
        if cu is not None:
            currents.append(cu)
        if er is not None:
            errs.append(er)
        print(f"        cmd POS_SET={target:<5} -> POS_ACT={pa}   CURRENT={cu}   ERROR={er}")
    travel = (max(observed) - min(observed)) if observed else 0
    cmax = max(currents) if currents else 0
    emax = max(errs) if errs else 0
    print(f"        → position travel={travel} counts (tracking if ≥{MOVE_THRESH}), "
          f"peak current={cmax}, max ERROR={emax}")
    return travel, cmax, emax


def main():
    ap = argparse.ArgumentParser(description="Inspire thumb flexion hardware/software diagnostic")
    ap.add_argument("--hand", choices=["l", "r", "b"], default="b", help="which hand(s)")
    ap.add_argument("--no-sweep", action="store_true", help="read registers only; do NOT move")
    args = ap.parse_args()

    sides = {"l": ["Left"], "r": ["Right"], "b": ["Left", "Right"]}[args.hand]
    print("Connecting to the Inspire hand(s) over Modbus (192.168.123.210/211:6000)…")
    print("NOTE: stop the manager first — it holds these connections. Hands must be powered.\n")
    ctrl = InspireHandController(hand_side="b", initialize_dds=False, haptics_only=True)
    if not args.no_sweep:
        ctrl._apply_max_speed()                          # sweep at full, visible speed
        time.sleep(0.1)

    print("=== BASELINE actuator states ===")
    base = {}
    for side in sides:
        st = _read(ctrl, side)
        base[side] = st
        _print_state(side, st)
        print()

    results = {}
    if not args.no_sweep:
        # Hold everything OPEN so both thumb motions are unobstructed and easy to see.
        open_pose = [0, 0, 0, 0, 0, 0]
        for side in sides:
            if not base.get(side):
                continue
            # Test the BEND (flexion) actuator with the thumb rotated OUT (rotate=0) so a curl
            # toward the palm is obvious; then the ROTATION actuator with the bend straight.
            bend = sweep_actuator(ctrl, side, BEND, open_pose)
            rot = sweep_actuator(ctrl, side, ROT, open_pose)
            ctrl._send_command(ctrl.ActStates.POSITION,
                               hand=("l" if side == "Left" else "r"), pos_set=open_pose)
            results[side] = {"bend": bend, "rot": rot}

    print("\n" + "=" * 76)
    print("READING THE RESULT (per hand):")
    for side in sides:
        r = results.get(side)
        if not r:
            continue
        b_travel, b_cur, b_err = r["bend"]
        print(f"\n  [{side}]")
        print(f"    THUMB-BEND(flex): travel={b_travel}, peak-current={b_cur}, ERROR={b_err}")
        print(f"    THUMB-ROTATION:   travel={r['rot'][0]}, peak-current={r['rot'][1]}, "
              f"ERROR={r['rot'][2]}")
        if b_err:
            print("    → HARDWARE: the bend actuator has a LATCHED error (over-current/temp/stall).")
        elif b_travel >= MOVE_THRESH:
            print("    → The bend MOTOR + encoder track the command. If the thumb still didn't")
            print("      physically CURL during the bend sweep, that's HARDWARE mechanical")
            print("      decoupling (slipped/snapped tendon or stripped gear). If it DID curl,")
            print("      the actuator is fine and the teleop issue is SOFTWARE upstream.")
        else:
            hint = ("jam/stall" if b_cur > 300 else "open circuit / no drive")
            print(f"    → HARDWARE: bend motor did NOT track (travel {b_travel}; {hint}).")
    print("=" * 76)
    print("\nTELL ME, per hand: (1) did the BEND sweep or the ROTATION sweep curl the thumb toward")
    print("the palm? (2) paste these travel/current/ERROR numbers. That pins hardware vs software.")


if __name__ == "__main__":
    main()
