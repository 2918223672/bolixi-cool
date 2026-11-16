#!/usr/bin/env python3
"""Repeat a closed-loop no-drag/drag comparison against a private Godot copy."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import numpy as np
from PIL import Image


AIM = Path(__file__).resolve().parents[1]
SOURCE_SIM = Path(os.environ.get("RM_AIM_SIM", "/home/chen/Desktop/rm_aim_sim"))
OUTPUT = AIM / "logs" / "drag_compare_20261109"
SIM = OUTPUT / "simulator"
BINARY = AIM / "build_drag_compare" / "auto_aim_sim"
CHANNEL = "drag_compare"
DRAG_K = 0.020  # Illustrative 1/m; not a measured coefficient for this robot.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not SIM.exists():
        subprocess.run(
            ["rsync", "-a", "--exclude", ".git", "--exclude", ".tools",
             str(SOURCE_SIM) + "/", str(SIM) + "/"], check=True)
        subprocess.run(["patch", "-p1", "-i", str(AIM / "tests" / "drag_sim.patch")],
                       cwd=SIM, check=True)
    if "drag_k" not in (SIM / "scripts" / "settings.gd").read_text():
        raise RuntimeError("The private simulator copy has no drag model")
    (SIM / ".tools").mkdir(exist_ok=True)
    godot = SIM / ".tools" / "godot"
    if not godot.exists():
        godot.symlink_to(SOURCE_SIM / ".tools" / "godot")

    original = (SOURCE_SIM / "configs" / "timed_static_1000.cfg").read_text()
    # Keep the documented simulator's 4 m target distance: this model is not
    # reliably recognized by the current detector at 6 m.
    original = original.replace('channel="timed_test"', f'channel="{CHANNEL}"')
    original = original.replace("gravity=9.7833", f"gravity=9.7833\ndrag_k={DRAG_K:.3f}")
    changes = {
        "static": "translate=false\nrotate=false\n",
        "spin": "translate=false\nrotate=true\nomega=4.0\n",
        "move": ("translate=true\nspeed=1.0\nacceleration=3.0\n"
                 "heading=1.5707963267948966\ntravel=1.5\nrotate=false\n"),
        "move_spin": ("translate=true\nspeed=1.0\nacceleration=3.0\n"
                      "heading=1.5707963267948966\ntravel=1.5\nrotate=true\nomega=4.0\n"),
    }
    for name, replacement in changes.items():
        scene = original.replace("translate=false\nrotate=false\n", replacement)
        (OUTPUT / f"{name}.cfg").write_text(scene)
    vision = (AIM / "configs" / "sim.yaml").read_text()
    for mode, k in (("baseline", 0.0), ("drag", DRAG_K)):
        (OUTPUT / f"{mode}.yaml").write_text(vision + f"\ndrag_k: {k:.3f}\n")


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def capture_simulator_frame(output_dir, label, elapsed, env):
    windows = subprocess.run(
        ["xwininfo", "-display", env["DISPLAY"], "-root", "-tree"],
        env=env, capture_output=True, text=True, check=True).stdout
    match = re.search(r'^\s+(0x[0-9a-fA-F]+) "RoboMaster Aim Lab', windows, re.MULTILINE)
    if not match:
        return False
    capture = subprocess.run(
        ["xwd", "-id", match.group(1)], env=env, capture_output=True, check=True).stdout
    if len(capture) < 100:
        return False
    header = struct.unpack(">25I", capture[:100])
    header_size, width, height, bytes_per_line, colors = header[0], header[4], header[5], header[12], header[19]
    offset = header_size + colors * 12
    pixels = np.frombuffer(capture, dtype=np.uint8, count=height * bytes_per_line, offset=offset)
    pixels = pixels.reshape(height, bytes_per_line)[:, :width * 4].reshape(height, width, 4)
    image = Image.fromarray(pixels[:, :, [2, 1, 0]])
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    image.save(frame_dir / f"{label}_{elapsed:05.1f}s.png")
    return True


def run_one(scene, mode, repeat, frames, timeout):
    label = f"{scene}_{mode}_{repeat}"
    state_path = Path(f"/dev/shm/rm_aim_sim_{CHANNEL}/state.json")
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    env.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
    auth = sorted(Path(env["XDG_RUNTIME_DIR"]).glob(".mutter-Xwaylandauth.*"))
    if auth:
        env["XAUTHORITY"] = str(auth[0])
    scene_path = OUTPUT / f"{scene}.cfg"
    vision_path = OUTPUT / f"{mode}.yaml"
    sim = vis = None
    try:
        with (OUTPUT / f"{label}.godot.log").open("w") as sim_log:
            sim = subprocess.Popen(
                [str(SIM / ".tools" / "godot"), "--path", str(SIM),
                 "--display-driver", "x11", "--rendering-method", "gl_compatibility",
                 "--max-fps", "60", "--", "--remote", "--config=" + str(scene_path)],
                env=env, stdout=sim_log, stderr=subprocess.STDOUT)
            for _ in range(300):
                if sim.poll() is not None:
                    raise RuntimeError("Simulator exited: " +
                                       (OUTPUT / f"{label}.godot.log").read_text()[-2000:])
                try:
                    state = json.loads(state_path.read_text())
                    if str(state.get("publisher_id", "")).startswith(f"{sim.pid}-"):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("No fresh simulator state")
            if state["health"]["maximum"] != 1000 or state["health"]["current"] != 1000:
                raise RuntimeError("Target did not start at 1000/1000 HP")
            if abs(state["weapon"].get("drag_k", -1) - DRAG_K) > 1e-9:
                raise RuntimeError("Simulator drag coefficient differs from test setting")
            start_sim_time = state["sim_time"]
            trace_path = OUTPUT / f"{label}.trace.jsonl"
            with (OUTPUT / f"{label}.vision.log").open("w") as vis_log:
                vis = subprocess.Popen(
                    [str(BINARY), str(vision_path), f"--channel={CHANNEL}",
                     "--display=false", "--stop-on-kill=true", f"--max-frames={frames}",
                     "--expected-health=1000", "--diagnostics=true",
                     f"--trace={trace_path}"],
                    cwd=AIM, env=env, stdout=vis_log, stderr=subprocess.STDOUT)
                print("RUNNING", label, "sim_pid", sim.pid, "vision_pid", vis.pid, flush=True)
                run_started = time.monotonic()
                next_capture = 0.75
                frames_captured = 0
                while vis.poll() is None:
                    elapsed = time.monotonic() - run_started
                    if elapsed >= next_capture and frames_captured < 4:
                        if capture_simulator_frame(OUTPUT, label, elapsed, env):
                            frames_captured += 1
                        next_capture += 2.0
                    if elapsed > timeout:
                        raise subprocess.TimeoutExpired(vis.args, timeout)
                    time.sleep(0.1)
                exit_code = vis.returncode
            if exit_code:
                raise RuntimeError("Vision exited: " +
                                   (OUTPUT / f"{label}.vision.log").read_text()[-3000:])
            for _ in range(70):
                state = json.loads(state_path.read_text())
                if (state["stats"]["fired"] == state["stats"]["resolved"]
                        and state["stats"]["pending"] == 0
                        and state["stats"]["in_flight"] == 0):
                    break
                time.sleep(0.1)
            stats = state["stats"]
            vision_log = (OUTPUT / f"{label}.vision.log").read_text()
            result_line = next((line for line in vision_log.splitlines()
                                if "SIM_RESULT " in line), "")
            summary_line = next((line for line in vision_log.splitlines()
                                 if "SIM_SUMMARY " in line), "")
            ttk_match = re.search(r"time_to_kill=([\d.-]+)s", result_line)
            frame_match = re.search(r"frames=(\d+)", summary_line)
            record = {
                "scene": scene, "mode": mode, "repeat": repeat, "frames_max": frames,
                "frames_processed": int(frame_match.group(1)) if frame_match else None,
                "start_sim_time": start_sim_time,
                "max_health": state["health"]["maximum"],
                "final_health": state["health"]["current"],
                "killed": not state["health"]["alive"],
                "death_time": state["health"].get("death_time", -1),
                "time_to_kill": float(ttk_match.group(1)) if ttk_match else None,
                "first_hit_time": stats.get("first_hit_time", -1),
                "stats": stats, "sim_time": state["sim_time"],
                "session_id": state["session_id"],
                "sim_drag_k": state["weapon"]["drag_k"],
                "vision_drag_k": 0.0 if mode == "baseline" else DRAG_K,
                "binary_sha256": digest(BINARY),
                "scene_sha256": digest(scene_path),
                "vision_sha256": digest(vision_path),
                "sim_model_sha256": digest(SIM / "scripts" / "simulation.gd"),
            }
            (OUTPUT / f"{label}.json").write_text(json.dumps(record, indent=2))
            print("RESULT", label, "HP", record["final_health"],
                  "hit", stats["hit"], "resolved", stats["resolved"],
                  "fired", stats["fired"], "killed", record["killed"], flush=True)
            return record
    finally:
        stop(vis)
        stop(sim)


def main():
    global CHANNEL, OUTPUT, SIM
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-run wall-clock timeout in seconds")
    parser.add_argument("--scene", choices=("static", "spin", "move", "move_spin"))
    parser.add_argument("--mode", choices=("baseline", "drag"))
    parser.add_argument("--channel", default=CHANNEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true",
                        help="keep completed records and skip completed trials")
    args = parser.parse_args()
    CHANNEL = args.channel
    OUTPUT = args.output.expanduser().resolve()
    SIM = OUTPUT / "simulator"
    prepare()
    if not BINARY.is_file():
        raise RuntimeError(f"Build auto_aim_sim first: {BINARY}")
    results_path = OUTPUT / "results.json"
    results = json.loads(results_path.read_text()) if args.resume and results_path.exists() else []
    completed = {(r["scene"], r["mode"], r["repeat"]) for r in results}
    scenes = (args.scene,) if args.scene else ("static", "spin", "move", "move_spin")
    modes = (args.mode,) if args.mode else ("baseline", "drag")
    for scene in scenes:
        for repeat in range(1, args.repeats + 1):
            ordered_modes = modes if repeat % 2 else tuple(reversed(modes))
            for mode in ordered_modes:
                if (scene, mode, repeat) in completed:
                    print("SKIP", scene, mode, repeat, "already recorded", flush=True)
                    continue
                results.append(run_one(scene, mode, repeat, args.frames, args.timeout))
                results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
