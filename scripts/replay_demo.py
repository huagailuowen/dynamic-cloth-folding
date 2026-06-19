#!/usr/bin/env python3
"""Replay the bundled dynamic cloth folding demonstration and save frames."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import cloth_env
from utils.cloth_variants import CLOTH_VARIANTS, apply_cloth_variant, variant_names
from utils import general_utils

logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/dynamic_cloth_folding_demo")
    parser.add_argument("--title", default="demo-replay-cpu")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--cloth-variant", choices=variant_names(), default="normal")
    return parser.parse_args()


def build_variant(title, run, max_steps, cloth_variant):
    old_argv = sys.argv[:]
    sys.argv = [
        "replay_demo",
        "--title",
        title,
        "--run",
        str(run),
        "--materials-randomization",
        "0",
        "--lights-randomization",
        "0",
        "--camera-position-randomization",
        "0",
        "--lookat-position-randomization",
        "0",
        "--albumentations-randomization",
        "0",
        "--dynamics-randomization",
        "0",
        "--max-path-length",
        str(max_steps),
    ]
    try:
        args = general_utils.argsparser()
        variant = general_utils.get_variant(args)
        return apply_cloth_variant(variant, cloth_variant)
    finally:
        sys.argv = old_argv


def run_demo(output_dir, title, run, fps, max_steps, cloth_variant):
    os.chdir(REPO_ROOT)
    os.makedirs(output_dir, exist_ok=True)

    variant = build_variant(title, run, max_steps, cloth_variant)
    general_utils.setup_save_folder(variant)

    env = cloth_env.ClothEnv(
        **variant["env_kwargs"],
        randomization_kwargs=variant["randomization_kwargs"],
    )
    env.reset()

    actions = np.genfromtxt(REPO_ROOT / "data" / "demos.csv", delimiter=",")[:, :3]
    frames = []
    metrics = []

    def capture(label):
        _, eval_image, _, _, _ = env.capture_images(aux_output=None, mask_type="corners")
        image = np.clip(eval_image, 0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, f"{label}.png"), image)
        frames.append(image)

    capture("frame_000_start")
    for step, delta in enumerate(actions[:max_steps]):
        action = np.clip(delta / env.output_max, -1, 1).astype(np.float32)
        _, reward, done, info = env.step(action.copy())
        metrics.append(
            [
                step,
                float(reward),
                bool(done),
                float(info.get("corner_sum_error", np.nan)),
                float(info.get("ctrl_error", np.nan)),
            ]
        )
        capture(f"frame_{step + 1:03d}")
        print(
            f"step {step:02d} reward={reward:.3f} done={done} "
            f"corner_sum={info.get('corner_sum_error', np.nan):.4f} "
            f"ctrl_error={info.get('ctrl_error', np.nan):.4f}",
            flush=True,
        )
        if done:
            break

    if frames:
        frames.extend([frames[-1]] * fps)
    imageio.mimsave(os.path.join(output_dir, "demo_replay.gif"), frames, fps=fps)
    np.savetxt(
        os.path.join(output_dir, "metrics.csv"),
        np.array(metrics, dtype=object),
        fmt="%s",
        delimiter=",",
        header="step,reward,done,corner_sum_error,ctrl_error",
        comments="",
    )
    with open(os.path.join(output_dir, "variant_config.json"), "w") as outfile:
        json.dump(variant["cloth_variant"], outfile, indent=2)
    print("cloth_variant", cloth_variant)
    print("variant_description", CLOTH_VARIANTS[cloth_variant]["description"])
    print("output_dir", os.path.realpath(output_dir))
    print("train_dir", variant["save_folder"])
    del env
    return {
        "cloth_variant": cloth_variant,
        "description": CLOTH_VARIANTS[cloth_variant]["description"],
        "output_dir": os.path.realpath(output_dir),
        "train_dir": variant["save_folder"],
        "frames": frames,
        "metrics": metrics,
    }


def main():
    args = parse_args()
    run_demo(
        output_dir=args.output_dir,
        title=args.title,
        run=args.run,
        fps=args.fps,
        max_steps=args.max_steps,
        cloth_variant=args.cloth_variant,
    )


if __name__ == "__main__":
    main()
