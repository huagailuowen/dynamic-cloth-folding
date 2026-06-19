#!/usr/bin/env python3
"""Run normal/light/stiff cloth replays and export a side-by-side video."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_demo import run_demo
from utils.cloth_variants import CLOTH_VARIANTS


COMPARISON_VARIANTS = ("normal", "light", "stiff")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/cloth_variant_comparison")
    parser.add_argument("--title", default="cloth-variant-comparison")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=24)
    return parser.parse_args()


def label_frame(frame, label):
    image = Image.fromarray(frame)
    band_height = 36
    canvas = Image.new("RGB", (image.width, image.height + band_height), "white")
    canvas.paste(image, (0, band_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), label, fill=(0, 0, 0))
    return np.array(canvas)


def write_mp4_from_frames(frames_dir, output_path, fps):
    frame_pattern = str(frames_dir / "comparison_%03d.png")
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for variant_name in COMPARISON_VARIANTS:
        result = run_demo(
            output_dir=str(output_dir / variant_name),
            title=f"{args.title}-{variant_name}",
            run=args.run,
            fps=args.fps,
            max_steps=args.max_steps,
            cloth_variant=variant_name,
        )
        results.append(result)

    max_len = max(len(result["frames"]) for result in results)
    frames_dir = output_dir / "comparison_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    comparison_frames = []
    labels = {
        "normal": "normal: 9x9 grid",
        "light": "light: 9x9 grid, 1/10 density",
        "stiff": "stiff: coarse 5x5 grid",
    }

    for frame_idx in range(max_len):
        labeled_frames = []
        for result in results:
            frames = result["frames"]
            frame = frames[min(frame_idx, len(frames) - 1)]
            labeled_frames.append(label_frame(frame, labels[result["cloth_variant"]]))
        canvas = np.concatenate(labeled_frames, axis=1)
        frame_path = frames_dir / f"comparison_{frame_idx:03d}.png"
        imageio.imwrite(frame_path, canvas)
        comparison_frames.append(canvas)

    imageio.mimsave(output_dir / "cloth_variant_comparison.gif", comparison_frames, fps=args.fps)
    try:
        write_mp4_from_frames(frames_dir, output_dir / "cloth_variant_comparison.mp4", args.fps)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"mp4 export failed: {exc}")

    with open(output_dir / "summary.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "variant",
            "description",
            "steps",
            "final_corner_sum_error",
            "final_ctrl_error",
            "output_dir",
        ])
        for result in results:
            metrics = result["metrics"]
            if metrics:
                final_corner_sum = metrics[-1][3]
                final_ctrl_error = metrics[-1][4]
            else:
                final_corner_sum = ""
                final_ctrl_error = ""
            writer.writerow([
                result["cloth_variant"],
                CLOTH_VARIANTS[result["cloth_variant"]]["description"],
                len(metrics),
                final_corner_sum,
                final_ctrl_error,
                result["output_dir"],
            ])

    print("comparison_dir", output_dir.resolve())
    print("comparison_gif", (output_dir / "cloth_variant_comparison.gif").resolve())
    print("comparison_mp4", (output_dir / "cloth_variant_comparison.mp4").resolve())


if __name__ == "__main__":
    main()
