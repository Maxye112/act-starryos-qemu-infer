from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def scalar_list(value, expected: int, name: str, row: int) -> list[float]:
    values = list(value)
    if len(values) < expected:
        raise ValueError(f"row {row}: {name} has {len(values)} values, expected {expected}")
    return [float(v) for v in values[:expected]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a lightweight CSV manifest for C++ ACT dataset evaluation.")
    parser.add_argument("--data-dir", type=Path, default=Path("output/dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deploy/cpp_onnxruntime/data/eval_manifest.csv"),
    )
    args = parser.parse_args()

    parquet_files = sorted(args.data_dir.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.data_dir}/data")

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    episodes_path = args.data_dir / "meta" / "episodes" / "chunk-000" / "episodes.parquet"
    episode_ranges: list[tuple[int, int, int]] = []
    if episodes_path.exists():
        episodes = pd.read_parquet(episodes_path)
        for _, row in episodes.iterrows():
            start = int(row["start_frame_index"])
            end = start + int(row["num_frames"])
            episode_ranges.append((start, end, int(row["episode_index"])))
    required = ["observation.image", "observation.state", "action"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write("index,episode_index,image_path,state_left,state_right,gt_left_vel,gt_right_vel,gt_gripper_target\n")
        for i, row in df.iterrows():
            state = scalar_list(row["observation.state"], 2, "observation.state", int(i))
            action = scalar_list(row["action"], 3, "action", int(i))
            image_path = str(row["observation.image"])
            episode_index = -1
            for start, end, ep in episode_ranges:
                if start <= int(i) < end:
                    episode_index = ep
                    break
            out.write(
                f"{i},{episode_index},{image_path},"
                f"{state[0]:.9g},{state[1]:.9g},"
                f"{action[0]:.9g},{action[1]:.9g},{action[2]:.9g}\n"
            )

    print(f"wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
