from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from project.dataset.density_map import load_or_create_density_map
from project.dataset.loader import image_to_mat_path, list_image_paths, resize_with_max_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and cache adaptive density maps.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--cache-dir", type=Path, default=Path("project/data/cache"))
    parser.add_argument("--max-dim", type=int, default=1536)
    parser.add_argument("--output-stride", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for part in args.parts:
        for split in args.splits:
            image_paths = list_image_paths(args.dataset_root, part, split)
            cache_root = args.cache_dir / f"part_{part}_{split}"

            for image_path in tqdm(image_paths, desc=f"Cache part {part} {split}"):
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = image_rgb.shape[:2]
                image_rgb = resize_with_max_dim(image_rgb, args.max_dim)
                h, w = image_rgb.shape[:2]
                scale_x = float(w) / float(max(orig_w, 1))
                scale_y = float(h) / float(max(orig_h, 1))

                load_or_create_density_map(
                    image_path=image_path,
                    mat_path=image_to_mat_path(image_path),
                    image_shape=(h, w),
                    cache_root=cache_root,
                    output_stride=args.output_stride,
                    force_rebuild=args.force,
                    point_scale=(scale_x, scale_y),
                )

    print("Density cache generation completed.")


if __name__ == "__main__":
    main()
