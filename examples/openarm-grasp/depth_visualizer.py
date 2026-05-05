"""Colorize mono16 depth frames for quick Dora/MuJoCo debugging."""

import os
from pathlib import Path
import struct
import time
import zlib

from dora import Node
import numpy as np
import pyarrow as pa


DEFAULT_WIDTH = int(os.getenv("IMAGE_WIDTH", "1280"))
DEFAULT_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "720"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "out/depth"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))
MIN_DEPTH_MM = float(os.getenv("MIN_DEPTH_MM", "0"))
MAX_DEPTH_MM = float(os.getenv("MAX_DEPTH_MM", "2000"))


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image as PNG using only the standard library."""
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"expected RGB image, got shape {image.shape}")

    rows = [b"\x00" + image[y].tobytes() for y in range(height)]
    raw = b"".join(rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=4))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Map near depth to warm colors and far depth to cool colors."""
    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)

    min_depth = MIN_DEPTH_MM
    max_depth = MAX_DEPTH_MM
    if min_depth <= 0:
        min_depth = float(np.percentile(depth_mm[valid], 2))
    if max_depth <= min_depth:
        max_depth = float(np.percentile(depth_mm[valid], 98))

    normalized = np.zeros(depth_mm.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (depth_mm[valid].astype(np.float32) - min_depth) / (max_depth - min_depth),
        0.0,
        1.0,
    )

    near = 1.0 - normalized
    rgb = np.stack(
        [
            np.clip(255.0 * near, 0, 255),
            np.clip(255.0 * (1.0 - np.abs(normalized - 0.5) * 2.0), 0, 255),
            np.clip(255.0 * normalized, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def main() -> None:
    node = Node()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_count = 0

    print(f"[depth-visualizer] saving every {SAVE_EVERY} frame(s) to {OUTPUT_DIR}")

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "depth":
            metadata = event["metadata"]
            width = int(metadata.get("width", DEFAULT_WIDTH))
            height = int(metadata.get("height", DEFAULT_HEIGHT))

            depth = event["value"].to_numpy().astype(np.uint16).reshape((height, width))
            rgb = colorize_depth(depth)

            output_metadata = metadata.copy()
            output_metadata["encoding"] = "rgb8"
            output_metadata["width"] = width
            output_metadata["height"] = height
            output_metadata["timestamp"] = time.time_ns()
            node.send_output("depth_image", pa.array(rgb.ravel()), output_metadata)

            if SAVE_EVERY > 0 and frame_count % SAVE_EVERY == 0:
                path = OUTPUT_DIR / f"depth_{frame_count:06d}.png"
                write_rgb_png(path, rgb)
                valid = depth[depth > 0]
                if valid.size:
                    print(
                        "[depth-visualizer] "
                        f"{path} range={int(valid.min())}-{int(valid.max())}mm"
                    )
                else:
                    print(f"[depth-visualizer] {path} no valid depth")

            frame_count += 1

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
