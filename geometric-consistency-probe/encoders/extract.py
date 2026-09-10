"""Task 4: extract representations and save them to disk.

    video -> preprocessing -> V-JEPA 2.1 -> representation/tokens -> saved tensor

"Save representations to disk so repeated experiments do not require
repeated inference" -- concretely, for one video keyed `video_id`, this
writes:

    out_dir/{video_id}/representation.npy   -- see
        encoders/REPRESENTATION_FORMAT.md for the exact shape/dtype
    out_dir/{video_id}/metadata.json        -- checkpoint id, pretrained
        flag, preprocessing config, input video shape, output
        representation shape/dtype -- everything needed to know exactly
        what produced this array without re-running inference

`extract_and_save`/`load_representation` are the two functions this
module exists to provide; `main()` is a thin CLI batch-processing
wrapper around them for a directory of Task 2/3-style scenes (each
`{scene_dir}/rgb.npy`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from encoders.base import VideoEncoder


def extract_and_save(encoder: VideoEncoder, video: np.ndarray, video_id: str, out_dir: str | Path) -> Path:
    """Run `encoder` on `video` and save the result + its full provenance.

    Returns the directory written to (`out_dir/{video_id}/`).
    """
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video must be (T, H, W, 3), got shape {video.shape}")

    representation = encoder.encode(video)

    video_dir = Path(out_dir) / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    np.save(video_dir / "representation.npy", representation)

    metadata = {
        "video_id": video_id,
        "encoder": type(encoder).__name__,
        "checkpoint": encoder.checkpoint,
        "pretrained": encoder.pretrained,
        "preprocessing_config": encoder.preprocessing_config.to_dict() if hasattr(encoder, "preprocessing_config") else None,
        "video_shape": list(video.shape),
        "num_frames": int(video.shape[0]),
        "frame_height": int(video.shape[1]),
        "frame_width": int(video.shape[2]),
        "representation_shape": list(representation.shape),
        "representation_dtype": str(representation.dtype),
        "device": getattr(encoder, "device", None),
    }
    (video_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return video_dir


def load_representation(out_dir: str | Path, video_id: str) -> tuple[np.ndarray, dict]:
    """Inverse of extract_and_save: returns (representation, metadata). No
    inference is run -- this only reads what was already saved.
    """
    video_dir = Path(out_dir) / video_id
    representation = np.load(video_dir / "representation.npy")
    metadata = json.loads((video_dir / "metadata.json").read_text())
    return representation, metadata


def extract_dataset(encoder: VideoEncoder, videos: dict[str, np.ndarray], out_dir: str | Path) -> list[str]:
    """extract_and_save for every (video_id, video) pair in `videos`. Returns the video_ids processed."""
    video_ids = []
    for video_id, video in videos.items():
        extract_and_save(encoder, video, video_id, out_dir)
        video_ids.append(video_id)
        print(f"extracted {video_id}: representation shape {encoder.output_dim if hasattr(encoder, 'output_dim') else '?'}")
    return video_ids


def _load_scene_videos(input_dir: Path) -> dict[str, np.ndarray]:
    """Load every `{input_dir}/*/rgb.npy` (Task 2/3's on-disk convention)."""
    videos = {}
    for rgb_path in sorted(input_dir.glob("*/rgb.npy")):
        video_id = rgb_path.parent.name
        videos[video_id] = np.load(rgb_path)
    if not videos:
        raise FileNotFoundError(f"No */rgb.npy files found under {input_dir}")
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Extract frozen V-JEPA 2.1 representations for every {input_dir}/*/rgb.npy video."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of {video_id}/rgb.npy videos (Task 2/3 layout)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Override the default V-JEPA 2.1 checkpoint")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from encoders.vjepa import DEFAULT_CHECKPOINT, DEFAULT_CROP_SIZE, VJEPAEncoder

    encoder = VJEPAEncoder(
        checkpoint=args.checkpoint or DEFAULT_CHECKPOINT,
        pretrained=args.pretrained,
        crop_size=args.crop_size or DEFAULT_CROP_SIZE,
        device=args.device,
    )

    videos = _load_scene_videos(Path(args.input_dir))
    extract_dataset(encoder, videos, args.output_dir)
    print(f"Done: {len(videos)} representations written to {args.output_dir}")


if __name__ == "__main__":
    main()
