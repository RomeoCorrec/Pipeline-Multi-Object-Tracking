from __future__ import annotations
import argparse
from .pipeline import MOTPipeline, load_config
from .counter import VirtualLineCounter


def main() -> None:
    parser = argparse.ArgumentParser(description="MOT Pipeline")
    parser.add_argument("--source", default="0",
                        help="Video source: int for webcam, path for video file")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--line", default=None,
                        help="Path to virtual_line.json")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_display:
        config.setdefault("pipeline", {})["display"] = False

    counter = VirtualLineCounter(args.line) if args.line else None
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    pipeline = MOTPipeline(config=config, counter=counter)
    pipeline.run(source)


if __name__ == "__main__":
    main()
