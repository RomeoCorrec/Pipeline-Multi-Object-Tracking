from __future__ import annotations
import csv
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

_MOT17_URL = "https://motchallenge.net/data/MOT17.zip"


class MOT17Evaluator:
    """
    Downloads MOT17 (once), runs the pipeline on every *-DPM sequence,
    computes MOTA/IDF1/ID-Switches/FPS via trackeval, prints a rich table,
    and saves a CSV.
    """

    def __init__(self, config: dict) -> None:
        self.mot17_dir = Path(config.get("benchmark", {}).get("mot17_dir", "data/MOT17"))
        self.output_dir = Path(config.get("benchmark", {}).get("output_dir", "results"))
        self.config = config

    def run(self) -> None:
        self._ensure_dataset()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sequences = sorted(self.mot17_dir.glob("MOT17-*-DPM"))
        if not sequences:
            raise FileNotFoundError(
                f"No MOT17-*-DPM sequences found in {self.mot17_dir}. "
                "Check mot17_dir in config."
            )

        results = []
        for seq_path in sequences:
            print(f"\n>>> {seq_path.name}")
            results.append(self._run_sequence(seq_path))

        self._display(results)
        self._save_csv(results)

    def _ensure_dataset(self) -> None:
        if self.mot17_dir.exists() and any(self.mot17_dir.iterdir()):
            return
        self.mot17_dir.parent.mkdir(parents=True, exist_ok=True)
        zip_path = self.mot17_dir.parent / "MOT17.zip"

        print(f"Downloading MOT17 from {_MOT17_URL} ...")
        with tqdm(unit="B", unit_scale=True, desc="MOT17.zip") as pbar:
            def _hook(b: int, bsize: int, tsize: int | None) -> None:
                if tsize:
                    pbar.total = tsize
                pbar.update(b * bsize - pbar.n)
            urllib.request.urlretrieve(_MOT17_URL, zip_path, reporthook=_hook)

        print("Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.mot17_dir.parent)
        zip_path.unlink()
        print(f"Dataset ready at {self.mot17_dir}")

    def _run_sequence(self, seq_path: Path) -> dict:
        from .pipeline import MOTPipeline

        frames = sorted((seq_path / "img1").glob("*.jpg"))
        pipeline = MOTPipeline(config=self.config)
        pred_lines: list[str] = []
        elapsed = 0.0

        for frame_idx, img_path in enumerate(frames, 1):
            frame = cv2.imread(str(img_path))
            t0 = time.perf_counter()

            dets = pipeline.detector.detect(frame)
            pipeline._attach_embeddings(frame, dets)
            tracks = pipeline.tracker.update(dets)

            elapsed += time.perf_counter() - t0
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                w, h = x2 - x1, y2 - y1
                pred_lines.append(
                    f"{frame_idx},{t.track_id},"
                    f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                )

        tracker_dir = self.output_dir / "bytetrack"
        tracker_dir.mkdir(parents=True, exist_ok=True)
        pred_path = tracker_dir / f"{seq_path.name}.txt"
        pred_path.write_text("".join(pred_lines))
        fps = len(frames) / elapsed if elapsed > 0 else 0.0

        metrics = self._eval(seq_path, pred_path, seq_path.name)
        metrics["sequence"] = seq_path.name
        metrics["fps"] = round(fps, 1)
        return metrics

    def _eval(self, seq_path: Path, pred_path: Path, seq_name: str) -> dict:
        try:
            # trackeval uses np.float which was removed in NumPy 1.24
            if not hasattr(np, "float"):
                np.float = float  # type: ignore[attr-defined]

            import trackeval  # type: ignore

            eval_cfg = trackeval.Evaluator.get_default_eval_config()
            eval_cfg.update({"PRINT_RESULTS": False, "OUTPUT_SUMMARY": False,
                              "OUTPUT_EMPTY_CLASSES": False})

            data_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            data_cfg.update({
                "GT_FOLDER": str(seq_path.parent),
                "TRACKERS_FOLDER": str(self.output_dir),
                "TRACKERS_TO_EVAL": ["bytetrack"],
                "SEQMAP_FILE": None,
                "SEQ_INFO": {seq_name: None},
                "OUTPUT_FOLDER": None,
                "TRACKER_SUB_FOLDER": "",
                "SKIP_SPLIT_FOL": True,
            })

            evaluator = trackeval.Evaluator(eval_cfg)
            dataset = [trackeval.datasets.MotChallenge2DBox(data_cfg)]
            metrics = [trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
            raw, _ = evaluator.evaluate(dataset, metrics)

            seq_res = raw["MotChallenge2DBox"]["bytetrack"][seq_name]["pedestrian"]
            clear = seq_res["CLEAR"]
            ident = seq_res["Identity"]
            return {
                "mota": round(float(clear.get("MOTA", 0)) * 100, 1),
                "idf1": round(float(ident.get("IDF1", 0)) * 100, 1),
                "id_switches": int(clear.get("IDSW", 0)),
            }
        except Exception as exc:
            print(f"  [trackeval error] {exc}")
            return {"mota": -1.0, "idf1": -1.0, "id_switches": -1}

    def _display(self, results: list[dict]) -> None:
        console = Console()
        table = Table(title="MOT17 Benchmark Results", show_lines=True)
        for col in ("Sequence", "MOTA ↑", "IDF1 ↑", "ID Sw. ↓", "FPS"):
            table.add_column(col, justify="right")
        for r in results:
            table.add_row(
                r["sequence"], str(r["mota"]), str(r["idf1"]),
                str(r["id_switches"]), str(r["fps"]),
            )
        console.print(table)

    def _save_csv(self, results: list[dict]) -> None:
        path = self.output_dir / f"benchmark_{date.today()}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["sequence", "mota", "idf1", "id_switches", "fps"]
            )
            w.writeheader()
            w.writerows(results)
        print(f"\nResults saved to {path}")
