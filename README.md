# MOT Pipeline

Real-time Multi-Object Tracking pipeline: YOLOv8 detection · ByteTrack from scratch · MobileNetV2 Re-ID · virtual line counter · MOT17 benchmark.

## Démonstration

<video src="presentation_MOT.mp4" controls loop muted width="100%"></video>

> Détection YOLOv8n · pistage ByteTrack · traînées de trajectoires · Re-ID MobileNetV2

---

## Features

- **ByteTrack from scratch** — Kalman Filter (8D state), Hungarian algorithm, dual-threshold association, IoU + cosine Re-ID cost
- **Swappable interfaces** — `BaseDetector` / `BaseEmbedder` ABCs; swap YOLOv8 or MobileNetV2 in one line
- **Virtual line counter** — JSON-configured line, cross-product direction detection, CSV export
- **MOT17 benchmark** — auto-download, MOTA / IDF1 / ID Switches / FPS via trackeval
- **CPU + GPU** — auto-detects CUDA; runs on any machine

---

## Quick Start

### pip

```bash
pip install -r requirements.txt
# Webcam
python -m mot_pipeline --source 0
# Video file
python -m mot_pipeline --source path/to/video.mp4
# With virtual line counter
python -m mot_pipeline --source path/to/video.mp4 --line config/virtual_line.json
```

### Docker

```bash
docker build -t mot-pipeline .
# Headless on a video file
docker run --rm -v $(pwd)/videos:/videos mot-pipeline --source /videos/test.mp4 --no-display
```

---

## Configuration

Edit `config/default.yaml`:

```yaml
device: auto          # auto | cpu | cuda
detector:
  model_path: yolov8n.pt
  conf_thresh: 0.25
  classes: [0]        # COCO class IDs (0=person)
tracker:
  conf_high: 0.6
  conf_low: 0.1
  max_lost_age: 30
  min_hits: 2
  reid_alpha: 0.8     # weight of IoU vs cosine cost
embedder:
  dim: 512
pipeline:
  output_video: output.mp4
  display: true
```

Virtual line (`config/virtual_line.json`):
```json
{"x1": 0, "y1": 540, "x2": 1920, "y2": 540}
```

---

## Benchmark — MOT17

```bash
python -c "
from mot_pipeline.benchmark import MOT17Evaluator
from mot_pipeline.pipeline import load_config
MOT17Evaluator(load_config()).run()
"
```

MOT17 (~1.9 GB) is downloaded automatically on first run.

### Results (YOLOv8n, MobileNetV2 Re-ID, CPU)

| Sequence | MOTA ↑ | IDF1 ↑ | ID Sw. ↓ | FPS |
|----------|--------|--------|----------|-----|
| MOT17-02-DPM | — | — | — | — |
| MOT17-04-DPM | — | — | — | — |
| MOT17-05-DPM | — | — | — | — |
| *Run benchmark to fill this table* | | | | |

---

## Generate Demo GIF

```bash
ffmpeg -i output.mp4 -vf "fps=15,scale=640:-1:flags=lanczos" \
       -loop 0 docs/demo.gif
```

---

## Architecture

```
frame → YOLOv8Detector ──→ List[Detection]
                        ↓
              MobileNetV2Embedder (high-conf crops)
                        ↓
              ByteTrack.update() ──→ List[Track]
                        ↓
        ┌───────────────┴────────────────┐
  VirtualLineCounter           Visualizer.draw()
        ↓                               ↓
   crossings.csv              annotated frame → VideoWriter
```

---

## Extending

### Swap detector (e.g. RT-DETR)

```python
from mot_pipeline.detector.base import BaseDetector

class RTDETRDetector(BaseDetector):
    def detect(self, frame):
        ...  # return List[Detection]

pipeline = MOTPipeline(config, detector=RTDETRDetector(...))
```

### Swap Re-ID model (e.g. OSNet)

```python
from mot_pipeline.reid.base import BaseEmbedder

class OSNetEmbedder(BaseEmbedder):
    def embed(self, crops):
        ...  # return np.ndarray [N, D], L2-normalized

pipeline = MOTPipeline(config, embedder=OSNetEmbedder(...))
```

---

## Tests

```bash
pytest tests/ -v
```
