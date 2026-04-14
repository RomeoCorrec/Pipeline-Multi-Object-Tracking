# MOT Pipeline — Design Spec
**Date:** 2026-04-14  
**Author:** Romeo  
**Status:** Approved

---

## 1. Objectif

Pipeline Python complet de Multi-Object Tracking (MOT) pour portfolio ingénieur Computer Vision. Critère de succès principal : projet **clé en main** — tout fonctionne end-to-end, Docker inclus, README professionnel.

---

## 2. Stack technique

| Composant | Librairie |
|-----------|-----------|
| Détection | ultralytics (YOLOv8) |
| Tracking | ByteTrack from scratch (scipy Hungarian, Kalman from scratch) |
| Re-ID | torchvision MobileNetV2 (interface abstraite swappable) |
| Visualisation | opencv-python |
| Benchmark | trackeval (pip) |
| Device | CPU par défaut, GPU auto-détecté (`torch.cuda.is_available()`) |
| Config | YAML (default.yaml) + JSON (virtual_line.json) |
| Container | Docker |

Python 3.11.

---

## 3. Structure du projet

```
mot-pipeline/
├── mot_pipeline/
│   ├── __init__.py
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseDetector(ABC)
│   │   └── yolo_detector.py     # YOLOv8Detector(BaseDetector)
│   ├── tracker/
│   │   ├── __init__.py
│   │   ├── bytetrack.py         # ByteTrack(BaseTracker)
│   │   ├── kalman.py            # KalmanFilter (from scratch, 8D state)
│   │   └── track.py             # Track dataclass + TrackState enum
│   ├── reid/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseEmbedder(ABC)
│   │   └── embedder.py          # MobileNetV2Embedder(BaseEmbedder)
│   ├── pipeline.py              # MOTPipeline
│   ├── visualizer.py            # draw_tracks, draw_trails, fps overlay
│   ├── counter.py               # VirtualLineCounter
│   └── benchmark.py             # MOT17Evaluator
├── config/
│   ├── default.yaml
│   └── virtual_line.json
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_reid.py
│   └── test_counter.py
├── results/                     # benchmark CSV outputs (gitignored)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 4. Dataclasses partagées

### `Detection`
```python
@dataclass
class Detection:
    bbox: list[float]              # [x1, y1, x2, y2]
    conf: float
    class_id: int
    embedding: np.ndarray | None = None
```

### `Track`
```python
class TrackState(Enum):
    New = 1
    Active = 2
    Lost = 3
    Removed = 4

@dataclass
class Track:
    track_id: int
    state: TrackState
    bbox: list[float]              # [x1, y1, x2, y2]
    age: int                       # frames depuis création
    hits: int                      # associations réussies
    time_since_update: int         # frames sans association
    embedding: np.ndarray | None = None
```

---

## 5. Interfaces abstraites

### `BaseDetector`
```python
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...
```

### `BaseEmbedder`
```python
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        # retourne array [N, D] normalisé L2
        ...
```

---

## 6. Detector : YOLOv8Detector

- Wrapping `ultralytics.YOLO`
- Paramètres config : `model_path` (défaut `yolov8n.pt`), `conf_thresh`, `iou_thresh`, `device`
- `detect(frame)` → filtre les classes cibles (configurable, défaut : `person`) → retourne `List[Detection]`
- Device passé à YOLO via `model.to(device)`

---

## 7. Tracker : ByteTrack

### Paramètres (default.yaml)
| Paramètre | Valeur défaut | Description |
|-----------|--------------|-------------|
| `conf_high` | 0.6 | Seuil haute confiance |
| `conf_low` | 0.1 | Seuil basse confiance |
| `iou_thresh_high` | 0.3 | IoU min association primaire |
| `iou_thresh_low` | 0.5 | IoU min association secondaire |
| `max_lost_age` | 30 | Frames avant Removed |
| `min_hits` | 2 | Hits min avant Active |
| `reid_alpha` | 0.8 | Poids IoU dans coût hybride |

### Flow `ByteTrack.update(detections: List[Detection]) -> List[Track]`
1. Split : `high_dets` (conf ≥ conf_high), `low_dets` (conf_low ≤ conf < conf_high)
2. **Association primaire** : coût hybride `α·IoU_cost + (1-α)·cosine_dist` entre tracks *Active+Lost* et `high_dets`. Hungarian (scipy). Seuil `iou_thresh_high`.
3. **Association secondaire** : tracks non-matchés vs `low_dets`. Coût = IoU pur. Seuil `iou_thresh_low`.
4. **Nouvelles tracks** : `high_dets` non-matchées → état *New*. Passage *Active* après `min_hits` hits consécutifs.
5. **Lost/Removed** : tracks non-matchées → `time_since_update++`. Si > `max_lost_age` → *Removed*.
6. Retourne tracks avec état *Active* (et *New* si hits ≥ min_hits).

### Kalman Filter (`kalman.py`)
- **État** 8D : `[cx, cy, ar, h, vx, vy, var, vh]`
- **Mesure** 4D : `[cx, cy, ar, h]`
- Matrices F, H, Q, R définies analytiquement (convention ByteTrack paper)
- Implémentation from scratch — `filterpy` non requis pour ce module

---

## 8. Re-ID : MobileNetV2Embedder

- `torchvision.models.mobilenet_v2(pretrained=True)`, features avant classifier
- Sortie adaptateur linéaire → 512D, normalisé L2
- `embed(crops)` : redimension 224×224, normalisation ImageNet, inférence batch
- Device transmis depuis config
- Interface `BaseEmbedder` permet le swap vers OSNet/ResNet50 (documenté README)

---

## 9. Pipeline : MOTPipeline

```python
class MOTPipeline:
    def __init__(self, config: dict,
                 detector: BaseDetector,
                 tracker: ByteTrack,
                 embedder: BaseEmbedder,
                 counter: VirtualLineCounter | None):
        ...

    def run(self, source: int | str) -> None:
        # source : int=webcam, str=video file, str=MOT17 frames dir
        ...
```

**Boucle frame :**
1. `cap.read()` → frame
2. `detector.detect(frame)` → detections
3. Crop ROIs des détections haute confiance → `embedder.embed(crops)` → attach embeddings
4. `tracker.update(detections)` → tracks
5. `counter.update(tracks)` si counter actif
6. `visualizer.draw(frame, tracks, counter)` → frame annoté
7. `writer.write(frame)` et/ou `cv2.imshow`

**Sortie** : fichier vidéo annoté (configurable) + stats terminales.

---

## 10. Visualizer

- Palette HSV fixe 20 couleurs, index = `track_id % 20`
- Trails : `deque(maxlen=30)` par track_id, `cv2.polylines` avec opacité dégradée
- Overlay : FPS (moyenne glissante 30 frames), nb tracks actifs
- Ligne virtuelle : segment rouge + labels `In: N / Out: M` par classe

---

## 11. VirtualLineCounter

- Config depuis `virtual_line.json` : `{"x1": float, "y1": float, "x2": float, "y2": float}`
- Détection croisement : produit vectoriel `(P2-P1) × (centroïde - P1)` — changement de signe entre frames consécutives
- Direction : côté positif = "in", côté négatif = "out" (configurable)
- Export CSV à l'arrêt du pipeline : colonnes `timestamp, track_id, class_name, direction`

---

## 12. Benchmark : MOT17Evaluator

- Téléchargement auto MOT17 depuis miroir officiel si absent (avec barre de progression `tqdm`)
- Exécution pipeline en mode headless sur chaque séquence MOT17
- Métriques via `trackeval` : MOTA, IDF1, ID Switches, FPS
- Affichage tableau `rich` dans le terminal
- Export `results/benchmark_YYYY-MM-DD.csv`

---

## 13. Configuration (default.yaml)

```yaml
device: auto          # auto | cpu | cuda
detector:
  model_path: yolov8n.pt
  conf_thresh: 0.25
  iou_thresh: 0.45
  classes: [0]        # 0 = person
tracker:
  conf_high: 0.6
  conf_low: 0.1
  iou_thresh_high: 0.3
  iou_thresh_low: 0.5
  max_lost_age: 30
  min_hits: 2
  reid_alpha: 0.8
embedder:
  type: mobilenetv2   # swappable
  dim: 512
pipeline:
  output_video: output.mp4
  display: true
benchmark:
  mot17_dir: data/MOT17
  output_dir: results/
```

---

## 14. Dockerfile

- Base : `python:3.11-slim`
- Installe dépendances système OpenCV (`libgl1`, `libglib2.0-0`)
- `COPY requirements.txt` + `pip install`
- `COPY mot_pipeline/ config/`
- `CMD ["python", "-m", "mot_pipeline.pipeline", "--source", "0"]`
- Image CPU-only (GPU via `nvidia/cuda` base documenté dans README)

---

## 15. README

Sections :
1. Demo GIF (instructions ffmpeg pour générer)
2. Features
3. Installation (pip + Docker)
4. Usage (webcam, fichier vidéo, benchmark)
5. Configuration (YAML + virtual_line.json)
6. **Tableau métriques** : MOTA, IDF1, ID Switches, FPS sur MOT17-Det
7. Architecture (schéma ASCII des composants)
8. Extending (comment swapper YOLOv8 ou l'embedder)

---

## 16. Tests

| Fichier | Ce qui est testé |
|---------|-----------------|
| `test_detector.py` | `YOLOv8Detector.detect()` sur frame synthétique |
| `test_tracker.py` | ByteTrack : association, états New/Active/Lost/Removed, Kalman predict/update |
| `test_reid.py` | `MobileNetV2Embedder.embed()` : shape output, normalisation L2 |
| `test_counter.py` | `VirtualLineCounter` : croisement détecté, direction, export CSV |

---

## 17. Dépendances (requirements.txt)

```
ultralytics>=8.0
opencv-python>=4.8
torch>=2.0
torchvision>=0.15
scipy>=1.11
numpy>=1.24
pyyaml>=6.0
tqdm>=4.65
rich>=13.0
trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git
```
