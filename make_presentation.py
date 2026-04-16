"""
Génère la vidéo de présentation du projet MOT Pipeline.
Entrée : pedestrian.mp4   Sortie : presentation_MOT.mp4
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── Constantes de style ──────────────────────────────────────────────────────
PALETTE: list[tuple[int, int, int]] = [
    tuple(int(c) for c in cv2.cvtColor(
        np.array([[[int(h * 180 / 20), 220, 240]]], dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )[0][0])
    for h in range(20)
]

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
SOURCE = "pedestrian.mp4"
OUTPUT = "presentation_MOT.mp4"
LINE_CFG = "config/virtual_line.json"


# ── Overlay utilitaires ──────────────────────────────────────────────────────

def put_text_shadow(img, text, pos, font, scale, color, thickness=1):
    """Texte avec ombre portée pour meilleure lisibilité."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    """Rectangle semi-transparent."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud(frame, tracks, fps, counter, frame_idx, total_frames):
    """HUD complet : compteurs, barre de progression, titre."""
    h, w = frame.shape[:2]

    # ── Fond HUD haut-gauche ───────────────────────────────────────────────
    draw_rounded_rect(frame, 8, 8, 240, 80, (20, 20, 20))
    put_text_shadow(frame, f"FPS : {fps:5.1f}", (18, 35), FONT_SMALL, 0.75, (0, 230, 0), 2)
    put_text_shadow(frame, f"Pistes actives : {len(tracks)}", (18, 65), FONT_SMALL, 0.65, (255, 255, 255), 1)

    # ── Compteur ligne virtuelle ───────────────────────────────────────────
    if counter is not None:
        for cls_name, cnt in counter.counts.items():
            draw_rounded_rect(frame, 8, 88, 280, 125, (20, 20, 80))
            put_text_shadow(
                frame,
                f"IN : {cnt.in_count}   OUT : {cnt.out_count}",
                (18, 115), FONT_SMALL, 0.70, (80, 180, 255), 2,
            )

    # ── Titre projet (haut-droite) ─────────────────────────────────────────
    title = "MOT Pipeline — ByteTrack + YOLOv8"
    (tw, th), _ = cv2.getTextSize(title, FONT_SMALL, 0.60, 1)
    tx = w - tw - 14
    draw_rounded_rect(frame, tx - 6, 8, w - 8, 38, (20, 20, 20))
    put_text_shadow(frame, title, (tx, 30), FONT_SMALL, 0.60, (200, 200, 200), 1)

    # ── Barre de progression (bas) ─────────────────────────────────────────
    bar_h, bar_y = 10, h - 18
    draw_rounded_rect(frame, 0, bar_y - 2, w, h, (10, 10, 10), alpha=0.70)
    progress = int(w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(frame, (0, bar_y), (progress, bar_y + bar_h), (0, 200, 100), -1)
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (80, 80, 80), 1)


def draw_tracks(frame, tracks, trails):
    """Boîtes, labels et traînées de mouvement."""
    live_ids = {t.track_id for t in tracks}
    for tid in list(trails):
        if tid not in live_ids:
            del trails[tid]

    for t in tracks:
        color = PALETTE[t.track_id % len(PALETTE)]
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Traînée
        trails.setdefault(t.track_id, deque(maxlen=40)).append((cx, cy))
        pts = list(trails[t.track_id])
        for i in range(1, len(pts)):
            alpha_i = i / len(pts)
            thickness = max(1, int(3 * alpha_i))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Boîte
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label avec fond
        label = f"#{t.track_id}"
        (lw, lh), _ = cv2.getTextSize(label, FONT_SMALL, 0.50, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    FONT_SMALL, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

        # Point central
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)


def draw_virtual_line(frame, counter):
    """Ligne virtuelle avec dégradé de couleur et label."""
    if counter is None:
        return
    p1 = tuple(map(int, counter.p1))
    p2 = tuple(map(int, counter.p2))
    cv2.line(frame, p1, p2, (0, 60, 255), 2, cv2.LINE_AA)
    # Tirets lumineux pour effet
    step = 40
    for x in range(p1[0], p2[0], step * 2):
        x2_ = min(x + step, p2[0])
        cv2.line(frame, (x, p1[1]), (x2_, p1[1]), (0, 140, 255), 1, cv2.LINE_AA)
    put_text_shadow(frame, "Ligne de comptage", (p1[0] + 8, p1[1] - 8),
                    FONT_SMALL, 0.55, (0, 180, 255), 1)


# ── Pipeline principal ───────────────────────────────────────────────────────

def run():
    from mot_pipeline.pipeline import load_config
    from mot_pipeline.tracker.bytetrack import ByteTrack

    # Imports lazy (évite le téléchargement YOLO au démarrage si déjà en cache)
    from mot_pipeline.detector.yolo_detector import YOLOv8Detector
    from mot_pipeline.reid.embedder import MobileNetV2Embedder

    config = load_config("config/default.yaml")
    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    print(f"[INFO] Device : {device}")

    detector = YOLOv8Detector({**config["detector"], "device": device})
    tracker = ByteTrack(config["tracker"])
    embedder = MobileNetV2Embedder({**config["embedder"], "device": device})
    counter = None  # ligne virtuelle désactivée pour la présentation

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {SOURCE!r}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS_SRC = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT, fourcc, FPS_SRC, (W, H))

    trails: dict[int, deque] = {}
    fps_buf: deque[float] = deque(maxlen=30)
    frame_idx = 0

    print(f"[INFO] Traitement de {TOTAL} frames ({W}×{H} @ {FPS_SRC:.0f} FPS)…")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # Inférence
        detections = detector.detect(frame)

        # Embeddings Re-ID
        high = [d for d in detections if d.conf >= tracker.conf_high]
        if high:
            crops = []
            valid = []
            for d in high:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                fh, fw = frame.shape[:2]
                crop = frame[max(0, y1):min(y2, fh), max(0, x1):min(x2, fw)]
                if crop.size > 0:
                    crops.append(crop)
                    valid.append(d)
            if crops:
                embs = embedder.embed(crops)
                for d, e in zip(valid, embs):
                    d.embedding = e

        tracks = tracker.update(detections)
        # Mesure FPS réel
        dt = time.perf_counter() - t0
        fps_buf.append(1.0 / dt if dt > 0 else 0.0)
        fps = sum(fps_buf) / len(fps_buf)

        # Dessin
        draw_virtual_line(frame, counter)
        draw_tracks(frame, tracks, trails)
        draw_hud(frame, tracks, fps, counter, frame_idx, TOTAL)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == TOTAL:
            pct = 100 * frame_idx / TOTAL
            print(f"  [{frame_idx:4d}/{TOTAL}] {pct:5.1f}%  |  FPS traitement : {fps:.1f}")

    cap.release()
    writer.release()
    print(f"\n[OK] Vidéo sauvegardée : {OUTPUT}")


if __name__ == "__main__":
    run()
