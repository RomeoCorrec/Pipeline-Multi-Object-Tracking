FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mot_pipeline/ ./mot_pipeline/
COPY config/ ./config/

# Pre-download YOLOv8n weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

ENTRYPOINT ["python", "-m", "mot_pipeline"]
CMD ["--source", "0", "--no-display"]
