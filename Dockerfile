FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    git wget curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch + CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# mmlab ecosystem
RUN pip install --no-cache-dir openmim \
    && mim install mmengine \
    && mim install mmcv==2.0.1 \
    && mim install mmdet==3.1.0 \
    && mim install mmpose==1.1.0

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.0" \
    python-multipart==0.0.9

# Copy project source
COPY musetalk/ ./musetalk/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY app.py api.py entrypoint.sh download_weights.sh ./

RUN mkdir -p /app/models /app/results /app/temp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
