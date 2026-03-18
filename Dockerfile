FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# OS依存ライブラリ
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU版を先にインストール（extra-index-urlが必要なため分離）
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# その他の依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CLIPモデルをビルド時にダウンロード（初回リクエストのタイムアウト防止）
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

# アプリケーションコードのコピー
COPY . .

CMD ["uvicorn", "insight:app", "--host", "0.0.0.0", "--port", "8080"]
