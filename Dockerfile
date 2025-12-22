FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# OS依存ライブラリのインストール（最新のパッケージ名に対応）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# ディレクトリ作成
RUN mkdir -p public/uploads public/compares output
CMD ["uvicorn", "insight:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "debug"]
