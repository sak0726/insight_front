# 開発環境と一致させるため 3.12 を使用
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
# Cloud Run 等の環境に合わせて 8080 を維持
ENV PORT=8080

WORKDIR /app

# OS依存ライブラリの追加
# libglib2.0-0: OpenCVの動作に必要となることが多いライブラリ
# libgl1-mesa-glx: OpenCV用
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# 実行権限やディレクトリ作成（必要に応じて）
RUN mkdir -p public/uploads public/compares output

# モデルのロード時間を考慮し、タイムアウト設定を調整可能にする
CMD ["uvicorn", "insight:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "60"]