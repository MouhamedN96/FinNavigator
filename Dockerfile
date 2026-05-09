# FinNavigator API server — FastAPI + LangChain agents
# Used by Fly.io / Railway / HuggingFace Spaces. Cloudflare Pages serves the UI separately.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Native deps for vision (pillow), git-lfs for HF model fetch, ffmpeg for media
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement, generally good hygiene)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl --fail http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT}"]
