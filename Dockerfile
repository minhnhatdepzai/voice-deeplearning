FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html vite.config.js ./
COPY src ./src
COPY public ./public
RUN npm run build

FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    VOICE_STUDIO_WORKER_MODE=subprocess
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libsndfile1 libgomp1 tini ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir uv==0.12.18
WORKDIR /app
COPY requirements.txt requirements.lock ./
RUN uv pip install --no-cache --python /opt/venv/bin/python -r requirements.lock
ENV HOME=/app/data HF_HOME=/app/data/huggingface \
    XDG_CACHE_HOME=/app/data/cache TORCH_HOME=/app/data/torch \
    UV_CACHE_DIR=/app/data/uv-cache
COPY . .
COPY --from=frontend /app/dist ./dist
RUN ln -s /opt/venv /app/.venv \
    && mkdir -p /app/data \
    && chown -R 1000:1000 /app/data
USER 1000:1000
EXPOSE 8000
ENTRYPOINT ["tini", "--", "/app/scripts/docker-entrypoint.sh"]
