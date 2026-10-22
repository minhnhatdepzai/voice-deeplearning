# Docker and CI/CD

## Requirements

Install Docker Engine, Docker Compose v2 and the NVIDIA Container Toolkit with
GPU access enabled. The app image uses Python 3.12; the model environments
install PyTorch CUDA 12.8 wheels on first start. CUDA driver,
VRAM and checkpoint requirements still depend on the selected models.
On Ubuntu/Debian, follow the [official NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
to add its repository, install `nvidia-container-toolkit`, run
`sudo nvidia-ctk runtime configure --runtime=docker`, and restart Docker.
Check `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
before starting Vocalab. Restarting Docker can briefly interrupt other containers.

## Start and update

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f vocalab
```

Open `http://127.0.0.1:8000`. First start clones a pinned VieNeu revision,
installs its fine-tuning environment and ClearVoice with PyTorch 2.8 CUDA 12.8
into the persistent volume;
allow at least 25 GB free for first-time downloads, environments and later
checkpoints. This may take several minutes. Checkpoint downloads happen
on first model use. The health check means the API is reachable; generate a
real sample to verify inference. For an API-only smoke test, set
`VOICE_STUDIO_BOOTSTRAP_MODE=skip` in `.env`; voice inference then needs the
upstream environments installed separately.
`bash scripts/smoke_container.sh` builds an ephemeral API-only container,
checks its web page, health route and Python imports, then removes it.
If another local service already uses port 8000, set `VOCALAB_PORT=8001` in
`.env` and open that port instead.

To update code while keeping recordings and model data:

```bash
git pull --ff-only
docker compose up -d --build
```

`vocalab-data` is a named volume. Back it up before major updates:
It starts empty and is separate from the checkout's `data/`; Docker does not
silently copy private recordings or adapters from the host. Use the public
release adapter installer for released voices, or make a reviewed data backup
and migrate it intentionally. Do not copy a host `.venv` into the volume.

```bash
docker run --rm -v voice-deeplearning_vocalab-data:/data:ro -v "$PWD":/backup ubuntu:24.04 tar -C /data -czf /backup/vocalab-data.tar.gz .
```

Check the actual volume name with `docker volume ls` if the Compose project
name differs. The web port binds to localhost intentionally. To make it
available remotely, put an authenticated HTTPS reverse proxy in front of it;
the API does not yet provide per-user authentication. Do not expose port 8000
directly to the internet.

## CI and releases

Every pull request and push to `main` runs Python tests, a Vite build and a
Compose syntax check. No GPU inference or listening quality benchmark is
claimed by that job. A `v*` tag triggers the separate GHCR image workflow.
Create a version tag only after reviewing release notes and model licensing:

```bash
git tag v1.1.0
git push origin v1.1.0
```

Published images contain code and application dependencies. They exclude
recordings, training data, private adapters and downloaded model checkpoints.
For a deployment using GHCR, change `image:` in Compose to the tagged image,
remove `build:`, and run `docker compose pull && docker compose up -d`.

## Worker behavior

The container runs one Uvicorn process. Speech conversion jobs persist the
uploaded input and progress in the volume and resume after an API restart.
Training uses a separate subprocess and logs under `data/worker_logs/`.
Training interrupted by a container restart may require a manual retry from
the library; the current local queue is not a distributed job system.
