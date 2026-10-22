# Contributing to Vocalab

Vocalab is a local React/Vite frontend with a FastAPI backend. Python modules
in `server/` serve media and inference; `training/` prepares datasets, runs
fine-tuning and evaluates checkpoints. `src/` contains the UI. Runtime data,
recordings, model caches and generated outputs live in ignored `data/`.

## Local setup

Use Python 3.12, Node 22, `ffmpeg`, and optionally an NVIDIA GPU. Run
`npm ci`, create `.venv`, install `requirements-dev.txt`, then run
`npm run build` and `.venv/bin/python -m pytest -q`. `./run.sh` installs the
application dependencies and starts the local API. `bash training/setup.sh`
installs the pinned VieNeu upstream training environment. Use
`bash training/setup_voice_filter.sh` for speaker separation.
The ClearVoice environment is pinned to the same CUDA 12.8 PyTorch family as
VieNeu in `training/clearvoice-constraints.txt`.

Docker instructions and data persistence are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
When changing a Python dependency range, regenerate the lock with
`uv pip compile requirements.txt -o requirements.lock --python-version 3.12`.

## Making a change

1. Open an issue describing the behavior and a sample that you have rights to use.
2. Keep voice recordings, datasets, checkpoints, tokens and `.env` out of commits.
3. Add a focused regression test for backend behavior that can break silently.
4. Run Python tests, frontend build and `docker compose config -q`.
5. In the pull request, describe changed routes, migration or data effects,
   test commands and their results. Include measured quality data when claiming
   improvements to naturalness, pronunciation, latency or speaker identity.

The repository MIT license covers code only. User recordings and external
datasets/checkpoints retain their own rights and licenses. Do not submit
someone else's voice or model weights without authorization.

## Data and migration notes

`data/voices.json` indexes local voices. Uploaded training state is written to
`data/upload_training/<voice-id>/status.json`; prepared clips and review edits
are in `data/prepared/upload_<voice-id>_*/`. Speech jobs are in
`data/speech_jobs/<job-id>/` and can resume on API restart when the input file
still exists. Back up the whole `data/` directory or Docker volume before
changing dataset formats. Run a single Uvicorn worker for the local job queue.
