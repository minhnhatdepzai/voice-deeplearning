# Architecture and change guide

## Runtime

`src/main.jsx` owns navigation, the voice library and training state polling.
`src/SpeechToSpeech.jsx` records a full browser utterance, submits it after
**Dừng & nghe**, polls a job and plays the finished WAV. The browser still
holds the recording until submission; the API caps uploads at 500 MB.

`server/app.py` contains FastAPI routes, input validation, GPU serialization
and media responses. `server/speech_jobs.py` persists conversion job state in
`data/speech_jobs/`; one in-process executor handles conversion. Jobs with
stored inputs are requeued after API restart. Keep Uvicorn at **one worker**
until the queue is moved to an external service.

The conversion path is MediaRecorder Opus → FFmpeg WAV → faster-whisper text →
VieNeu TTS chunks → FFmpeg concat/postprocess. It preserves words through ASR
and synthesis, but timing and emotion from the input speaker are not carried
through as direct voice conversion would be.

## Training

`training/upload_train.py` filters the selected speaker, calls
`training/uploaded_audio.py` to segment and transcribe, pauses for human review,
then runs preliminary disjoint evaluation and full-data LoRA training.
`training/dataset_review.py` stores corrections separately from the original
candidate list; approval rewrites the accepted train manifest and records a
hash so preliminary evaluation cannot be reused for changed labels.
`src/TrainingReview.jsx` is the clip review UI.

The desktop launcher uses `systemd-run --user` for long training workers.
Docker sets `VOICE_STUDIO_WORKER_MODE=subprocess`; worker logs are in
`data/worker_logs/`. A container restart marks interrupted uploaded training
as failed, so the library can retry it. Conversion jobs resume automatically.

`training/setup.sh` pins the VieNeu upstream Git revision and builds its
separate environment. `training/setup_voice_filter.sh` pins ClearVoice and
PyTorch CUDA 12.8 in another environment. Change upstream pins only with a
real synthesis, separation and reload test. App dependencies are specified
in `requirements.txt` and frozen in `requirements.lock`.

## Where to change behavior

| Goal | Files to start with | Check |
| --- | --- | --- |
| Change recording or job UX | `src/SpeechToSpeech.jsx`, `server/app.py`, `server/speech_jobs.py` | `tests/test_speech_jobs.py`, real browser recording |
| Tune ASR/TTS profiles | `server/speech_to_speech.py`, `server/realtime_voice.py` | held-out speech benchmark plus listening comparison |
| Improve speaker filtering | `server/target_voice.py`, `server/dominant_speaker.py` | labeled overlapping-speaker clips and residual audio |
| Change training labels or splits | `training/uploaded_audio.py`, `training/dataset_review.py`, `training/upload_train.py` | manifest counts, disjoint split, adapter reload |
| Upgrade container/model environment | `Dockerfile`, `compose.yaml`, `training/setup*.sh` | `scripts/smoke_container.sh`, then GPU inference on a toolkit-enabled host |

The MIT license applies to repository code. External checkpoints, datasets,
voice recordings and generated user media must be handled under their own
licenses and consent terms. Benchmarks in `TRAINING_REPORT.md` are historical
measurements, not guarantees for a newly uploaded voice.
