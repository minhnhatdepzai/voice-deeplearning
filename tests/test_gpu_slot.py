import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from server import app


class GpuSlotTest(unittest.TestCase):
    def test_cuda_oom_has_actionable_api_error(self):
        failure = subprocess.CalledProcessError(1, ["model"], stderr=b"RuntimeError: CUDA out of memory. Tried to allocate 32 MiB")
        with patch.object(app.subprocess, "run", side_effect=failure):
            with self.assertRaises(HTTPException) as raised:
                app._run("model")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("GPU", raised.exception.detail)
        self.assertNotIn("RuntimeError", raised.exception.detail)

    def test_training_lock_blocks_inference_then_releases(self):
        with TemporaryDirectory() as directory, patch.object(app, "DATA", Path(directory)):
            lock_file = Path(directory) / "auto_train/worker.lock"
            lock_file.parent.mkdir()
            holder = subprocess.Popen([
                sys.executable, "-c",
                "import fcntl,sys,time; f=open(sys.argv[1],'a+'); "
                "fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(1)",
                str(lock_file),
            ], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "locked")
                with self.assertRaises(HTTPException) as raised:
                    with app._gpu_slot(wait_seconds=.1):
                        pass
                self.assertEqual(raised.exception.status_code, 503)
            finally:
                holder.wait(timeout=5)
                holder.stdout.close()
            with app._gpu_slot(wait_seconds=.1):
                pass


if __name__ == "__main__":
    unittest.main()
