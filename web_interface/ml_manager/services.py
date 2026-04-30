import sys
import os
import json
import tempfile
import subprocess


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ROOT = os.path.dirname(BASE_DIR)
MAIN_SCRIPT = os.path.join(ML_ROOT, "main.py")


def execute_ml_workflow(job_instance):
    print(f'---start Job for {job_instance.user} ...', flush=True)

    job_instance.status = 'running'
    job_instance.save()

    params = job_instance.parameters or {}
    if not isinstance(params, dict):
        raise ValueError("TrainingJob.parameters must be a JSON object (dict).")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"training_job_{job_instance.pk}_",
            dir=ML_ROOT,
            delete=False,
        ) as temp_json:
            json.dump(params, temp_json)
            temp_json.flush()
            temp_path = temp_json.name

        cmd = [sys.executable, MAIN_SCRIPT, "--run_config", temp_path]
        result = subprocess.run(cmd, cwd=ML_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ML workflow failed (code {result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        job_instance.status = "done"
        job_instance.save()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
