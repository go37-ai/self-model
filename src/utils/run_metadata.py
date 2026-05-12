"""Utilities for organizing cloud run results with datetime-prefixed S3 paths.

Every cloud run should:
1. Call get_run_prefix() at startup to get a unique YYYY-MM-DD_HHMM prefix
2. Call tag_run() to create a git tag linking this code to the results
3. Use that prefix for all S3 uploads
4. Call generate_readme() before uploading to create a README.md documenting the run
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def get_run_prefix():
    """Generate a datetime prefix for this run: YYYY-MM-DD_HHMM."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")


def get_git_commit():
    """Get the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_readme(output_dir, script_name, args_dict, model_name, description,
                    file_descriptions=None):
    """Generate a README.md documenting a cloud run.

    Args:
        output_dir: Path to write README.md
        script_name: Name of the script (e.g., "run_capping_v3.py")
        args_dict: Dict of command-line arguments
        model_name: Model identifier
        description: One-line description of what this run does
        file_descriptions: Optional dict mapping filename -> description
    """
    commit = get_git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Run: {script_name}",
        f"",
        f"**Description:** {description}",
        f"**Timestamp:** {timestamp}",
        f"**Git commit:** {commit}",
        f"**Model:** {model_name}",
        f"",
        f"## Parameters",
        f"",
    ]

    for key, value in sorted(args_dict.items()):
        lines.append(f"- **{key}:** {value}")

    if file_descriptions:
        lines.append("")
        lines.append("## Output Files")
        lines.append("")
        for filename, desc in sorted(file_descriptions.items()):
            lines.append(f"- **{filename}:** {desc}")

    readme_path = Path(output_dir) / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines) + "\n")
    return readme_path


def tag_run(run_prefix, script_name, args_dict):
    """Create an annotated git tag linking this commit to the run's results.

    Tag name: run/{run_prefix}
    Tag message includes script name and all arguments for reproducibility.
    Silently skips if not in a git repo or if tagging fails.
    """
    tag_name = f"run/{run_prefix}"
    message = f"Script: {script_name}\nArgs: {json.dumps(args_dict, default=str, indent=2)}"
    try:
        result = subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", message],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("Created git tag: %s", tag_name)
        else:
            logger.warning("Git tag failed: %s", result.stderr.strip())
    except Exception as e:
        logger.warning("Git tagging skipped: %s", e)


def get_s3_base(model_name, run_prefix):
    """Get the S3 base path for a run.

    Returns: s3://go37-ai/self-model-results/{model_name}/{run_prefix}/
    """
    return f"s3://go37-ai/self-model-results/{model_name}/{run_prefix}"


def s3_upload(local_path, s3_path, recursive=False):
    """Upload to S3 and return True on success, False on failure.

    Wraps subprocess.run with returncode check so callers can branch on the
    result (e.g., skip pod shutdown if any upload failed).
    """
    cmd = ["aws", "s3", "cp"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([str(local_path), str(s3_path)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("S3 upload failed [%s -> %s]: %s",
                     local_path, s3_path, r.stderr.strip())
        return False
    return True


def conditional_shutdown(upload_success: bool, keep_alive: bool = False) -> None:
    """Stop the RunPod instance only if all uploads succeeded and keep_alive=False.

    On failure or when keep_alive is True, log instructions for manual recovery
    and leave the pod running. The pod's local /workspace/run.log survives so
    you can ssh in, scp the log, and inspect intermediate state before manual
    `runpodctl stop pod $RUNPOD_POD_ID`.

    Args:
        upload_success: True iff every required S3 upload succeeded.
        keep_alive: Force-skip shutdown regardless of upload outcome (e.g.
                    --no-shutdown debug flag).
    """
    if keep_alive:
        logger.warning("keep_alive=True — skipping pod shutdown. "
                       "Manual stop: runpodctl stop pod $RUNPOD_POD_ID")
        return
    if not upload_success:
        logger.error("Upload failed — keeping pod alive so you can ssh in and recover. "
                     "Pod: scp /workspace/run.log <local>; then "
                     "runpodctl stop pod $RUNPOD_POD_ID when done.")
        return
    import os
    try:
        logger.info("All uploads OK — shutting down pod ...")
        # IMPORTANT: do NOT use `runpodctl config --apiKey ...` here. That command
        # tries to sync SSH keys to the cloud, which 401s on keys without
        # SSH-management scope, and the failure aborts the whole config command
        # BEFORE the API key is persisted to ~/.runpod/config.toml. Subsequent
        # `runpodctl stop pod` then uses the stale/broken cached key and 401s too.
        # Writing config.toml directly skips the broken SSH-sync step entirely.
        # See reference_pod_setup.md for the full diagnosis (2026-05-12).
        os.system(
            ". /etc/rp_environment 2>/dev/null; "
            "mkdir -p /root/.runpod && "
            "printf 'apikey = \"%s\"\\napiurl = \"https://api.runpod.io/graphql\"\\n' "
            "\"$RUNPOD_API_KEY\" > /root/.runpod/config.toml && "
            "runpodctl stop pod \"$RUNPOD_POD_ID\" 2>&1"
        )
    except Exception as e:
        logger.warning("Auto-shutdown failed: %s — pod left running", e)
