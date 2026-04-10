"""Utilities for organizing cloud run results with datetime-prefixed S3 paths.

Every cloud run should:
1. Call get_run_prefix() at startup to get a unique YYYYMMDDHHMM prefix
2. Use that prefix for all S3 uploads
3. Call generate_readme() before uploading to create a README.md documenting the run
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_run_prefix():
    """Generate a datetime prefix for this run: YYYYMMDDHHMM."""
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")


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


def get_s3_base(model_name, run_prefix):
    """Get the S3 base path for a run.

    Returns: s3://go37-ai/self-model-results/{model_name}/{run_prefix}/
    """
    return f"s3://go37-ai/self-model-results/{model_name}/{run_prefix}"
