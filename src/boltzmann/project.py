"""Project and run management utilities."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _get_templates_dir() -> Path:
    """Get the path to the templates directory."""
    return Path(__file__).parent.parent / "templates"


def _load_config_template(data_path: str) -> str:
    """Load config.py template and substitute the data path.

    Args:
        data_path: Path to set for csv_path in the config.

    Returns:
        Config file content with substituted data path.
    """
    template_path = _get_templates_dir() / "config.py"
    content = template_path.read_text()

    # Replace the csv_path value (handles various quoting styles)
    content = re.sub(
        r'("csv_path":\s*)"[^"]*"',
        f'\\1"{data_path}"',
        content
    )

    return content


def _load_generator_template() -> str:
    """Load synthetic_generator.py template from templates directory.

    Returns:
        Generator script content.
    """
    template_path = _get_templates_dir() / "synthetic_generator.py"
    return template_path.read_text()


def create_project(project_path: str | Path) -> Path:
    """Create a new BoltzmaNN9 project.

    Args:
        project_path: Path where the project will be created.

    Returns:
        Path to the created project.

    Raises:
        FileExistsError: If project directory already exists.
    """
    project_path = Path(project_path)

    if project_path.exists():
        raise FileExistsError(f"Project directory already exists: {project_path}")

    # Create project structure
    project_path.mkdir(parents=True)
    data_dir = project_path / "data"
    data_dir.mkdir()
    output_dir = project_path / "output"
    output_dir.mkdir()

    # Create config.py from template with correct data path
    data_csv_path = "data/data.csv"
    config_content = _load_config_template(data_csv_path)
    config_file = project_path / "config.py"
    config_file.write_text(config_content)

    # Create synthetic_generator.py in data folder from template
    generator_content = _load_generator_template()
    generator_file = data_dir / "synthetic_generator.py"
    generator_file.write_text(generator_content)

    print(f"Created project: {project_path}")
    print(f"  - config.py")
    print(f"  - data/")
    print(f"    - synthetic_generator.py")
    print(f"  - output/")
    print(f"\nNext steps:")
    print(f"  1. cd {project_path}/data && python synthetic_generator.py")
    print(f"  2. Edit {project_path}/config.py as needed")
    print(f"  3. python boltzmann.py train --project {project_path}")

    return project_path


def create_run(project_path: str | Path, run_name: str | None = None) -> Path:
    """Create a new run directory within a project.

    Args:
        project_path: Path to the project.
        run_name: Optional custom run name. If None, uses timestamp.

    Returns:
        Path to the created run directory.
    """
    project_path = Path(project_path)
    output_dir = project_path / "output"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Generate run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        run_dir_name = f"run_{timestamp}_{run_name}"
    else:
        run_dir_name = f"run_{timestamp}"

    run_dir = output_dir / run_dir_name
    run_dir.mkdir()

    # Create subdirectories
    (run_dir / "plots").mkdir()
    (run_dir / "checkpoints").mkdir()

    return run_dir


def save_run_config(run_dir: Path, config_path: Path) -> None:
    """Copy config file to run directory.

    Args:
        run_dir: Path to the run directory.
        config_path: Path to the original config file.
    """
    dest = run_dir / "config.py"
    shutil.copy2(config_path, dest)


def get_run_paths(run_dir: Path) -> Dict[str, Path]:
    """Get standard paths within a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dictionary with paths for model, log, plots, checkpoints, config.
    """
    return {
        "model": run_dir / "model.pt",
        "log": run_dir / "training.log",
        "plots": run_dir / "plots",
        "checkpoints": run_dir / "checkpoints",
        "config": run_dir / "config.py",
        "history": run_dir / "history.json",
        "metrics": run_dir / "metrics.json",
    }


def find_latest_run(project_path: str | Path) -> Path | None:
    """Find the most recent run in a project.

    Args:
        project_path: Path to the project.

    Returns:
        Path to the latest run directory, or None if no runs exist.
    """
    project_path = Path(project_path)
    output_dir = project_path / "output"

    if not output_dir.exists():
        return None

    runs = sorted(output_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None


def list_runs(project_path: str | Path) -> list[Path]:
    """List all runs in a project.

    Args:
        project_path: Path to the project.

    Returns:
        List of run directory paths, sorted by date (newest first).
    """
    project_path = Path(project_path)
    output_dir = project_path / "output"

    if not output_dir.exists():
        return []

    return sorted(output_dir.glob("run_*"), reverse=True)
