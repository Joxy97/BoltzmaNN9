#!/usr/bin/env python
"""BoltzmaNN9 command-line interface."""

import argparse
import sys
from pathlib import Path


def cmd_new_project(args):
    """Create a new project."""
    from boltzmann.project import create_project

    try:
        create_project(args.project_name)
    except FileExistsError as e:
        print(f"Error: {e}")
        return 1
    return 0


def cmd_train(args):
    """Train a model."""
    from boltzmann.config import load_config
    from boltzmann.pipeline import train_rbm, save_model
    from boltzmann.project import create_run, save_run_config, get_run_paths
    from boltzmann.run_utils import RunLogger, save_history, save_plots

    project_path = Path(args.project)

    if not project_path.exists():
        print(f"Error: Project not found: {project_path}")
        return 1

    config_path = project_path / "config.py"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    # Create run directory
    run_dir = create_run(project_path, args.name)
    paths = get_run_paths(run_dir)

    # Copy config to run
    save_run_config(run_dir, config_path)

    print(f"Starting training run: {run_dir.name}")
    print("=" * 60)

    # Train with logging
    with RunLogger(paths["log"]):
        # Load config and adjust data path to be relative to project
        cfg = load_config(config_path)

        # Make data path relative to project
        data_path = cfg.get("data", {}).get("csv_path", "")
        if data_path and not Path(data_path).is_absolute():
            cfg["data"]["csv_path"] = str(project_path / data_path)

        # Import here to avoid circular imports
        from boltzmann.data import BMDataset, split_rbm_loaders
        from boltzmann.model import RBM
        from boltzmann.utils import resolve_device, resolve_pin_memory

        # Setup
        device = resolve_device(cfg.get("device", "auto"))
        print(f"Using device: {device}")

        data_cfg = cfg.get("data", {})
        drop_cols = data_cfg.get("drop_cols", [])
        dataset = BMDataset(cfg["data"]["csv_path"], drop_cols=drop_cols)

        print(f"Loaded dataset: {len(dataset)} samples")
        print(f"  Columns: {dataset.columns}")

        dl = cfg.get("dataloader", {})
        pin_memory = resolve_pin_memory(dl.get("pin_memory", "auto"), device)

        loaders = split_rbm_loaders(
            dataset,
            batch_size=dl.get("batch_size", 256),
            split=tuple(dl.get("split", (0.8, 0.1, 0.1))),
            seed=dl.get("seed", 42),
            shuffle_train=dl.get("shuffle_train", True),
            num_workers=dl.get("num_workers", 0),
            pin_memory=pin_memory,
            drop_last_train=dl.get("drop_last_train", True),
        )

        # Model
        model_cfg = dict(cfg.get("model", {}))
        model = RBM(model_cfg).to(device)

        # Train
        train_cfg = dict(cfg.get("train", {}))
        history = model.fit(
            loaders["train"],
            val_loader=loaders["val"],
            **train_cfg,
        )

    # Save outputs
    import torch
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "nv": model.nv,
        "nh": model.nh,
        "config": cfg,
    }
    torch.save(checkpoint, paths["model"])
    print(f"Model saved to: {paths['model']}")

    save_history(history, paths["history"])
    save_plots(history, paths["plots"], model=model)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"  - model.pt")
    print(f"  - config.py")
    print(f"  - training.log")
    print(f"  - history.json")
    print(f"  - plots/")

    return 0


def cmd_evaluate(args):
    """Evaluate a trained model."""
    from boltzmann.config import load_config
    from boltzmann.pipeline import load_model
    from boltzmann.project import get_run_paths, find_latest_run
    from boltzmann.run_utils import save_metrics

    # Determine run directory
    run_path = Path(args.run)

    if not run_path.exists():
        # Maybe it's a project path - find latest run
        project_path = run_path
        run_path = find_latest_run(project_path)
        if run_path is None:
            print(f"Error: No runs found in project: {project_path}")
            return 1
        print(f"Using latest run: {run_path.name}")

    paths = get_run_paths(run_path)

    # Load model and config from run
    if not paths["model"].exists():
        print(f"Error: Model not found: {paths['model']}")
        return 1

    if not paths["config"].exists():
        print(f"Error: Config not found: {paths['config']}")
        return 1

    model, saved_cfg = load_model(paths["model"])
    cfg = load_config(paths["config"])

    # Find project path from run path (output/run_xxx -> project)
    project_path = run_path.parent.parent

    # Adjust data path
    data_path = cfg.get("data", {}).get("csv_path", "")
    if data_path and not Path(data_path).is_absolute():
        cfg["data"]["csv_path"] = str(project_path / data_path)

    print(f"Evaluating run: {run_path.name}")
    print("=" * 60)

    # Import and setup
    from boltzmann.data import BMDataset, split_rbm_loaders
    from boltzmann.tester import RBMTester
    from boltzmann.utils import resolve_device, resolve_pin_memory

    device = resolve_device(cfg.get("device", "auto"))
    model = model.to(device)
    print(f"Using device: {device}")

    data_cfg = cfg.get("data", {})
    drop_cols = data_cfg.get("drop_cols", [])
    dataset = BMDataset(cfg["data"]["csv_path"], drop_cols=drop_cols)

    dl = cfg.get("dataloader", {})
    pin_memory = resolve_pin_memory(dl.get("pin_memory", "auto"), device)

    loaders = split_rbm_loaders(
        dataset,
        batch_size=dl.get("batch_size", 256),
        split=tuple(dl.get("split", (0.8, 0.1, 0.1))),
        seed=dl.get("seed", 42),
        shuffle_train=False,
        num_workers=dl.get("num_workers", 0),
        pin_memory=pin_memory,
        drop_last_train=False,
    )

    print(f"Evaluating on {len(loaders['test'].dataset)} test samples")

    # Basic metrics
    eval_cfg = cfg.get("eval", {})
    test_metrics = model.evaluate(loaders["test"], recon_k=eval_cfg.get("recon_k", 1))
    print("Test metrics:", test_metrics)

    # Conditional NLL
    cond_cfg = cfg.get("conditional", {})
    tester = RBMTester(
        model=model,
        test_dataloader=loaders["test"],
        clamp_idx=cond_cfg["clamp_idx"],
        target_idx=cond_cfg["target_idx"],
    )
    conditional_results = tester.conditional_nll(
        n_samples=cond_cfg.get("n_samples", 100),
        burn_in=cond_cfg.get("burn_in", 500),
        thin=cond_cfg.get("thin", 10),
    )

    # Save metrics
    all_metrics = {
        "test_metrics": test_metrics,
        "conditional_nll_nats": conditional_results["mean_nll_nats"],
        "conditional_nll_bits": conditional_results["mean_nll_bits"],
        "conditional_nll_per_bit": conditional_results["mean_nll_per_bit"],
        "n_target_bits": conditional_results["n_target_bits"],
    }
    save_metrics(all_metrics, paths["metrics"])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test metrics: {test_metrics}")
    print(f"Conditional NLL (nats): {conditional_results['mean_nll_nats']:.4f}")
    print(f"Conditional NLL (bits): {conditional_results['mean_nll_bits']:.4f}")
    print(f"Conditional NLL per bit: {conditional_results['mean_nll_per_bit']:.4f}")
    print(f"Target bits: {conditional_results['n_target_bits']}")
    print(f"\nMetrics saved to: {paths['metrics']}")

    return 0


def cmd_preprocess_raw(args):
    """Preprocess raw data based on config."""
    from boltzmann.config import load_config
    from boltzmann.preprocessor import DataPreprocessor

    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    cfg = load_config(config_path)

    print(f"Running preprocessor with config: {config_path}")
    print("=" * 60)

    preprocessor = DataPreprocessor(cfg, config_dir=config_path.parent)
    df = preprocessor.fit_transform()

    print(f"\nPreprocessing complete!")
    print(f"  Output CSV: {preprocessor.output_csv_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Visible blocks: {preprocessor.get_visible_blocks_sizes()}")

    return 0


def cmd_list_runs(args):
    """List all runs in a project."""
    from boltzmann.project import list_runs

    project_path = Path(args.project)

    if not project_path.exists():
        print(f"Error: Project not found: {project_path}")
        return 1

    runs = list_runs(project_path)

    if not runs:
        print(f"No runs found in: {project_path}")
        return 0

    print(f"Runs in {project_path}:")
    print("-" * 40)
    for run in runs:
        print(f"  {run.name}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="BoltzmaNN9 - Restricted Boltzmann Machine toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # new_project command
    new_proj_parser = subparsers.add_parser(
        "new_project",
        help="Create a new project",
    )
    new_proj_parser.add_argument(
        "project_name",
        type=str,
        help="Name/path for the new project",
    )

    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
    )
    train_parser.add_argument(
        "--project", "-p",
        type=str,
        required=True,
        help="Path to project directory",
    )
    train_parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Optional name suffix for the run",
    )

    # evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model",
    )
    eval_parser.add_argument(
        "--run", "-r",
        type=str,
        required=True,
        help="Path to run directory or project (uses latest run)",
    )

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List runs in a project",
    )
    list_parser.add_argument(
        "--project", "-p",
        type=str,
        required=True,
        help="Path to project directory",
    )

    # preprocess_raw command
    preprocess_parser = subparsers.add_parser(
        "preprocess_raw",
        help="Preprocess raw data using DataPreprocessor",
    )
    preprocess_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config file",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "new_project":
        return cmd_new_project(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "list":
        return cmd_list_runs(args)
    elif args.command == "preprocess_raw":
        return cmd_preprocess_raw(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
