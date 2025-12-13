"""RBM experiment configuration."""

config = {
    "device": "auto",  # "auto" | "cpu" | "cuda:0" | "mps"

    "data": {
        "csv_path": "data.csv",  # Path to training data CSV
        "drop_cols": ['x'],
    },

    "model": {
        "bm_type": "rbm",   # Currently only RBM is supported
        "visible_blocks": {"v1": 4, "v2": 4},
        "hidden_blocks": {"h1": 5, "h":10, "h2": 5},
        "cross_block_restrictions": [("v1", "h2"), ("v2", "h1")],
        "initialization": "random",
    },

    "preprocess": {
        "q_low": 0.001,
        "q_high": 0.999,
        "add_missing_bit": True,
        "max_categories": 200,
        "min_category_freq": 2,
    },

    "dataloader": {
        "batch_size": 256,
        "split": [0.8, 0.1, 0.1],
        "seed": 42,
        "shuffle_train": True,
        "num_workers": 0,
        "drop_last_train": True,
        "pin_memory": "auto",  # "auto" | True | False
    },

    "train": {
        "epochs": 100,
        "lr": 1e-1,
        "k": 10,
        "kind": "mean-field",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "clip_value": 0.05,
        "clip_norm": 5.0,
        "lr_schedule": {"mode": "cosine", "min_lr": 1e-4},
        "sparse_hidden": True,
        "rho": 0.1,
        "lambda_sparse": 0.01,
        "early_stopping": True,
        "es_patience": 8,
    },

    "eval": {
        "recon_k": 1,
    },

    "conditional": {
        "clamp_idx": [0, 1, 2, 3],
        "target_idx": [4, 5, 6, 7],
        "n_samples": 100,
        "burn_in": 500,
        "thin": 10,
    },
}
