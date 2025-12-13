from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None


# -----------------------------
# Helpers: device
# -----------------------------
def resolve_device(device_cfg: str) -> str:
    """
    Resolve device string deterministically. Returns a torch-style device string.
    """
    if device_cfg is None:
        return "cpu"
    device_cfg = str(device_cfg).lower().strip()
    if device_cfg in ("cpu", "cuda", "cuda:0", "cuda:1", "mps"):
        return device_cfg
    if device_cfg != "auto":
        # accept arbitrary explicit torch device strings like "cuda:2"
        return str(device_cfg)

    # auto
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    # mps availability check
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------
# Helpers: parsing "path/to.csv/col"
# -----------------------------
def parse_source_key(key: str) -> Tuple[str, str]:
    """
    Parse "path/to/file.csv/column_name" -> (csv_path, column_name).

    Notes:
    - This assumes the column name does not contain "/".
    - Windows paths are supported if they use "/" in config. If you need
      backslashes, use raw strings and still include a final "/" before column.
    """
    key = str(key).strip()
    if "/" not in key:
        raise ValueError(
            f"Invalid visible_blocks key '{key}'. Expected 'path/to.csv/column_name'."
        )
    csv_path, col = key.rsplit("/", 1)
    csv_path = csv_path.strip()
    col = col.strip()
    if not csv_path or not col:
        raise ValueError(
            f"Invalid visible_blocks key '{key}'. Parsed csv_path='{csv_path}', col='{col}'."
        )
    return csv_path, col


def safe_feature_name(source_key: str) -> str:
    """
    Create a stable, filesystem-safe, column-safe feature name from a source key.
    Example: "data/raw_1.csv/price" -> "data__raw_1.csv__price"
    """
    # Keep it readable and deterministic
    return (
        source_key.replace("\\", "/")
        .replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )


# -----------------------------
# Bit packing utilities
# -----------------------------
def int_to_bits_msb_first(x: np.ndarray, nbits: int) -> np.ndarray:
    """
    x: (N,) uint64
    returns: (N, nbits) uint8 bits, MSB first
    """
    x = x.astype(np.uint64, copy=False)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint64)
    return ((x[:, None] >> shifts) & 1).astype(np.uint8)


def bits_to_int_msb_first(bits: np.ndarray) -> np.ndarray:
    """
    bits: (N, nbits) uint8, MSB first
    returns: (N,) uint64
    """
    bits = bits.astype(np.uint64, copy=False)
    nbits = bits.shape[1]
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.uint64)
    return (bits << shifts).sum(axis=1)


def binary_to_gray(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.uint64, copy=False)
    return (q ^ (q >> 1)).astype(np.uint64)


def gray_to_binary(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.uint64, copy=False)
    q = g.copy()
    shift = 1
    while True:
        shifted = q >> shift
        if np.all(shifted == 0):
            break
        q ^= shifted
        shift <<= 1
    return q


# -----------------------------
# Specs / metadata
# -----------------------------
@dataclass
class FloatGraySpec:
    source_key: str
    feature_name: str
    nbits: int
    low: float
    high: float
    q_low: float
    q_high: float
    add_missing_bit: bool = True


@dataclass
class BinarySpec:
    source_key: str
    feature_name: str
    add_missing_bit: bool = True


@dataclass
class CategoricalSpec:
    source_key: str
    feature_name: str
    categories: List[str]
    add_unk: bool = True
    add_missing: bool = True


# -----------------------------
# Main preprocessor
# -----------------------------
class DataPreprocessor:
    """
    Industry-grade RBM preprocessor:
    - Reads multiple raw CSV files based on config["model"]["visible_blocks"]
    - Auto-detects types: float -> Gray K-bit, categorical -> one-hot, binary -> passthrough
    - Adds missing indicator bits (default)
    - Writes processed dataset to config["data"]["csv_path"]
    - Writes metadata JSON next to output CSV
    - Resolves relative paths against the provided config_dir
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        config_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        self.config = config
        self.config_dir = Path(config_dir).resolve() if config_dir is not None else None

        self.device_str: str = resolve_device(config.get("device", "auto"))

        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})

        raw_output_path = data_cfg.get("csv_path") or "data/processed.csv"
        self.output_csv_path = self._resolve_path(raw_output_path)
        self.drop_cols: List[str] = list(data_cfg.get("drop_cols", []))

        self.bm_type = str(model_cfg.get("bm_type", "rbm")).lower().strip()
        if self.bm_type != "rbm":
            raise ValueError(f"Only bm_type='rbm' is supported, got: {self.bm_type}")

        # visible_blocks: { "path/to/raw.csv/col": K_i, ... }
        raw_visible = model_cfg.get("visible_blocks", {})
        if not isinstance(raw_visible, dict) or len(raw_visible) == 0:
            raise ValueError("model.visible_blocks must be a non-empty dict.")

        # Preprocessing params (optional; safe defaults)
        prep_cfg = config.get("preprocess", {})  # optional block (not required)
        self.q_low: float = float(prep_cfg.get("q_low", 0.001))
        self.q_high: float = float(prep_cfg.get("q_high", 0.999))
        self.add_missing_bit: bool = bool(prep_cfg.get("add_missing_bit", True))
        self.max_categories: int = int(prep_cfg.get("max_categories", 200))
        self.min_category_freq: int = int(prep_cfg.get("min_category_freq", 1))
        self.force_float: bool = bool(prep_cfg.get("force_float", False))
        # if force_float=True, numeric columns are always treated as float quantized (unless binary)

        # Store parsed requests
        self.requested: List[Tuple[str, str, int]] = []  # (source_key, col, K)
        for source_key, K in raw_visible.items():
            csv_path, col = parse_source_key(source_key)
            try:
                K_i = int(K)
            except Exception as e:
                raise ValueError(f"K_i for '{source_key}' must be int-like, got {K}") from e
            if K_i <= 0:
                raise ValueError(f"K_i for '{source_key}' must be >=1, got {K_i}")
            self.requested.append((source_key, col, K_i))

        # Will be filled after fit_transform
        self.float_specs: List[FloatGraySpec] = []
        self.bin_specs: List[BinarySpec] = []
        self.cat_specs: List[CategoricalSpec] = []

        self.visible_blocks_out: Dict[str, List[str]] = {}  # feature -> produced bit columns
        self.processed_columns: List[str] = []

    def _resolve_path(self, path_like: Union[str, os.PathLike]) -> Path:
        """
        Resolve a path relative to the config directory if provided.
        Absolute paths pass through unchanged.
        """
        p = Path(path_like)
        if p.is_absolute() or self.config_dir is None:
            return p
        return (self.config_dir / p).resolve()

    # -----------------------------
    # I/O: load raw columns
    # -----------------------------
    def _load_raw_dataframe(self) -> pd.DataFrame:
        """
        Load all requested columns from their CSV files, merge into one dataframe.
        Validates consistent row counts across files.
        """
        # Group by csv path to avoid re-reading files
        by_file: Dict[Path, List[Tuple[str, str, int]]] = {}
        for source_key, col, k in self.requested:
            csv_path, _ = parse_source_key(source_key)
            resolved_csv = self._resolve_path(csv_path)
            by_file.setdefault(resolved_csv, []).append((source_key, col, k))

        merged_parts: List[pd.DataFrame] = []
        expected_len: Optional[int] = None

        for csv_path, items in by_file.items():
            p = csv_path
            if not p.exists():
                raise FileNotFoundError(f"Raw CSV not found: {p}")

            # Read only required columns if possible
            usecols = list({col for (_, col, _) in items})
            df = pd.read_csv(p, usecols=usecols)

            if expected_len is None:
                expected_len = len(df)
            else:
                if len(df) != expected_len:
                    raise ValueError(
                        f"Row count mismatch: file '{p}' has {len(df)} rows, expected {expected_len}. "
                        f"Align datasets before preprocessing (same order / same rows)."
                    )

            # Rename raw columns to stable feature names (based on full source_key)
            renamed = {}
            for source_key, col, _k in items:
                renamed[col] = safe_feature_name(source_key)
            df = df.rename(columns=renamed)

            merged_parts.append(df)

        merged = pd.concat(merged_parts, axis=1)

        # drop requested columns if user asked (in output space)
        for c in self.drop_cols:
            if c in merged.columns:
                merged = merged.drop(columns=[c])

        return merged

    # -----------------------------
    # Type detection
    # -----------------------------
    @staticmethod
    def _is_binary_series(s: pd.Series) -> bool:
        """
        True if (ignoring NaNs) values are subset of {0,1} or {False,True}.
        """
        x = s.dropna()
        if x.empty:
            return False
        # Try numeric interpretation
        if pd.api.types.is_bool_dtype(x):
            return True
        if pd.api.types.is_numeric_dtype(x):
            vals = set(pd.unique(x.astype(float)))
            return vals.issubset({0.0, 1.0})
        return False

    # -----------------------------
    # Float preprocessing: quantize + Gray bits
    # -----------------------------
    def _fit_float_range(self, x: np.ndarray) -> Tuple[float, float]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("Cannot fit float range: all values are NaN/inf.")
        lo = float(np.quantile(x, self.q_low))
        hi = float(np.quantile(x, self.q_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # fallback: min/max if quantiles collapse
            lo = float(np.min(x))
            hi = float(np.max(x))
        if hi <= lo:
            raise ValueError(f"Degenerate float range after fallback: lo={lo}, hi={hi}")
        return lo, hi

    def _encode_float_gray(
        self,
        s: pd.Series,
        nbits: int,
        low: float,
        high: float,
        feature_name: str,
        add_missing_bit: bool,
    ) -> pd.DataFrame:
        x = s.to_numpy(dtype=np.float64, copy=False)

        miss = ~np.isfinite(x)
        x_filled = x.copy()
        # midpoint fill (but missing indicator makes it safe)
        x_filled[miss] = 0.5 * (low + high)

        # clip
        x_filled = np.clip(x_filled, low, high)

        # normalize -> quantize
        denom = (high - low)
        u = (x_filled - low) / denom  # in [0,1]
        qmax = (1 << nbits) - 1
        q = np.rint(u * qmax).astype(np.uint64)
        q = np.clip(q, 0, qmax).astype(np.uint64)

        g = binary_to_gray(q)
        bits = int_to_bits_msb_first(g, nbits)  # (N, nbits)

        col_bits = [f"{feature_name}__g{j:02d}" for j in range(nbits)]
        out = pd.DataFrame(bits, columns=col_bits, index=s.index, dtype=np.uint8)

        if add_missing_bit:
            out[f"{feature_name}__missing"] = miss.astype(np.uint8)

        return out

    # -----------------------------
    # Binary preprocessing
    # -----------------------------
    def _encode_binary(
        self,
        s: pd.Series,
        feature_name: str,
        add_missing_bit: bool,
    ) -> pd.DataFrame:
        miss = s.isna().to_numpy(dtype=bool)

        # strict 0/1
        if pd.api.types.is_bool_dtype(s):
            v = s.fillna(False).astype(bool).to_numpy(dtype=np.uint8)
        else:
            v = s.fillna(0).astype(float).to_numpy()
            v = np.clip(v, 0, 1)
            # values other than 0/1 are suspicious; round and validate
            v_round = np.rint(v).astype(np.uint8)
            # validate (after rounding)
            bad = ~np.isin(v_round, [0, 1])
            if bad.any():
                idx = np.where(bad)[0][:5]
                raise ValueError(
                    f"Binary column '{feature_name}' has non-binary values at rows {idx.tolist()}."
                )
            v = v_round

        out = pd.DataFrame(index=s.index)
        out[f"{feature_name}__bin"] = v.astype(np.uint8)

        if add_missing_bit:
            out[f"{feature_name}__missing"] = miss.astype(np.uint8)

        return out

    # -----------------------------
    # Categorical preprocessing: one-hot (+UNK +MISSING)
    # -----------------------------
    def _fit_categories(self, s: pd.Series) -> List[str]:
        s_obj = s.astype("object")
        miss = s_obj.isna()
        vc = s_obj[~miss].value_counts()

        # filter by frequency
        kept = vc[vc >= self.min_category_freq].index.astype(str).tolist()

        # cap categories
        if len(kept) > self.max_categories:
            kept = kept[: self.max_categories]

        return kept

    def _encode_categorical(
        self,
        s: pd.Series,
        categories: List[str],
        feature_name: str,
        add_unk: bool = True,
        add_missing: bool = True,
    ) -> pd.DataFrame:
        s_obj = s.astype("object")
        miss = s_obj.isna()

        cols = [f"{feature_name}__{c}" for c in categories]
        if add_unk:
            cols.append(f"{feature_name}__UNK")
        if add_missing:
            cols.append(f"{feature_name}__MISSING")

        out = pd.DataFrame(0, index=s.index, columns=cols, dtype=np.uint8)
        cat_set = set(categories)

        # set one-hot
        for idx, val in s_obj[~miss].items():
            v = str(val)
            if v in cat_set:
                out.at[idx, f"{feature_name}__{v}"] = 1
            else:
                if add_unk:
                    out.at[idx, f"{feature_name}__UNK"] = 1

        if add_missing:
            out.loc[miss, f"{feature_name}__MISSING"] = 1

        return out

    # -----------------------------
    # Main pipeline
    # -----------------------------
    def fit_transform(self) -> pd.DataFrame:
        """
        Fits necessary per-feature parameters (ranges/categories) on the loaded raw data,
        then transforms into RBM-ready bits, writes CSV + metadata, and returns the processed df.
        """
        raw = self._load_raw_dataframe()

        self.float_specs.clear()
        self.bin_specs.clear()
        self.cat_specs.clear()
        self.visible_blocks_out.clear()

        blocks: List[pd.DataFrame] = []

        # For each requested feature (in config order), process
        # We reconstruct "source_key -> feature_name" mapping
        source_to_feature: Dict[str, str] = {}
        source_to_bits: Dict[str, int] = {}
        for source_key, _col, k in self.requested:
            fn = safe_feature_name(source_key)
            source_to_feature[source_key] = fn
            source_to_bits[source_key] = int(k)

        # Process in stable order: as provided in visible_blocks
        for source_key, _col, k in self.requested:
            feature_name = source_to_feature[source_key]
            if feature_name not in raw.columns:
                raise KeyError(
                    f"Expected column '{feature_name}' from source '{source_key}' not found in merged dataframe."
                )

            s = raw[feature_name]

            # Decide type
            if self._is_binary_series(s):
                # binary
                encoded = self._encode_binary(s, feature_name, add_missing_bit=self.add_missing_bit)
                self.bin_specs.append(BinarySpec(source_key=source_key, feature_name=feature_name, add_missing_bit=self.add_missing_bit))
                self.visible_blocks_out[feature_name] = list(encoded.columns)
                blocks.append(encoded)
                continue

            if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
                # categorical
                cats = self._fit_categories(s)
                encoded = self._encode_categorical(
                    s,
                    categories=cats,
                    feature_name=feature_name,
                    add_unk=True,
                    add_missing=True,
                )
                self.cat_specs.append(
                    CategoricalSpec(
                        source_key=source_key,
                        feature_name=feature_name,
                        categories=cats,
                        add_unk=True,
                        add_missing=True,
                    )
                )
                self.visible_blocks_out[feature_name] = list(encoded.columns)
                blocks.append(encoded)
                continue

            # numeric non-binary -> float quantize (Gray) using K_i = k
            if not pd.api.types.is_numeric_dtype(s):
                # last resort: treat as categorical
                cats = self._fit_categories(s.astype("object"))
                encoded = self._encode_categorical(
                    s.astype("object"),
                    categories=cats,
                    feature_name=feature_name,
                    add_unk=True,
                    add_missing=True,
                )
                self.cat_specs.append(
                    CategoricalSpec(
                        source_key=source_key,
                        feature_name=feature_name,
                        categories=cats,
                        add_unk=True,
                        add_missing=True,
                    )
                )
                self.visible_blocks_out[feature_name] = list(encoded.columns)
                blocks.append(encoded)
                continue

            # float encoding
            x = s.to_numpy(dtype=np.float64, copy=False)
            low, high = self._fit_float_range(x)

            encoded = self._encode_float_gray(
                s=s,
                nbits=int(k),
                low=low,
                high=high,
                feature_name=feature_name,
                add_missing_bit=self.add_missing_bit,
            )

            self.float_specs.append(
                FloatGraySpec(
                    source_key=source_key,
                    feature_name=feature_name,
                    nbits=int(k),
                    low=float(low),
                    high=float(high),
                    q_low=float(self.q_low),
                    q_high=float(self.q_high),
                    add_missing_bit=self.add_missing_bit,
                )
            )
            self.visible_blocks_out[feature_name] = list(encoded.columns)
            blocks.append(encoded)

        X = pd.concat(blocks, axis=1)
        # enforce strict {0,1} uint8
        X = X.astype(np.uint8)

        # Validate
        if not ((X.values == 0) | (X.values == 1)).all():
            raise ValueError("Processed dataset contains values outside {0,1}.")

        self.processed_columns = list(X.columns)

        # Write outputs
        self._export(X)

        return X

    def _export(self, X: pd.DataFrame) -> None:
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(self.output_csv_path, index=False)

        meta = self.export_metadata()
        meta_path = self.output_csv_path.with_suffix(self.output_csv_path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def export_metadata(self) -> Dict[str, Any]:
        """
        Metadata includes enough to:
        - audit preprocessing
        - rebuild visible_blocks sizes
        - decode floats approximately if desired (range + bits)
        """
        return {
            "type": "DataPreprocessor",
            "version": 1,
            "device": self.device_str,
            "bm_type": self.bm_type,
            "output_csv_path": str(self.output_csv_path),
            "q_low": self.q_low,
            "q_high": self.q_high,
            "add_missing_bit": self.add_missing_bit,
            "max_categories": self.max_categories,
            "min_category_freq": self.min_category_freq,
            "float_specs": [asdict(s) for s in self.float_specs],
            "binary_specs": [asdict(s) for s in self.bin_specs],
            "categorical_specs": [asdict(s) for s in self.cat_specs],
            "visible_blocks_out": self.visible_blocks_out,  # feature_name -> [bit cols]
            "visible_blocks_sizes": {k: len(v) for k, v in self.visible_blocks_out.items()},
            "processed_columns": self.processed_columns,
        }

    # Convenience: build RBM-ready visible_blocks dict
    def get_visible_blocks_sizes(self) -> Dict[str, int]:
        if not self.visible_blocks_out:
            raise RuntimeError("Run fit_transform() first.")
        return {k: len(v) for k, v in self.visible_blocks_out.items()}
