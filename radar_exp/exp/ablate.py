"""Ablation sweeps over experiment parameters.

This module reads an ablation configuration specifying a base
experiment configuration and a parameter grid.  It enumerates all
combinations of the grid, overrides the base configuration accordingly
and executes each run via `run_experiment`.  Results are recorded to
a registry file (CSV) containing the parameter settings and summary
metrics for each run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml

from .runner import run_experiment, _override


def _generate_param_combinations(param_grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """Return a list of dictionaries for all combinations of the grid."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combo = {k: v for k, v in zip(keys, prod)}
        combos.append(combo)
    return combos


def run_ablation(ablation_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Run an ablation sweep.

    Parameters
    ----------
    ablation_cfg : dict
        Configuration dictionary.  Must contain `base_config` (path
        to base YAML file) and `param_grid` (dictionary mapping dotted
        parameter names to lists of values).  Optional keys: `name`
        and `output_root`.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per experiment.  Columns include the
        parameter names and the summary metrics returned by
        `run_experiment`.
    """
    base_path = ablation_cfg.get("base_config")
    if base_path is None:
        raise ValueError("ablation configuration must specify base_config")
    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    param_grid = ablation_cfg.get("param_grid", {})
    combos = _generate_param_combinations(param_grid)
    name_base = ablation_cfg.get("name", "ablation")
    output_root = ablation_cfg.get("output_root", "runs")
    registry_rows = []
    for i, combo in enumerate(combos):
        # Override base config with parameter combo
        cfg = _override(base_cfg, combo)
        run_name = f"{name_base}_{i}"
        print(f"Running combination {i+1}/{len(combos)}: {combo}")
        summary = run_experiment(cfg, run_name=run_name, output_root=output_root)
        row = {"run_id": summary.get("run_dir", run_name)}
        # Flatten parameter combo into row
        row.update(combo)
        # Include summary metrics
        for k, v in summary.items():
            if k != "run_dir":
                row[k] = v
        registry_rows.append(row)
    df = pd.DataFrame(registry_rows)
    # Save registry to CSV
    registry_path = Path(output_root) / "registry.csv"
    if registry_path.exists():
        # Append to existing registry
        existing = pd.read_csv(registry_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(registry_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation sweeps over radar experiment parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to ablation YAML configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        abl_cfg = yaml.safe_load(f)
    df = run_ablation(abl_cfg)
    print("Ablation completed. Results written to registry.csv")
    print(df)


if __name__ == "__main__":
    main()