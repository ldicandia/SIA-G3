from __future__ import annotations

import csv
import json

import pytest

from tp4.cli import main
from tp4.errors import ConfigError
from tp4.experiments.sweep import MAX_CHILDREN, child_name, expand_grid
from tp4.parallel import run_parallel, spawn_seeds

SWEEP = "configs/pca_scaling_sweep.json"


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cli_end_to_end_runs_both_children_in_two_workers(tmp_path):
    out = tmp_path / "runs"
    assert main(["run", SWEEP, "--out-root", str(out)]) == 0
    sweep_dir = out / "pca_scaling_sweep"
    rows = _read_csv(sweep_dir / "sweep.csv")
    assert [r["child"] for r in rows] == ["standardize-true", "standardize-false"]
    assert list(rows[0])[:4] == ["child", "params.standardize", "varied_parameter", "parameter_value"]
    assert [row["varied_parameter"] for row in rows] == ["params.standardize"] * 2
    assert [row["parameter_value"] for row in rows] == ["True", "False"]
    assert [r["params.standardize"] for r in rows] == ["True", "False"]
    for row in rows:
        child = sweep_dir / row["child"]
        run = json.loads((child / "run.json").read_text(encoding="utf-8"))
        assert run["seed"] == 42
        assert run["experiment"] == row["child"]
        assert (child / "summary.json").stat().st_size > 0
        assert (child / "biplot.png").stat().st_size > 0  # child_plots is true in the shipped config
    child_summary = json.loads((sweep_dir / rows[0]["child"] / "summary.json").read_text())
    assert float(rows[0]["pc1_explained_variance_pct"]) == child_summary["pc1_explained_variance_pct"]
    assert rows[1]["pc1_explained_variance_pct"] != rows[0]["pc1_explained_variance_pct"]
    summary = json.loads((sweep_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == {"grid_keys": ["params.standardize"], "n_children": 2, "workers": 2}


def test_sweep_csv_is_byte_identical_for_one_and_two_workers(tmp_path, write_config):
    outputs = []
    for workers in (1, 2):
        cfg = write_config(SWEEP, workers=workers, child_plots=False, plots=False)
        out = tmp_path / f"runs{workers}"
        assert main(["run", str(cfg), "--out-root", str(out)]) == 0
        outputs.append((out / "pca_scaling_sweep" / "sweep.csv").read_bytes())
    assert outputs[0] == outputs[1]
    assert outputs[0].startswith(b"child,params.standardize,")


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"grid": {}}, "grid"),
        ({"grid": {"params.standardize": []}}, "params.standardize"),
        ({"grid": {"params.standardize": [True, True]}}, "duplicate"),
        ({"grid": "nope"}, "grid"),
        ({"workers": 0}, "workers"),
        ({"child_plots": "yes"}, "child_plots"),
        ({"grid": {"params.top_n": [1, "x"]}}, "top_n"),
        ({"grid": {"params.standardize": [{"a": 1}]}}, "booleans, numbers or strings"),
    ],
)
def test_invalid_sweep_configs_exit_2_before_creating_runs(tmp_path, capsys, write_config, overrides, needle):
    cfg = write_config(SWEEP, **overrides)
    out = tmp_path / "runs"
    assert main(["run", str(cfg), "--out-root", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert needle in err
    assert not (out / "pca_scaling_sweep").exists()


def test_nested_sweep_is_refused(tmp_path, capsys):
    nested = tmp_path / "nested.json"
    nested.write_text(
        json.dumps(
            {
                "experiment": "nested",
                "kind": "sweep",
                "seed": 1,
                "base": SWEEP,
                "grid": {"params.standardize": [True]},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "runs"
    assert main(["run", str(nested), "--out-root", str(out)]) == 2
    err = capsys.readouterr().err
    assert "nested sweeps" in err
    assert not (out / "nested").exists()


def test_too_many_children_is_rejected(tmp_path, capsys, write_config):
    cfg = write_config(SWEEP, grid={
        "params.top_n": list(range(1, MAX_CHILDREN // 2 + 2)),
        "params.loadings_pcs": list(range(1, MAX_CHILDREN // 2 + 2)),
    })
    assert main(["run", str(cfg), "--out-root", str(tmp_path / "runs")]) == 2
    assert str(MAX_CHILDREN) in capsys.readouterr().err


def test_child_name_slugs_values_and_is_valid():
    assert child_name("params.standardize", True) == "standardize-true"
    assert child_name("params.lr", 0.01) == "lr-0p01"
    assert child_name("params.k", -3) == "k-m3"
    assert child_name("params.mode", "Hex Grid!") == "mode-hex_grid_"
    with pytest.raises(ConfigError):
        child_name("params.x", "a" * 80)
    with pytest.raises(ConfigError):
        child_name("params.x", None)


def test_expand_grid_varies_one_leaf_at_a_time_in_order_without_changing_base():
    base = {"experiment": "b", "seed": 1, "params": {"a": 1, "b": {"c": "x"}}}
    grid = {"params.a": [1, 2], "params.b.c": ["x", "y"]}
    children = expand_grid(base, grid, seed=7, plots=False)
    assert [name for name, _ in children] == ["a-1", "a-2", "c-x", "c-y"]
    assert [child["params"] for _, child in children] == [
        {"a": 1, "b": {"c": "x"}}, {"a": 2, "b": {"c": "x"}},
        {"a": 1, "b": {"c": "x"}}, {"a": 1, "b": {"c": "y"}},
    ]

    def leaves(node, prefix=""):
        return {path: value for key, item in node.items()
                for path, value in (leaves(item, prefix + key + ".").items() if isinstance(item, dict)
                                    else [(prefix + key, item)])}

    baseline = leaves(base)
    for _, child in children:
        current = leaves(child)
        differences = {key for key in baseline.keys() | current.keys()
                       if key not in {"experiment", "plots", "seed"} and baseline.get(key) != current.get(key)}
        assert len(differences) <= 1
    assert all(child["seed"] == 7 and child["plots"] is False for _, child in children)
    children[-1][1]["params"]["b"]["c"] = "changed"
    assert children[0][1]["params"]["b"]["c"] == "x"
    assert base == {"experiment": "b", "seed": 1, "params": {"a": 1, "b": {"c": "x"}}}
    seeded = expand_grid(base, {"seed": [3, 4], "params.a": [2]}, seed=7, plots=True)
    assert [child["seed"] for _, child in seeded] == [3, 4, 7]
    summed = expand_grid(base, {"params.a": list(range(10)), "params.b.c": list(range(10))}, seed=7, plots=False)
    assert len(summed) == 20


def test_expand_grid_rejects_colliding_names():
    with pytest.raises(ConfigError, match="duplicate"):
        expand_grid({}, {"a.x": [1], "b.x": [1]} | {"c.x": [1, 1]}, seed=0, plots=False)
    with pytest.raises(ConfigError, match="duplicate"):
        expand_grid({}, {"params.v": ["A", "a"]}, seed=0, plots=False)


def test_run_parallel_preserves_order_through_real_workers():
    assert run_parallel(abs, [-3, 1, -2, 5], 2) == [3, 1, 2, 5]


def test_run_parallel_edges():
    assert run_parallel(abs, [], 2) == []
    assert run_parallel(abs, [-1], 4) == [1]
    assert run_parallel(abs, [-1, -2], 1) == [1, 2]
    with pytest.raises(ValueError):
        run_parallel(abs, [1], 0)


def test_spawn_seeds_deterministic_and_distinct():
    a = spawn_seeds(42, 5)
    assert a == spawn_seeds(42, 5)
    assert len(set(a)) == 5
    assert a != spawn_seeds(43, 5)
    assert spawn_seeds(42, 0) == []
