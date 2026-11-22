"""Multi-seed aggregation reads run artifacts and pairs B/C by seed (synthetic runs)."""

import json

from reports.seeds import collect, render


def _run(root, name, mrr, best=2):
    run = root / name
    run.mkdir()
    epochs = [{"epoch": e, "proxy": {"recall@5": 0.9, "mrr@10": mrr if e == best else 0.5,
                                     "ndcg@10": 0.8}} for e in range(4)]
    (run / "history.json").write_text(json.dumps({"epochs": epochs}), "utf-8")
    (run / "summary.json").write_text(json.dumps(
        {"best_epoch": best, "epochs_run": 3, "wall_clock_seconds": 1.0}), "utf-8")


def test_pairs_by_seed_and_ignores_unfinished_or_other_runs(tmp_path):
    _run(tmp_path, "model-b-v2-s1", 0.90)
    _run(tmp_path, "model-c-v2-s1", 0.93)
    _run(tmp_path, "model-b-v2-s2", 0.92)
    _run(tmp_path, "model-c-v1", 0.99)              # other prefix: ignored
    (tmp_path / "model-c-v2-s2").mkdir()             # unfinished: no summary.json
    data = collect(tmp_path, "v2")
    assert set(data["b"]) == {1, 2} and set(data["c"]) == {1}
    assert data["b"][1]["mrr@10"] == 0.90            # the best epoch's value, not the last
    text = render(data, "v2")
    assert "Paired seeds with both runs finished: [1]" in text
    assert "+0.0300" in text
