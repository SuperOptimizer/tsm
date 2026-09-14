"""Plan / skip / core-partition logic of dev/prep_regions.py (no network, no GPU, no subprocess)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "prep_regions", os.path.join(ROOT, "dev", "prep_regions.py"))
pr = importlib.util.module_from_spec(_spec)
sys.modules["prep_regions"] = pr
_spec.loader.exec_module(pr)

PY = "/fake/python"


def make_config(tmp_path, name, models=("lasagna",), workers=8):
    out = tmp_path / "out" / name
    doc = {
        "volume": {"url": "https://example.invalid/v.zarr", "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [256, 8192, 8192]},
        "budget": {"ram_bytes": 24 << 30, "array_bytes": 4 << 30},
        "out_dir": str(out),
        "extra": {
            "teacher": {"models": list(models)},
            "labels": {
                "teachers_dir": str(out / "teachers"),
                "require_teachers": [],
                "workers": workers,
                "faces": {"enabled": True, "source": "rectoverso",
                          "rectoverso_store": str(out / "rectoverso" / "rectoverso.zarr")},
            },
        },
    }
    d = tmp_path / "configs"
    d.mkdir(exist_ok=True)
    (d / f"paris4_{name}.json").write_text(json.dumps(doc, indent=2))
    return d, out


@pytest.fixture
def two_regions(tmp_path):
    d, _ = make_config(tmp_path, "r1", models=("lasagna",))
    make_config(tmp_path, "r2", models=("lasagna", "fiber"))
    return d, pr.load_regions(["r1", "r2"], d)


# --------------------------------------------------------------------------- #
# core partitioning
# --------------------------------------------------------------------------- #
def test_core_ranges_are_disjoint_and_respect_the_reserve():
    rngs = pr.core_ranges(32, 4, 3)
    assert rngs == [(0, 9), (10, 18), (19, 27)]
    assert pr.usable_cores(32, 4) == 28
    seen = [c for lo, hi in rngs for c in range(lo, hi + 1)]
    assert sorted(seen) == list(range(28))          # disjoint, no reserved core used
    assert max(seen) < 32 - 4


@pytest.mark.parametrize("total,reserve,slots", [(24, 4, 3), (32, 4, 1), (8, 4, 4), (6, 4, 3)])
def test_core_ranges_partition_exactly(total, reserve, slots):
    rngs = pr.core_ranges(total, reserve, slots)
    usable = pr.usable_cores(total, reserve)
    seen = [c for lo, hi in rngs for c in range(lo, hi + 1)]
    assert sorted(seen) == list(range(usable))
    assert len(rngs) == min(slots, usable)
    assert max(hi - lo for lo, hi in rngs) - min(hi - lo for lo, hi in rngs) <= 1


def test_core_ranges_never_empty_even_when_the_reserve_eats_the_box():
    assert pr.core_ranges(4, 8, 3) == [(0, 0)]
    assert pr.cpuspec((0, 0)) == "0"
    assert pr.cpuspec((3, 9)) == "3-9"


def test_labels_ram_share_and_worker_fit():
    assert pr.labels_ram_bytes(3, reserve_gib=8, total=128 * pr.GIB) == (120 * pr.GIB) // 3
    assert pr.labels_ram_bytes(1, reserve_gib=8, total=0) == 0
    # a 40 GiB share holds 6 bricks of 5.9 GiB, but never more than asked for
    assert pr.fit_labels_workers(40 * pr.GIB, 8) == 6
    assert pr.fit_labels_workers(40 * pr.GIB, 4) == 4
    assert pr.fit_labels_workers(2 * pr.GIB, 8) == 1      # always at least one worker
    assert pr.fit_labels_workers(0, 8) == 8               # unknown RAM -> trust the config


# --------------------------------------------------------------------------- #
# plan construction
# --------------------------------------------------------------------------- #
def test_teacher_models_follow_the_fiber_subset(two_regions):
    _, regions = two_regions
    r1, r2 = regions
    assert pr.teacher_models(r1, ["r2"]) == ["lasagna"]
    assert pr.teacher_models(r2, ["r2"]) == ["lasagna", "fiber"]
    assert pr.teacher_models(r2, []) == ["lasagna"]
    # a mismatch with the config's own list is reported, not silently obeyed
    assert pr.teacher_models_note(r2, ["lasagna"]) == " [config lists lasagna,fiber]"
    assert pr.teacher_models_note(r2, ["lasagna", "fiber"]) == ""


def test_plan_resolves_one_command_per_stage(two_regions):
    _, regions = two_regions
    plan = pr.build_plan(regions, fiber_regions=["r2"], gpu=1, total_cores=32,
                         reserve_cores=4, labels_parallel=3, labels_ram=48 * pr.GIB,
                         python=PY)
    assert [j.action for j in plan.jobs] == ["run"] * 6

    bands = plan.get("r1", "bands")
    assert bands.cmd[:3] == ["taskset", "-c", "0-27"]
    assert bands.cmd[3:5] == [PY, os.path.join(ROOT, "dev", "rectoverso_slab.py")]
    assert "--out" in bands.cmd and bands.cmd[bands.cmd.index("--out") + 1].endswith("/rectoverso")
    assert bands.env["CUDA_VISIBLE_DEVICES"] == ""
    assert bands.derived is None                      # bands runs off the region config itself

    teach = plan.get("r2", "teachers")
    assert teach.cmd == [PY, "-m", "tsm", "teacher", str(regions[1].config)]
    assert teach.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert teach.derived is None                      # config already lists lasagna,fiber

    lab = plan.get("r2", "labels")
    assert lab.cmd[:3] == ["taskset", "-c", "10-18"]  # second region -> second core range
    assert lab.cmd[3:7] == [PY, "-m", "tsm", "labels"]
    assert lab.env["CUDA_VISIBLE_DEVICES"] == ""
    assert lab.derived["extra"]["labels"]["workers"] == 8
    assert lab.derived["extra"]["labels"]["workers_ram_bytes"] == 48 * pr.GIB
    assert str(lab.config).endswith("prep/r2_labels.json")


def test_plan_writes_a_derived_teacher_config_only_for_a_restricted_model_set(two_regions):
    _, regions = two_regions
    plan = pr.build_plan(regions, fiber_regions=[], total_cores=32, python=PY)
    teach = plan.get("r2", "teachers")
    assert teach.derived is not None
    assert teach.derived["extra"]["teacher"]["models"] == ["lasagna"]
    assert str(teach.config).endswith("prep/r2_teacher.json")
    # and nothing is written until the job actually runs
    assert not os.path.exists(teach.config)
    pr.write_config(teach.config, teach.derived)
    assert json.loads(teach.config.read_text())["extra"]["teacher"]["models"] == ["lasagna"]


def test_core_ranges_wrap_when_more_regions_than_slots(tmp_path):
    for k in range(1, 5):
        d, _ = make_config(tmp_path, f"r{k}")
    regions = pr.load_regions(["r1", "r2", "r3", "r4"], d)
    plan = pr.build_plan(regions, total_cores=32, labels_parallel=3, python=PY)
    specs = [plan.get(f"r{k}", "labels").cmd[2] for k in range(1, 5)]
    assert specs == ["0-9", "10-18", "19-27", "0-9"]


def test_only_filters_stages(two_regions):
    _, regions = two_regions
    plan = pr.build_plan(regions, only=["labels"], total_cores=32, python=PY)
    assert [j.stage for j in plan.runnable()] == ["labels", "labels"]
    assert plan.get("r1", "bands").reason == "not selected (--only)"


# --------------------------------------------------------------------------- #
# skip logic on synthetic markers
# --------------------------------------------------------------------------- #
def test_stage_done_uses_the_real_completion_markers(two_regions):
    _, regions = two_regions
    r = regions[1]
    models = ["lasagna", "fiber"]
    assert not pr.stage_done(r, "bands")
    assert not pr.stage_done(r, "teachers", models)
    assert not pr.stage_done(r, "labels")

    # the .zarr appears before its bricks are filled -- only coverage.json means done
    (r.rv_dir / "rectoverso.zarr").mkdir(parents=True)
    (r.rv_dir / "rectoverso.zarr" / "zarr.json").write_text("{}")
    assert not pr.stage_done(r, "bands")
    (r.rv_dir / "coverage.json").write_text("{}")
    assert pr.stage_done(r, "bands")

    r.teachers_dir.mkdir(parents=True)
    (r.teachers_dir / "lasagna.summary.json").write_text("{}")
    assert pr.missing_models(r, models) == ["fiber"]
    assert not pr.stage_done(r, "teachers", models)
    (r.teachers_dir / "fiber.summary.json").write_text("{}")
    assert pr.stage_done(r, "teachers", models)

    r.labels_dir.mkdir(parents=True)
    (r.labels_dir / "labels.summary.json").write_text("{}")
    assert pr.stage_done(r, "labels")


def test_teacher_out_name_handles_the_level_shift_suffix():
    assert pr.teacher_out_name("lasagna") == "lasagna"
    assert pr.teacher_out_name("m7@l2") == "m7_l2"


def test_plan_skips_finished_stages_and_resumes_partial_teachers(two_regions):
    _, regions = two_regions
    r1, r2 = regions
    (r1.rv_dir).mkdir(parents=True)
    (r1.rv_dir / "coverage.json").write_text("{}")
    (r1.labels_dir).mkdir(parents=True)
    (r1.labels_dir / "labels.summary.json").write_text("{}")
    (r2.teachers_dir).mkdir(parents=True)
    (r2.teachers_dir / "lasagna.summary.json").write_text("{}")

    plan = pr.build_plan(regions, fiber_regions=["r2"], total_cores=32, python=PY)
    assert plan.get("r1", "bands").action == "skip"
    assert plan.get("r1", "teachers").action == "run"
    assert plan.get("r1", "labels").action == "skip"
    assert "labels.summary.json" in plan.get("r1", "labels").reason

    t2 = plan.get("r2", "teachers")
    assert t2.action == "run"
    assert t2.derived["extra"]["teacher"]["models"] == ["fiber"]   # lasagna already done
    assert "resume" in t2.detail


def test_force_reruns_everything(two_regions):
    _, regions = two_regions
    r1 = regions[0]
    r1.rv_dir.mkdir(parents=True)
    (r1.rv_dir / "coverage.json").write_text("{}")
    plan = pr.build_plan(regions, total_cores=32, force=True, python=PY)
    assert all(j.action == "run" for j in plan.jobs)


# --------------------------------------------------------------------------- #
# marker log / status / dry run
# --------------------------------------------------------------------------- #
def test_marker_log_round_trip(tmp_path, capsys):
    log = pr.MarkerLog(tmp_path / "sub" / "chain.log")
    log.write("r2", "bands", "START", "workers=8")
    log.write("r2", "bands", "DONE", "961s")
    log.write("r2", "teachers", "START")
    log.write("r3", "labels", "FAILED", "rc=1")
    text = (tmp_path / "sub" / "chain.log").read_text()
    assert "PREP_r2_bands_START workers=8" in text
    assert "PREP_r2_bands_DONE 961s" in text
    assert log.states() == {("r2", "bands"): "DONE", ("r2", "teachers"): "START",
                            ("r3", "labels"): "FAILED"}
    assert pr.MarkerLog(tmp_path / "missing.log").states() == {}


def test_status_table(two_regions, tmp_path):
    _, regions = two_regions
    r1, r2 = regions
    r1.rv_dir.mkdir(parents=True)
    (r1.rv_dir / "coverage.json").write_text("{}")
    r2.teachers_dir.mkdir(parents=True)
    (r2.teachers_dir / "lasagna.summary.json").write_text("{}")
    markers = {("r1", "teachers"): "START", ("r2", "bands"): "FAILED"}
    table = pr.format_status(regions, markers, ["r2"])
    rows = {line.split()[0]: line for line in table.splitlines()[1:]}
    assert "done" in rows["r1"] and "running" in rows["r1"]
    assert "FAILED" in rows["r2"]
    assert "pending(fiber)" in rows["r2"]              # lasagna done, fiber outstanding
    assert "lasagna,fiber" in rows["r2"]


def test_dry_run_prints_every_resolved_command_and_touches_nothing(tmp_path, capsys, monkeypatch):
    d, _ = make_config(tmp_path, "r1")
    make_config(tmp_path, "r2", models=("lasagna", "fiber"))
    monkeypatch.setattr(sys, "executable", PY)
    rc = pr.main(["--regions", "r1,r2", "--config-dir", str(d), "--gpu", "1",
                  "--fiber", "r2", "--cores", "32", "--labels-parallel", "3",
                  "--log", str(tmp_path / "chain.log"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run: nothing executed" in out
    assert "6 stage run(s) to do, 0 skipped" in out
    assert "label core ranges 0-9, 10-18, 19-27" in out
    assert f"{PY} -m tsm teacher" in out
    assert f"{PY} -m tsm labels" in out
    assert "rectoverso_slab.py" in out
    assert "CUDA_VISIBLE_DEVICES=1" in out            # teacher pinned to --gpu
    assert "taskset -c 0-9" in out and "taskset -c 10-18" in out
    # nothing executed and no marker written
    assert not (tmp_path / "chain.log").exists()
    assert not (tmp_path / "out" / "r1" / "prep").exists()


def test_status_mode_exits_without_planning(tmp_path, capsys, monkeypatch):
    d, _ = make_config(tmp_path, "r1")
    rc = pr.main(["--regions", "r1", "--config-dir", str(d), "--status",
                  "--log", str(tmp_path / "chain.log")])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0].split() == ["region", "models", "bands", "teachers", "labels"]
    assert "pending" in out
    assert not (tmp_path / "chain.log").exists()


def test_run_job_records_start_and_failure(tmp_path, monkeypatch):
    d, out = make_config(tmp_path, "r1")
    regions = pr.load_regions(["r1"], d)
    plan = pr.build_plan(regions, total_cores=32, python=PY)
    job = plan.get("r1", "labels")
    calls = []

    def fake_call(cmd, **kw):
        calls.append((cmd, kw))
        return 7

    monkeypatch.setattr(pr.subprocess, "call", fake_call)
    log = pr.MarkerLog(tmp_path / "chain.log")
    assert pr.run_job(job, log) == 7
    assert calls[0][0] == job.cmd and calls[0][1]["cwd"] == str(pr.ROOT)
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert log.states() == {("r1", "labels"): "FAILED"}
    assert job.log.exists()                            # per-stage stdout log
    assert json.loads(job.config.read_text())["extra"]["labels"]["workers"] >= 1


def test_run_plan_stops_a_region_after_a_failed_stage(tmp_path, monkeypatch):
    d, _ = make_config(tmp_path, "r1")
    regions = pr.load_regions(["r1"], d)
    plan = pr.build_plan(regions, total_cores=32, python=PY)
    seen = []

    def fake_call(cmd, **kw):
        seen.append(" ".join(cmd))
        return 1                                       # the bands stage fails

    monkeypatch.setattr(pr.subprocess, "call", fake_call)
    rc = pr.run_plan(plan, pr.MarkerLog(tmp_path / "chain.log"), labels_parallel=2)
    assert rc == 1
    assert len(seen) == 1
    assert "rectoverso" in " ".join(seen)


def test_run_plan_runs_the_stages_in_order(tmp_path, monkeypatch):
    d, _ = make_config(tmp_path, "r1")
    regions = pr.load_regions(["r1"], d)
    plan = pr.build_plan(regions, total_cores=32, python=PY)
    seen = []
    monkeypatch.setattr(pr.subprocess, "call",
                        lambda cmd, **kw: (seen.append(cmd), 0)[1])
    assert pr.run_plan(plan, pr.MarkerLog(tmp_path / "chain.log")) == 0
    stages = ["bands" if "rectoverso_slab.py" in " ".join(c) else c[c.index("-m") + 2]
              for c in seen]
    assert stages == ["bands", "teacher", "labels"]


def test_missing_region_config_is_an_error(tmp_path):
    d, _ = make_config(tmp_path, "r1")
    with pytest.raises(FileNotFoundError):
        pr.load_regions(["r9"], d)


def test_parse_regions_arg():
    assert pr.parse_regions_arg("r1, r2 ,r3") == ["r1", "r2", "r3"]
    assert pr.parse_regions_arg("") == []
