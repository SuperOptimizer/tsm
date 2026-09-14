"""Resumable orchestrator: prepare training data for a list of regions, stage by stage.

    cd ~/tsm_next && nohup uv run python dev/prep_regions.py \
        --regions r1,r2,r3,r4,r5,r6,r7,r8 --gpu 1 \
        >> ~/tsm-output/prep_regions.log 2>&1 &

Per region (``configs/regions/paris4_<region>.json``) three stages run in order:

  bands     ``dev/rectoverso_slab.py <config> --out <out_dir>/rectoverso``
            CPU, ~16 min for an 8192^2 slab.  Done when ``<out>/coverage.json``
            exists (the .zarr's ``zarr.json`` appears *before* the bricks are
            filled, so it is not a completion marker).
  teachers  ``tsm teacher <config>`` for the models this region needs --
            ``lasagna`` everywhere plus ``fiber`` for ``--fiber`` (default
            r2,r4,r6,r8).  Serialised on one GPU (``--gpu``).  A model is done
            when ``<teachers_dir>/<model>.summary.json`` exists; if only some
            models are outstanding a derived config listing just those is
            written to ``<out_dir>/prep/<region>_teacher.json``.
  labels    ``tsm labels <config>``, CPU, ~8 h for an 8192^2 slab, resumable
            brick-wise.  Up to ``--labels-parallel`` regions at a time, each
            pinned with ``taskset`` to a disjoint core range (``--reserve-cores``
            cores at the top of the machine are left for the GPU trainers'
            loaders).  Done when ``<out_dir>/labels/labels.summary.json`` exists.
            A derived config (``<out_dir>/prep/<region>_labels.json``) pins
            ``extra.labels.workers`` and ``workers_ram_bytes`` so that N
            concurrent label runs cannot each claim the whole box's RAM.

bands -> teachers -> labels pipeline per region; the teacher stage is the only
one on the GPU and is serialised, label runs of earlier regions overlap the
teacher run of later ones.  Everything is idempotent: rerunning skips finished
stages, and the underlying stages resume brick-wise inside an unfinished one.

Progress markers ``PREP_<region>_<stage>_{START,DONE,FAILED}`` are appended to
``--log`` (default ``~/tsm-output/desk_eval_chain.log``); ``--status`` prints the
table and exits, ``--dry-run`` prints the plan and the resolved commands.

The script only ever touches ``<out_dir>`` of the regions it is given and runs
the stages out of the checkout it was invoked from -- it never signals or
inspects any other process.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = ROOT / "configs" / "regions"
DEFAULT_LOG = Path.home() / "tsm-output" / "desk_eval_chain.log"
DEFAULT_FIBER = "r2,r4,r6,r8"
STAGES = ("bands", "teachers", "labels")
GIB = 1 << 30


# --------------------------------------------------------------------------- #
# regions
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    name: str
    config: Path
    raw: dict
    out_dir: Path

    @property
    def teachers_dir(self) -> Path:
        t = self.raw.get("extra", {}).get("labels", {}).get("teachers_dir")
        return Path(t) if t else self.out_dir / "teachers"

    @property
    def rv_dir(self) -> Path:
        """Directory dev/rectoverso_slab.py writes into (parent of rectoverso.zarr)."""
        s = self.raw.get("extra", {}).get("labels", {}).get("faces", {}).get("rectoverso_store")
        return Path(s).parent if s else self.out_dir / "rectoverso"

    @property
    def labels_dir(self) -> Path:
        return self.out_dir / "labels"

    @property
    def prep_dir(self) -> Path:
        return self.out_dir / "prep"

    @property
    def config_models(self) -> list[str]:
        return [str(m) for m in self.raw.get("extra", {}).get("teacher", {}).get("models", [])]

    @property
    def labels_workers(self) -> int:
        return int(self.raw.get("extra", {}).get("labels", {}).get("workers", 1) or 1)


def region_config_path(config_dir: Path, name: str, prefix: str = "paris4_") -> Path:
    p = Path(config_dir) / f"{prefix}{name}.json"
    return p if p.exists() else Path(config_dir) / f"{name}.json"


def load_regions(names, config_dir: Path = DEFAULT_CONFIG_DIR, prefix: str = "paris4_") -> list[Region]:
    out = []
    for name in names:
        p = region_config_path(Path(config_dir), name, prefix)
        if not p.exists():
            raise FileNotFoundError(f"no config for region {name!r} under {config_dir}")
        raw = json.loads(p.read_text())
        out.append(Region(name=name, config=p, raw=raw, out_dir=Path(raw["out_dir"])))
    return out


def parse_regions_arg(s: str) -> list[str]:
    return [v.strip() for v in str(s).split(",") if v.strip()]


# --------------------------------------------------------------------------- #
# core partitioning
# --------------------------------------------------------------------------- #
def usable_cores(total: int, reserve: int) -> int:
    """Cores the orchestrator may pin work to: the low ``total - reserve`` ones."""
    return max(1, int(total) - max(0, int(reserve)))


def core_ranges(total: int, reserve: int, slots: int) -> list[tuple[int, int]]:
    """``slots`` disjoint inclusive core ranges over cores ``0 .. total-reserve-1``."""
    usable = usable_cores(total, reserve)
    slots = max(1, min(int(slots), usable))
    base, rem = divmod(usable, slots)
    out, lo = [], 0
    for i in range(slots):
        n = base + (1 if i < rem else 0)
        out.append((lo, lo + n - 1))
        lo += n
    return out


def cpuspec(rng: tuple[int, int]) -> str:
    lo, hi = rng
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def mem_total_bytes() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


#: measured peak live RAM of one fine label brick for these regions -- a 128x256x256 core
#: brick with halo 48 (haloed 224x352x352), faces + rectoverso + winding_fine channels, from
#: ``tsm labels <config> --dry-run`` (tsm.labels._fine_budget_rows).  Used only to pick a
#: default ``labels.workers`` that fits the per-run RAM share; ``tsm labels`` re-checks it.
FINE_BRICK_GIB = 5.9


def fit_labels_workers(ram_bytes: int, want: int, brick_gib: float = FINE_BRICK_GIB) -> int:
    """Largest ``workers <= want`` whose ``workers x brick_gib`` fits ``ram_bytes``."""
    if ram_bytes <= 0:
        return max(1, int(want))
    return max(1, min(int(want), int(ram_bytes // int(brick_gib * GIB))))


def labels_ram_bytes(parallel: int, reserve_gib: float = 8.0, total: int | None = None) -> int:
    """Per-label-run ceiling for ``workers x per-brick RAM`` (see labels.workers_ram_bytes)."""
    tot = mem_total_bytes() if total is None else int(total)
    if tot <= 0:
        return 0
    usable = max(GIB, tot - int(reserve_gib * GIB))
    return int(usable // max(1, int(parallel)))


# --------------------------------------------------------------------------- #
# skip logic
# --------------------------------------------------------------------------- #
def teacher_models(region: Region, fiber_regions) -> list[str]:
    """Models this region needs: ``lasagna`` everywhere, ``fiber`` for the chosen subset.

    The region configs list the same thing in ``extra.teacher.models`` so the plan is
    explicit there too; ``--fiber`` is what actually decides, and a mismatch is only a
    note in the plan (``teacher_models_note``).
    """
    return ["lasagna"] + (["fiber"] if region.name in set(fiber_regions) else [])


def teacher_models_note(region: Region, models) -> str:
    cfg = region.config_models
    if list(models) == cfg:
        return ""
    return f" [config lists {','.join(cfg) or 'nothing'}]"


def teacher_out_name(model: str) -> str:
    """``m7@l2`` writes ``m7_l2.summary.json``; everything else uses its own name."""
    name, _, shift = str(model).partition("@l")
    return f"{name}_l{shift}" if shift else name


def missing_models(region: Region, models) -> list[str]:
    return [m for m in models
            if not (region.teachers_dir / f"{teacher_out_name(m)}.summary.json").exists()]


def stage_done(region: Region, stage: str, models=()) -> bool:
    if stage == "bands":
        return (region.rv_dir / "coverage.json").exists()
    if stage == "teachers":
        return not missing_models(region, models)
    if stage == "labels":
        return (region.labels_dir / "labels.summary.json").exists()
    raise ValueError(f"unknown stage {stage!r}")


# --------------------------------------------------------------------------- #
# derived configs
# --------------------------------------------------------------------------- #
def teacher_config(region: Region, models) -> tuple[Path, dict | None]:
    """(config path, derived doc or None) for a teacher run over exactly ``models``."""
    if list(models) == region.config_models:
        return region.config, None
    doc = copy.deepcopy(region.raw)
    doc.setdefault("extra", {}).setdefault("teacher", {})["models"] = list(models)
    return region.prep_dir / f"{region.name}_teacher.json", doc


def labels_config(region: Region, workers: int, ram_bytes: int) -> tuple[Path, dict | None]:
    """(config path, derived doc or None) for a label run with pinned worker/RAM limits."""
    lab = region.raw.get("extra", {}).get("labels", {})
    if int(workers) == int(lab.get("workers", 1) or 1) and not ram_bytes:
        return region.config, None
    doc = copy.deepcopy(region.raw)
    d = doc.setdefault("extra", {}).setdefault("labels", {})
    d["workers"] = int(workers)
    if ram_bytes:
        d["workers_ram_bytes"] = int(ram_bytes)
    return region.prep_dir / f"{region.name}_labels.json", doc


def write_config(path: Path, doc: dict | None) -> Path:
    if doc is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    region: str
    stage: str
    action: str                       # "run" | "skip"
    reason: str = ""
    cmd: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    config: Path | None = None
    derived: dict | None = None
    log: Path | None = None
    detail: str = ""

    def shell(self) -> str:
        env = " ".join(f"{k}={v!r}" if " " in v else f"{k}={v}" for k, v in sorted(self.env.items()))
        return (env + " " if env else "") + " ".join(self.cmd)


def _taskset(spec: str | None) -> list[str]:
    return ["taskset", "-c", spec] if spec else []


def bands_job(region: Region, cpus: str | None, workers: int, python: str | None = None) -> Job:
    cmd = _taskset(cpus) + [python or sys.executable, str(ROOT / "dev" / "rectoverso_slab.py"), str(region.config),
                            "--out", str(region.rv_dir), "--workers", str(int(workers))]
    return Job(region.name, "bands", "run", cmd=cmd, env={"CUDA_VISIBLE_DEVICES": ""},
               config=region.config, log=region.prep_dir / "bands.log",
               detail=f"workers={workers} cpus={cpus or 'all'}")


def teachers_job(region: Region, models, gpu: int, python: str | None = None) -> Job:
    path, doc = teacher_config(region, models)
    cmd = [python or sys.executable, "-m", "tsm", "teacher", str(path)]
    return Job(region.name, "teachers", "run", cmd=cmd, env={"CUDA_VISIBLE_DEVICES": str(gpu)},
               config=path, derived=doc, log=region.prep_dir / "teachers.log",
               detail="models=" + ",".join(models) + f" gpu={gpu}")


def labels_job(region: Region, cpus: str, workers: int, ram_bytes: int,
               python: str | None = None) -> Job:
    path, doc = labels_config(region, workers, ram_bytes)
    cmd = _taskset(cpus) + [python or sys.executable, "-m", "tsm", "labels", str(path)]
    detail = f"workers={workers} cpus={cpus}"
    if ram_bytes:
        detail += f" workers_ram={ram_bytes / GIB:.1f}GiB"
    return Job(region.name, "labels", "run", cmd=cmd, env={"CUDA_VISIBLE_DEVICES": ""},
               config=path, derived=doc, log=region.prep_dir / "labels.log", detail=detail)


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    jobs: list[Job]
    ranges: list[tuple[int, int]]
    bands_cpus: str
    regions: list[Region]
    labels_ram: int = 0

    def for_region(self, name: str) -> list[Job]:
        return [j for j in self.jobs if j.region == name]

    def get(self, name: str, stage: str) -> Job:
        return next(j for j in self.jobs if j.region == name and j.stage == stage)

    def runnable(self) -> list[Job]:
        return [j for j in self.jobs if j.action == "run"]


def build_plan(regions: list[Region], *, fiber_regions=(), only=None, gpu: int = 0,
               total_cores: int = 0, reserve_cores: int = 4, labels_parallel: int = 3,
               labels_workers: int | None = None, labels_ram: int | None = None,
               bands_workers: int = 8, force: bool = False,
               python: str | None = None) -> Plan:
    total_cores = int(total_cores or os.cpu_count() or 1)
    ranges = core_ranges(total_cores, reserve_cores, labels_parallel)
    bands_cpus = cpuspec((0, usable_cores(total_cores, reserve_cores) - 1))
    ram = labels_ram_bytes(labels_parallel) if labels_ram is None else int(labels_ram)
    only = set(only) if only else set(STAGES)

    jobs: list[Job] = []
    for i, r in enumerate(regions):
        models = teacher_models(r, fiber_regions)
        for stage in STAGES:
            if stage not in only:
                jobs.append(Job(r.name, stage, "skip", reason="not selected (--only)"))
                continue
            if stage == "bands":
                j = bands_job(r, bands_cpus, bands_workers, python)
                done, why = stage_done(r, "bands"), f"{r.rv_dir / 'coverage.json'} exists"
            elif stage == "teachers":
                todo = models if force else missing_models(r, models)
                j = teachers_job(r, todo or models, gpu, python)
                done = not todo
                why = "all summaries present: " + ",".join(models)
                j.detail += teacher_models_note(r, models)
                if todo and todo != models:
                    j.detail += " (resume: " + ",".join(
                        m for m in models if m not in todo) + " already done)"
            else:
                w = (int(labels_workers) if labels_workers is not None
                     else fit_labels_workers(ram, r.labels_workers))
                j = labels_job(r, cpuspec(ranges[i % len(ranges)]), w, ram, python)
                done = stage_done(r, "labels")
                why = f"{r.labels_dir / 'labels.summary.json'} exists"
            if done and not force:
                j.action, j.reason = "skip", why
            jobs.append(j)
    return Plan(jobs=jobs, ranges=ranges, bands_cpus=bands_cpus, regions=regions, labels_ram=ram)


def format_plan(plan: Plan) -> str:
    lines = [f"[prep] {len(plan.regions)} regions, label core ranges "
             + ", ".join(cpuspec(r) for r in plan.ranges)
             + f"; bands/other CPU work on {plan.bands_cpus}",
             f"[prep] RAM share per concurrent label run {plan.labels_ram / GIB:.1f} GiB "
             f"(~{FINE_BRICK_GIB:g} GiB per fine brick); MemTotal {mem_total_bytes() / GIB:.0f} GiB"]
    for r in plan.regions:
        lines.append(f"[prep] --- {r.name}  config={r.config}  out_dir={r.out_dir}")
        for j in plan.for_region(r.name):
            if j.action == "skip":
                lines.append(f"[prep]   {j.stage:<8} SKIP  {j.reason}")
            else:
                lines.append(f"[prep]   {j.stage:<8} RUN   {j.detail}")
                if j.derived is not None:
                    lines.append(f"[prep]   {'':<8}       derived config {j.config}")
                lines.append(f"[prep]   {'':<8}       $ {j.shell()}")
    n = len(plan.runnable())
    lines.append(f"[prep] {n} stage run(s) to do, {len(plan.jobs) - n} skipped")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# marker log / status
# --------------------------------------------------------------------------- #
MARKER_RE = re.compile(r"\bPREP_([A-Za-z0-9]+)_(bands|teachers|labels)_(START|DONE|FAILED)\b")


class MarkerLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, region: str, stage: str, event: str, note: str = "") -> str:
        line = (time.strftime("%Y-%m-%dT%H:%M:%S") + f" PREP_{region}_{stage}_{event}"
                + (f" {note}" if note else ""))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
        print("[prep] " + line, flush=True)
        return line

    def states(self) -> dict[tuple[str, str], str]:
        """Last event seen per (region, stage)."""
        out: dict[tuple[str, str], str] = {}
        try:
            text = self.path.read_text()
        except OSError:
            return out
        for m in MARKER_RE.finditer(text):
            out[(m.group(1), m.group(2))] = m.group(3)
        return out


def format_status(regions: list[Region], markers: dict[tuple[str, str], str],
                  fiber_regions=()) -> str:
    rows = [["region", "models", "bands", "teachers", "labels"]]
    for r in regions:
        models = teacher_models(r, fiber_regions)
        cells = []
        for stage in STAGES:
            done = stage_done(r, stage, models)
            ev = markers.get((r.name, stage), "")
            if done:
                cells.append("done")
            elif ev == "START":
                cells.append("running")
            elif ev == "FAILED":
                cells.append("FAILED")
            else:
                cells.append("pending")
            if stage == "teachers" and not done:
                miss = missing_models(r, models)
                if len(miss) != len(models):
                    cells[-1] += "(" + ",".join(miss) + ")"
        rows.append([r.name, ",".join(models), *cells])
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip() for row in rows)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run_job(job: Job, log: MarkerLog) -> int:
    write_config(Path(job.config), job.derived)
    if job.log is not None:
        job.log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(job.env)
    log.write(job.region, job.stage, "START", job.detail)
    t0 = time.perf_counter()
    try:
        with open(job.log or os.devnull, "a") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {job.shell()}\n")
            fh.flush()
            rc = subprocess.call(job.cmd, cwd=str(ROOT), env=env, stdout=fh,
                                 stderr=subprocess.STDOUT)
    except Exception as exc:  # launch failure is just another failed stage
        log.write(job.region, job.stage, "FAILED", f"{type(exc).__name__}: {exc}")
        return 1
    dt = time.perf_counter() - t0
    if rc == 0:
        log.write(job.region, job.stage, "DONE", f"{dt:.0f}s")
    else:
        log.write(job.region, job.stage, "FAILED", f"rc={rc} after {dt:.0f}s, see {job.log}")
    return rc


def run_plan(plan: Plan, log: MarkerLog, labels_parallel: int = 3) -> int:
    """bands -> teachers -> labels per region; bands and teachers serialised, labels parallel."""
    gates = {"bands": threading.Semaphore(1), "teachers": threading.Semaphore(1),
             "labels": threading.Semaphore(max(1, int(labels_parallel)))}
    failures: list[str] = []
    flock = threading.Lock()

    def one(region: Region) -> None:
        for stage in STAGES:
            job = plan.get(region.name, stage)
            if job.action == "skip":
                print(f"[prep] {region.name}/{stage}: skip ({job.reason})", flush=True)
                continue
            with gates[stage]:
                rc = run_job(job, log)
            if rc != 0:
                with flock:
                    failures.append(f"{region.name}/{stage}")
                return

    threads = [threading.Thread(target=one, args=(r,), name=f"prep-{r.name}") for r in plan.regions]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        print("[prep] FAILED stages: " + ", ".join(sorted(failures)), flush=True)
        return 1
    print("[prep] all stages done", flush=True)
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prep_regions.py",
        description="Resumable bands -> teachers -> labels orchestrator for a list of regions.")
    p.add_argument("--regions", default="r1,r2,r3,r4,r5,r6,r7,r8",
                   help="comma-separated region names (default: r1..r8)")
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--config-prefix", default="paris4_")
    p.add_argument("--fiber", default=DEFAULT_FIBER,
                   help=f"regions that also get the fibre teacher (default {DEFAULT_FIBER}; "
                        "'' for none)")
    p.add_argument("--only", choices=STAGES, action="append",
                   help="run only these stages (repeatable)")
    p.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES for the teacher stage")
    p.add_argument("--labels-parallel", type=int, default=3)
    p.add_argument("--labels-workers", type=int, default=None,
                   help="override extra.labels.workers for the label runs")
    p.add_argument("--labels-ram-gib", type=float, default=None,
                   help="per-label-run ceiling for workers x per-brick RAM "
                        "(default: (MemTotal - 8 GiB) / --labels-parallel)")
    p.add_argument("--bands-workers", type=int, default=8)
    p.add_argument("--reserve-cores", type=int, default=4,
                   help="cores at the top of the machine left unpinned for the GPU trainers")
    p.add_argument("--cores", type=int, default=0, help="core count to partition (default: nproc)")
    p.add_argument("--log", default=str(DEFAULT_LOG), help="marker log to append to")
    p.add_argument("--status", action="store_true", help="print the status table and exit")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the commands only")
    p.add_argument("--force", action="store_true", help="ignore the done markers and rerun")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    names = parse_regions_arg(args.regions)
    regions = load_regions(names, Path(args.config_dir), args.config_prefix)
    fiber = parse_regions_arg(args.fiber) if args.fiber else []
    log = MarkerLog(Path(args.log))

    if args.status:
        print(format_status(regions, log.states(), fiber))
        return 0

    plan = build_plan(
        regions, fiber_regions=fiber, only=args.only, gpu=args.gpu,
        total_cores=args.cores, reserve_cores=args.reserve_cores,
        labels_parallel=args.labels_parallel, labels_workers=args.labels_workers,
        labels_ram=None if args.labels_ram_gib is None else int(args.labels_ram_gib * GIB),
        bands_workers=args.bands_workers, force=args.force)
    print(format_plan(plan), flush=True)
    if args.dry_run:
        print("[prep] dry run: nothing executed")
        return 0
    if not plan.runnable():
        return 0
    return run_plan(plan, log, args.labels_parallel)


if __name__ == "__main__":
    sys.exit(main())
