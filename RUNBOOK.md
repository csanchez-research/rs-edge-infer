# RUNBOOK - rs-edge-infer

Copy-paste command guide. The *why* lives in `README.md` and the decisions note;
this is just the ordered steps. Two phases: the **DEV box** builds the artifacts,
the **TARGET** measures them. They share one artifact: the OpenVINO IR models.

The normal deploy path is `deploy.sh`, run **from the dev box**: it builds the
image locally, ships code + image to a target over SSH, runs the experiment there
and pulls the results back. The dev box is the single source of truth, so a target
needs no build toolchain or package index of its own.

Conventions: `$` = dev-box shell, `(dev)` = inside the DEV container, `(target)` =
on an N5105 / Lenovo. All Python runs **inside Docker**, never loose on the host.

---

## 0. Prerequisites

Two harness products go in `input/` by hand (gitignored):

```
input/encoder.pt         # the trained paper-1 encoder (checkpoint)
input/norm_stats.json    # per-band mean/std (needed by prepare_data pack + probe)
```

If either is missing, the scripts fail fast at startup with a clear message.

Optional local config at the repo root (both gitignored, both shipped by the
deploy rsync so they travel to the target):

```
.env            # RSEI_TASMOTA_IP=192.168.1.50   (empty/absent = RAPL only)
nexus.env       # private pip index (see README); absent = plain public PyPI
```

---

## 1. DEV box - build the artifacts (one time)

Open the repo in VS Code and **Reopen in Container** (uses
`.devcontainer/Dockerfile.dev`). Then, inside that container:

```bash
# (dev) 1. DATA: download reBEN once, carve the seeded subset (calib 512 / train 20k
#            / test 10k), pack INT8 calib + 1 real benchmark sample. --keep-archives
#            makes the final cleanup a no-op so the 117 GB stay on disk while you
#            iterate (reclaim them only at the end, step 6).
python pipeline/prepare_data.py --keep-archives
#   -> input/calib/calib_int8.npz , input/bench_sample.npz ,
#      input/reben_subset/{calib,train,test}/ , state/data/data_manifest.json

# (dev) 2. EXPORT: paper-1 encoder -> ONNX (inference mode, no masking, strict load)
python pipeline/01_export_onnx.py --checkpoint input/encoder.pt
#   -> input/<model_id>/onnx/model.onnx

# (dev) 3. QUANTIZE: build FP32 / FP16 / INT8 OpenVINO IR (INT8 reads the calib pack)
python pipeline/02_quantize.py
#   -> input/<model_id>/{fp32,fp16,int8}/model.xml (+ .bin)

# (dev) 4. VERIFY: provenance gate, sha256 chain checkpoint -> onnx -> IR
python pipeline/00_verify.py
#   -> output/verify_*.json   (exit 0 = all OK)

# (dev) 5. ACCURACY axis (dev only). NEEDS scikit-learn -> REBUILD the dev image first
#            (it is in requirements-dev.txt). Features per precision, then the probe.
python pipeline/05_extract_features.py                   # -> output/<model_id>/features/<split>_<precision>.npz
python pipeline/06_linear_probe.py --require-classes 19  # -> output/<model_id>/probe_accuracy.json (mAP per precision + delta vs fp32)

# (dev) 6. CLEANUP, only after everything above validates. Reclaims the 117 GB.
python pipeline/prepare_data.py --stage cleanup
```

Re-run a single stage of step 1 with `--stage {download,select,extract,pack,cleanup}`
(one at a time; `--stage` takes ONE value, not a list). Re-run one precision in step 3
with `--precisions int8` (or `fp32 fp16 int8`). Quick accuracy smoke:
`05 --max-train 2000 --max-test 1000` then `06 --seeds 1`.

The IR models now live under `input/<model_id>/`. `deploy.sh` ships them to the
target; you do not copy anything by hand.

---

## 2. TARGET machine - bring-up (HOST level, ONE-TIME, outside Docker)

OS/host setup for a measurement machine (N5105, Lenovo). Do it once per machine,
over SSH, **before** the first deploy. `deploy.sh` auto-installs `rsync`, `zstd`
and `tmux` if they are missing (tmux is always installed, so the unattended
`run-bg` mode is always available), but Docker and the group membership below are
bring-up prerequisites it does not fix for you.

```bash
# 1. OS: Debian 13 (Trixie), headless (SSH server only, no desktop).

# 2. Docker Engine + Compose plugin (NOT Docker Desktop), see
#    docs.docker.com/engine/install/debian ; then groups for sudo-less docker
#    and /dev/dri (iGPU):
sudo usermod -aG docker,render "$USER"      # log out/in for the groups to apply

# 3. python3 (stdlib only, the host scripts need NO pip):
python3 --version                           # measure_power + 04_merge use stdlib

# 4. RAPL readable (CPU energy). energy_uj is root-only (0400), so the orchestrator
#    runs under sudo (deploy.sh does this for you); this just confirms the counter:
ls /sys/class/powercap/intel-rapl:*/energy_uj

# 5. CPU governor + thermal (measurement stability):
sudo apt install linux-cpupower
sudo cpupower frequency-set -g performance  # Tiny PCs throttle; 03 logs thermal

# 6. iGPU axis (optional): /dev/dri must exist and OpenVINO must SEE the GPU.
ls /dev/dri                                 # card0 / renderD128 present?
#    00_test (run by the deploy below) reports igpu_available=false if the Intel
#    GPU compute runtime is missing. CPU works regardless.
```

One-time from the dev box: set up key-based SSH so the deploy is non-interactive
for the file transfer (you still type your sudo password on the target when the
measurement starts, for RAPL):

```bash
$ ssh-copy-id user@<target-ip>
```

---

## 3. Deploy + measure (from the DEV box, via deploy.sh)

`deploy.sh` builds the lean image locally for the profile, ships code + image to
the target, and runs there. One script, five modes:

```
./deploy.sh <user@host> <noavx2|avx2> [test | run | run-bg | collect | gpu-probe]
```

| Mode | What it does |
|------|--------------|
| `test` | build + ship, then the readiness gate (`00_test` + energy-sources check), live. Stops before measuring. |
| `run` | build + ship, then the full experiment in the **foreground** (you watch it; Ctrl-C stops it). |
| `run-bg` | same, inside a **tmux** session on the target, so it survives an SSH disconnect. Long unattended runs. |
| `collect` | pull the target's `output/`, `logs/` **and** `state/` back into `results/<LABEL>/`. |
| `gpu-probe` | sweep candidate GPU OpenCL runtimes (NEO): build each on the dev box, run on the target iGPU, record which compiles the model (writes `state/gpu_runtime_probe.json`). See §3.5. |

Useful env vars: `LABEL` (results subfolder, default = profile), `REMOTE_DIR`
(default `experiment/rs-edge-infer`), `TASMOTA_IP` (overrides the `.env`),
`REBUILD=1` (force a rebuild even if the image exists locally), `RERUN=1`
(re-run even if the target already holds results for this profile).

**Re-run guard:** `run` and `run-bg` stop early, before building or shipping
anything, if the target already has results for this profile (per-precision JSON
under `output/runs/`). This stops a finished measurement being redone by accident.
To measure again on purpose, pass `RERUN=1`; `test` and `collect` are never
blocked.

### 3.1 Readiness gate (always run this first on a new or changed setup)

```bash
# (dev) build for noavx2, ship, run 00_test + energy check on the N5105.
# REBUILD=1 forces a fresh image (needed after a base or code change).
REBUILD=1 LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 test
```

Green gate = OpenVINO runs (no SIGILL), iGPU seen, and at least one energy source
live (RAPL as root + Tasmota if the `.env` has an IP). You type your sudo password
on the target when prompted. `REBUILD=1` is only needed the first time or after the
image changes; afterwards you can drop it.

### 3.2 Full measurement run

```bash
# (dev) long run, detached in tmux on the target (survives SSH drop):
LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 run-bg
#   idle baseline -> per precision (fp32 fp16 int8): sampler on, benchmark, sampler off
#   -> then 04_merge_stats folds latency + energy into one summary.

# reattach to watch it; detach again with Ctrl-b then d (it keeps running):
ssh -t user@<target-ip> 'tmux attach -t rsei'

# shorter smoke run to check the flow end to end:
N_ITER=50 REPEATS=2 LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 run-bg
```

Use `run` instead of `run-bg` for a short foreground run you watch directly.

### 3.3 Bring the results back

```bash
# (dev) pull output/ + logs/ into results/<LABEL>/
LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 collect
```

Run ids are prefixed `<host>_<target>_<stamp>`, so N5105 and Lenovo results can
share one results tree without colliding.

### 3.4 Second machine (Lenovo, avx2)

Same commands, swap the profile and label. The image self-detects the CPU, but the
profile here sets `FORCE_TARGET` so the right variant is built on the dev box:

```bash
REBUILD=1 LABEL=lenovo ./deploy.sh user@<lenovo-ip> avx2 test
LABEL=lenovo ./deploy.sh user@<lenovo-ip> avx2 run-bg
LABEL=lenovo ./deploy.sh user@<lenovo-ip> avx2 collect
```

### 3.5 Find a working iGPU runtime / OpenVINO (gpu-probe)

`gpu-probe` finds which **(GPU OpenCL runtime x OpenVINO)** combination compiles the
model on the target iGPU. It builds each candidate on the dev box and compiles the
real IR for **every precision we benchmark (fp32/fp16/int8)** on the target iGPU - a
candidate PASSes only if **all** compile (int8 carries extra quant/dequant kernels, so
fp32 alone is not enough). The record is **cumulative**: already-tested candidates are
skipped, so a re-run only builds new recipes (`GPU_PROBE_FORCE=1` re-runs all, e.g. to
re-test with a changed probe). Edit the candidates in `tools/gpu_recipes.sh`.

```bash
# needs the fp32 IR built (02_quantize). Builds on dev, runs on the target iGPU.
LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 gpu-probe
LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 collect   # brings the record back in state/
```

Two outputs:
- `state/gpu_runtime_probe.json` - the full sweep: every candidate, PASS/FAIL,
  driver, OpenCL C version, OpenVINO version, **and the IR sha256** (same chain of
  custody as the manifests).
- `docker/requirements-gpu.recommended.txt` - an **advisory** pin line (e.g.
  `openvino==2024.6.0`) derived from the first PASS. It is **not** the authoritative
  pin: set that yourself, deliberately, in `requirements-{noavx2,avx2}.txt` (and, if
  the winner needs a non-stock NEO, transplant its recipe into `docker/Dockerfile`).
  Then `REBUILD=1 ... test` to confirm CPU + iGPU on the pinned version.

Finding here (N5105, Gen11): the iGPU compiles with **OpenVINO 2024.6, not 2025.x**
(2025.x regressed the LayerNorm/MVN kernel to OpenCL 2.0 work-group functions the
Gen11's OpenCL C 1.2 cannot compile). The lean image is pinned to `openvino==2024.6.0`.
If nothing passes on a given machine, the iGPU is a documented limitation there and
that machine's results stand on CPU.

On Gen8-11 the OpenCL runtime is itself a moving target. The stock Ubuntu 24.04
package (NEO 23.43, OpenCL C 1.2) cannot compile the ViT LayerNorm/MVN kernel, and
current mainline NEO (26.31.39395.13) **drops** Gen11 - Intel moved Gen8-11 to a
legacy track (see `LEGACY_PLATFORMS.md` upstream). The usable legacy runtime for
Gen8-11 is Intel's `24.35.30872.36` (with IGC `1.0.17537.24`). The exact runtime
that actually passed on your machine is recorded in `state/gpu_runtime_probe.json`,
so the pin is reproducible rather than assumed.

---

## 4. Manual path (debug one step, directly on the target)

Only for debugging a single step. Needs the project already on the target (a prior
deploy leaves it under `$REMOTE_DIR`). This builds on the target instead of shipping
a prebuilt image, so it needs a package index reachable from the target.

```bash
# (target) build the lean image for THIS cpu (auto avx2/noavx2)
docker compose build                       # force: FORCE_TARGET=noavx2 docker compose build

# (target) environment gate only
docker compose run --rm rsei test

# (target) one benchmark, one precision
docker compose run --rm rsei benchmark --model <model_id> --precision fp32

# (target) merge afterwards (--run-dir = the run folder the orchestrator made)
python3 host/04_merge_stats.py --run-dir output/runs/<host>_<target>_<stamp>
```

---

## 5. Full canonical order (reference)

```
DEV (one time):  prepare_data --keep-archives -> 01_export -> 02_quantize -> 00_verify
                                                                    |
                                   accuracy axis:  05_extract_features -> 06_linear_probe

DEPLOY (per target, from the dev box):
   deploy.sh <host> <profile> test        # gate: 00_test + energy check
   deploy.sh <host> <profile> run-bg      # 00_test -> idle -> 03_benchmark -> 04_merge_stats
   deploy.sh <host> <profile> collect     # results -> results/<LABEL>/

LAST:  prepare_data --stage cleanup        # only once everything above has validated
```
