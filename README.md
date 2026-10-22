# rs-edge-infer

Proof-of-concept harness for deploying a self-supervised remote-sensing encoder
on low-cost commodity x86 hardware, characterising the **latency / accuracy /
energy** trade-off across quantisation levels (FP32 / FP16 / INT8) and OpenVINO
back-ends (CPU vs iGPU), over a real hardware gradient (no-AVX2, AVX2 and
AVX2+VNNI x86 CPUs).

The encoder is the paper-1 MAE (STRATA): **ViT-S/16**, trained on
**paired SAR + optical** imagery in **6-channel early fusion** (optical B4/B3/B2
+ SAR VV/VH/VV−VH), 224×224 crops, mask ratio 0.70. Deployment uses the encoder
only. These values are already set in `config.py`.

---

## Built on

This repository is the **deployment study** for the self-supervised SAR–optical
encoder introduced in our companion paper (**Paper 1, STRATA-S12**). It trains
nothing: it consumes the pretrained encoder (`encoder.pt`) and mirrors the probe
recipe from the Paper-1 evaluation harness
(https://github.com/csanchez-research/strata-s12-eval-harness), and characterises
that encoder's latency / energy / accuracy at inference time on commodity x86
hardware. Paper 1: C. Sanchez-Leon, D. Chen and D. Wang, "Deficit-Driven Stratified
Acquisition for Building STRATA-S12: An Open SAR–Optical Corpus for Self-Supervised
Learning," arXiv:2610.17680, 2026 (https://doi.org/10.48550/arXiv.2610.17680).

---

## 1. Hardware under study

| Role | Machine | CPU | AVX2 | VNNI | Role in the paper |
|------|---------|-----|------|------|-------------------|
| Floor | Mini-PC Moort M1 | Intel N5105 (Jasper Lake) | **No** | No | "no-AVX2" case, rarely studied |
| Step | Lenovo M720q | Intel i3-9100T (Coffee Lake) | Yes | No | "with-AVX2" case (no VNNI) |
| Ref. | Dev. laptop | Intel i7-14700HX (Raptor Lake) | Yes | Yes | present-day reference, CPU latency only |

Energy is measured with a **Tasmota energy plug** (active power W, via local
HTTP) **and** the CPU-package **RAPL** counter. Single clock: the host's.

---

## 2. Two Docker images (and why)

There are **two separate images** because export and measurement are different
jobs with different tools, and the measurement one must stay minimal.

| | **dev image** (`.devcontainer/Dockerfile.dev`) | **lean image** (`docker/Dockerfile`) |
|---|---|---|
| Job | prep data (prepare_data) + convert (01_export, 02_quantize) + accuracy probe (05_extract_features, 06_linear_probe) + smoke-test | measure latency/energy (00_test, 03_benchmark) |
| Deps | torch/timm/openvino/nncf + data prep (rasterio/pandas/zstandard + aria2) + scikit-learn (probe) + ruff (heavy) | openvino + numpy + ujson only (minimal) |
| Runs on | your dev PC (Windows/WSL via VS Code) | the N5105 / Lenovos |
| AVX2 | doesn't care (not measuring) | self-detects avx2/noavx2 |

Why not one image:
- **torch can't run on the N5105** (its wheels are AVX2-compiled → SIGILL on
  Jasper Lake). So the measurement image must not contain torch.
- **Scientific validity:** the paper measures *minimal-footprint* inference. A
  fat image with torch loaded would be a different system, not the edge
  deployment claimed.

Shared artifact between them: the **IR models** (`input/<model_id>/<precision>/
model.xml`). The dev image produces them from `encoder.pt`; the benchmark step
(03) consumes them and runs in **both** images - dev to test on your PC (Phase 1),
lean to measure on the mini-PC (Phase 2). Flow:
**encoder.pt → (convert, dev only) → IR → (measure, dev or lean).**

For Phase 1 on your PC you only need the **dev image**; the lean image comes in
only when you move to the N5105.

---

## 3. Architecture decisions

- **Host Debian Trixie, container Ubuntu 24.04.** The host provides the kernel
  and the i915 iGPU driver; the container provides OpenVINO (officially
  supported on Ubuntu). Different layers, which is why Debian on the host is
  fine.
- **One Python dialect across the project (3.11+).** The dev image is
  `python:3.11-slim` and is the reference: every artifact (IR, features,
  accuracy) was produced with it. The lean image sits on Ubuntu 24.04
  (Python 3.12 >= 3.11), so the same code runs everywhere (e.g. `datetime.UTC`)
  and `ruff` targets `py311`. The lean Python is only glue (timestamps, JSON,
  the timing loop); the measured numbers come from OpenVINO's C++ runtime, not
  from the Python version.
- **One self-detecting image.** At build time the Dockerfile tests the CPU
  (`/proc/cpuinfo`), installs the matching profile - `noavx2` or `avx2` - and
  writes the choice to `/opt/rsei_target`, baked immutably into the image; the
  entrypoint exports it as `RSEI_TARGET`. The same benchmark code runs on both;
  only the OpenVINO install differs. Build the image **on the machine you will
  run it on** (build on the N5105 -> noavx2; on a Lenovo -> avx2); `00_test`
  re-checks at run time and fails loudly on a mismatch. Force a profile with
  `--build-arg FORCE_TARGET=noavx2` (e.g. cross-building).
- **OpenVINO pinned to 2024.6 (lean image).** It supports pre-AVX2 CPUs (2026.0+
  requires AVX2) AND compiles the Jasper Lake (Gen11) iGPU kernel: OpenVINO 2025.x
  regressed the ViT LayerNorm/MVN kernel to OpenCL 2.0 work-group functions the
  Gen11's OpenCL C 1.2 cannot compile. This was bisected with the `gpu-probe` sweep
  (2024.6 PASS / 2025.1 FAIL, stock NEO unchanged; see RUNBOOK §3.5). The dev box
  still builds the IR with its own OpenVINO; 2024.6 reads it fine.
- **Energy measured from the host**, wrapping the container. RAPL lives on the
  host; cross-referencing by timestamp avoids permission headaches and keeps a
  single clock.
- **Pinned dependencies.** All Python deps are fully pinned (`==`) in
  `docker/requirements-*.txt` for reproducibility.
- **Central `config.py`.** Single source of truth (channels, input mode, patch
  size, precisions, devices, protocol, paths, plug IP). Every value is
  overridable via `RSEI_*` environment variables. A snapshot of the config,
  plus full system metadata, is embedded into every result file.

---

## 4. Data flow

```
                 +------------ HOST (Debian) -------------+
 run_experiment  |  detect AVX2 -> build image            |
                 |  00_test (gate) - does OpenVINO run?   |
                 |  idle baseline  - measure_power.py     |
    per precision|  [ measure_power.py ] <-> docker run --+--> container: 03_benchmark
                 |        (RAPL + Tasmota, host clock)    |        (latency, JSON)
                 +----------------------------------------+
 04_merge_stats: cross-ref JSON(latency) + CSV(energy) by timestamp -> summary.csv
```

---

## 5. Configuration (`config.py`)

Key values you may need to set before running:

| Setting | Env override | Default | Notes |
|---------|-------------|---------|-------|
| SAR channels | `RSEI_SAR_CH` | 3 | paper 1 |
| Optical channels | `RSEI_OPT_CH` | 3 | paper 1 |
| Input mode | `RSEI_INPUT_MODE` | `stacked` | early fusion, single 6-ch input (paper 1) |
| Patch H / W | `RSEI_INPUT_H` / `RSEI_INPUT_W` | 224 / 224 | paper-1 crop size |
| Precisions | `RSEI_PRECISIONS` | `fp32 fp16 int8` | |
| Devices | `RSEI_DEVICES` | `CPU GPU` | |
| Iterations | `RSEI_N_ITER` | 200 | measured |
| Warmup | `RSEI_WARMUP` | 20 | discarded |
| Tasmota IP | `RSEI_TASMOTA_IP` | (empty) | empty = RAPL only |
| Idle seconds | `RSEI_IDLE_SECS` | 60 | baseline duration |

---

## 6. Usage

### Artifact layout (every step identified by model + precision)
```
input/
  encoder.pt                      # harness product - drop it here (gitignored)
  norm_stats.json                 # harness product - band mean/std (needed by prepare_data pack)
  calib/   calib_int8.npz         # INT8 calibration pack (prepare_data, from reBEN validation)
  bench_sample.npz                # 1 real test patch, for 03 on the lean target
  reben_subset/{train,test}/      # extracted reBEN subset (accuracy axis; shared, calib tree is cleaned)
  <model_id>/                     # MODEL_ID (default: checkpoint stem, e.g. "encoder")
    onnx/   model.onnx
    fp32/   model.xml + model.bin
    fp16/   model.xml + model.bin
    int8/   model.xml + model.bin
state/
  <model_id>/
    manifest.json                 # provenance: sha256 of every artifact
  data/
    data_manifest.json            # data provenance (zenodo record, md5s, seed, counts, pack sha256)
output/
  runs/
    <run_id>/                     # one self-contained folder per orchestrated run
      host_env.json  envtest.json  summary.csv
      bench_<precision>.json      # latency per device (03)
      power_<label>.csv           # energy: idle + per precision (host sampler)
  <model_id>/                     # per-model products (vary by model/precision, NOT by PC)
    features/<split>_<precision>.npz  # cached 384-d features (05)
    probe_accuracy.json           # accuracy axis: micro/macro-mAP per precision + delta vs fp32 (06)
```
Transformed models live under `input/` (what the benchmark consumes); results
under `output/`; the provenance manifest under `state/`. Set `RSEI_MODEL_ID` to
name a model explicitly. `input/`, `state/` and `output/` are all gitignored
(data, not code) - `state/` is the local source of truth of the data, kept
across the whole project rather than wiped between runs.

### One-time preparation (runs in the DEV container, not on the N5105)
All dev-box steps run inside the dev image (`.devcontainer/Dockerfile.dev`,
deps from `docker/requirements-dev.txt`): open the repo in VS Code and "Reopen
in Container", then run them there. Two inputs come from the eval harness
(separate project) and are dropped into `input/`: `encoder.pt` and
`norm_stats.json`.
```bash
# Download reBEN once, carve the seeded subset (calib 512 / train 20k / test 10k),
# pack the INT8 calib set AND one real benchmark sample. --keep-archives makes the
# final cleanup a no-op, so the 117 GB archives (and the calib tree) stay on disk
# while you iterate; reclaim them only at the very end (last command below).
python pipeline/prepare_data.py --keep-archives
#   -> input/calib/calib_int8.npz , input/bench_sample.npz ,
#      input/reben_subset/{calib,train,test}/ , state/data/data_manifest.json

# Export the paper-1 encoder (inference mode, no masking) to ONNX
python pipeline/01_export_onnx.py --checkpoint input/encoder.pt
#   -> input/<model_id>/onnx/model.onnx

# Generate fp32 / fp16 / int8  (int8 reads the calib pack above, subset_size = pack size)
python pipeline/02_quantize.py
#   -> input/<model_id>/<precision>/model.xml

# Verify the artifacts match the provenance manifest (integrity gate)
python pipeline/00_verify.py
#   -> exit 0 if all recorded sha256 match; writes output/verify_*.json

# Accuracy axis (dev only): features per precision, then the frozen linear probe.
# NOTE: needs scikit-learn -> REBUILD the dev image first (it is in requirements-dev.txt).
python pipeline/05_extract_features.py                   # -> output/<model_id>/features/<split>_<precision>.npz
python pipeline/06_linear_probe.py --require-classes 19  # -> output/<model_id>/probe_accuracy.json (mAP per precision)

# ONLY when everything above has validated: reclaim the 117 GB archives + calib tree.
python pipeline/prepare_data.py --stage cleanup
```
Share the repo folder (or copy `input/<model_id>/`, `input/bench_sample.npz`
**and** `state/<model_id>/`) to the mini-PC so the benchmark finds the models +
the real sample and `00_verify` can confirm the IR you copied matches the dev
build. Re-run `00_verify` after the benchmark to also close the chain (result
sha256 vs manifest). The accuracy axis runs on the dev box only (accuracy is a
property of the IR, measured once; the targets only measure latency/energy).

### Run the experiment (from the dev box, via `deploy.sh`)
The normal path is `deploy.sh`: it builds the lean image on the dev box, ships
code + image to a target over SSH, runs there, and pulls the results back. The
target needs no build toolchain or package index of its own. See the RUNBOOK for
the full flow; the short version per target:
```bash
# build + ship + readiness gate (00_test + energy check). REBUILD=1 the first time
# or after a base/code change; drop it afterwards.
REBUILD=1 LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 test

# full run, detached in tmux (survives an SSH drop), then bring the results back
LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 run-bg
LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 collect
```
Under the hood `deploy.sh` invokes `host/run_experiment.sh` on the target (00_test
gate -> idle baseline -> per-precision benchmark -> `04_merge_stats`). The Tasmota
IP travels in the repo-root `.env` (`RSEI_TASMOTA_IP`); empty/absent = RAPL only.
`run`/`run-bg` stop early if the target already holds results for this profile, so
a finished measurement is not redone by accident; pass `RERUN=1` to measure again.
`collect` brings back `output/`, `logs/` **and** `state/` (provenance), so each
machine's `results/<LABEL>/` is self-contained for the paper. A fifth mode,
`gpu-probe`, sweeps candidate GPU OpenCL runtimes to find one that compiles the
model on the iGPU and records the result in `state/` (see the RUNBOOK §3.5).
The run id (`<host>_<target>_<stamp>`, e.g. `moort_noavx2_20261010-093000`)
encodes machine + with/without AVX2 + date, and prefixes every output file. To run
the orchestrator directly on a target (debug), see the RUNBOOK's manual path.

### Everything lives in the volume
All artifacts (ONNX, IR models, result JSON, energy CSV, summary) are written
under `output/`, which is the mounted volume in Docker and a plain repo folder
natively - reachable from the host either way. `config.py` auto-detects whether
it runs inside the container (`/work`) or natively (repo-relative `./output`).

### Logs
Every script logs to the console AND to a file under `logs/` (mounted volume,
gitignored). Each log filename **leads with the script name**, then the run id
when there is one, so logs line up with the result files of the same run
(e.g. `logs/03_benchmark__moort_noavx2_<stamp>__fp32.log`,
`logs/00_test_<stamp>.log`). Shared setup is `pipeline/logsetup.py`
(`get_logger`), which also creates `output/` and `logs/` on first run (so a
clean clone just works). Override the location with `RSEI_LOGS_DIR`.

### JSON and integrity
- JSON I/O goes through `pipeline/jsonio.py`: it uses **ujson** when installed
  (faster, as in the paper-1 pipeline) and falls back to stdlib `json`
  transparently. The active backend is logged.
- Artifacts carry **SHA256** checksums for traceability (not cryptographic
  signing, which would be overkill), recorded in a single provenance manifest
  per model, `state/<model_id>/manifest.json`: `01_export` records the hashes of
  `encoder.pt` and the ONNX, `02_quantize` records each IR (`.xml` + `.bin`) per
  precision, and each benchmark result independently re-records the SHA256 of the
  exact IR bytes it measured. This gives an end-to-end chain of custody:
  encoder.pt -> onnx -> IR -> result. The manifest is human-readable JSON and is
  the local source of truth of the data (gitignored, kept across the project).

The **code** is a volume too: `docker-compose.yml` and `run_experiment.sh`
bind-mount `config.py` and `pipeline/` into the container, so edits you make
from outside (VS Code) are live inside with no rebuild.

## Development (VS Code)

- `.vscode/` sets ruff as the formatter (format + fix + organise imports on
  save), recommends the Ruff / Python / Docker extensions, and ships debug
  configs for every pipeline step (`launch.json`).
- `.devcontainer/` builds a dev image (`Dockerfile.dev`, deps from
  `docker/requirements-dev.txt`: torch + timm + openvino + nncf for model prep,
  rasterio + pandas + pyarrow + zstandard for data prep, scikit-learn for the
  accuracy probe, + ruff; plus the apt tool `aria2` for the segmented/resumable
  reBEN download) and mounts the whole repo at `/workspace`, so you edit, run and
  debug every dev-box step (`prepare_data`, `01_export`, `02_quantize`,
  `05_extract_features`, `06_linear_probe`) inside the container. Open the folder
  in VS Code and pick "Reopen in Container". Changing `requirements-dev.txt`
  (e.g. adding scikit-learn) needs a container **rebuild**.
- `pyproject.toml` holds the ruff config. Run it anywhere with:
  ```bash
  ruff format .      # make everything look the same
  ruff check .       # lint
  ```
  The lint ruleset reflects the spirit of NASA/JPL "Power of Ten" and ESA
  ECSS/BSSC (bounded complexity, no dead code, explicit error handling). Those
  standards target C/C++ flight software; there is no official ESA/NASA Python
  standard, so this is the closest faithful mapping, enforced by ruff.

## Nexus pip source (Nexus-first, with fallback)

Pull dependencies from your own Nexus (primary), falling back to public indexes
for anything it does not serve. Copy the template and set your host once:
```bash
cp nexus.env.example nexus.env     # edit <NEXUS_IP>
```
`nexus.env` (gitignored) defines three values, read by **all** build paths:
- `PIP_INDEX_URL` - primary index = your Nexus (serves + caches).
- `PIP_EXTRA_INDEX_URL` - fallback(s), space-separated (keep public PyPI + the
  PyTorch cpu index as a safety net for packages Nexus lacks).
- `PIP_TRUSTED_HOST` - your Nexus host.

Wired into every build:
- **lean image** (N5105/Lenovo): `host/run_experiment.sh` sources `nexus.env`.
- **compose**: `docker compose --env-file nexus.env build`.
- **dev image** (VS Code devcontainer): `devcontainer.json` passes them from the
  host env, so launch VS Code with them loaded:
  ```bash
  set -a; source nexus.env; set +a; code .
  ```
  then "Dev Containers: Rebuild Container".

Empty/unset everywhere -> plain public PyPI (no Nexus needed). Note: this is
"Nexus first, fall back for *missing* packages"; pip does not auto-fall-back if
the Nexus host itself is *unreachable*. To route PyTorch through Nexus too, add
a pytorch proxy repo and point `PIP_INDEX_URL` at a Nexus **group** repo.

### Windows note
Windows is fine as a **dev box** for export/quantise and for CPU smoke-testing
the pipeline. The actual **measurements run on Linux**: RAPL energy
(`/sys/class/powercap`) and the iGPU-via-Docker path are Linux-only, so the
energy and iGPU axes - the core of the paper - are measured on the Debian
mini-PC and Lenovos, not on Windows.

---

## 7. What is wired (and the one input you provide)

Everything is wired from paper 1 and reproduced faithfully from the eval harness:

- **Encoder.** `pipeline/encoder_infer.py` rebuilds the ViT encoder (no masking),
  infers its dimensions from the checkpoint, and loads `encoder.pt` with
  `strict=True`: it refuses (raises) on any missing / extra / shape-mismatched
  weight, so a silent partial load can never inject random weights.
- **INT8 calibration.** `prepare_data.py` builds the reBEN calibration pack
  (`calib_int8.npz`) from the *validation* split, and `02_quantize.py` ->
  `calibration_dataset()` loads it (no random noise), with an explicit
  `subset_size` = the pack size so NNCF uses all of it.
- **Benchmark input.** `03_infer_benchmark.py` -> `load_eval_sample()` loads a
  real preprocessed test patch (`bench_sample.npz`, packed by `prepare_data`,
  numpy-only so the lean target needs no rasterio/torch). If it is missing it
  falls back to synthetic noise and flags `synthetic_input: true` (latency/energy
  still valid; only task accuracy would not be). Note: ViT latency is
  input-independent, so the real sample removes the asterisk without changing the
  timing.
- **Accuracy axis.** `05_extract_features.py` runs each precision's IR as a frozen
  feature extractor over train/test (reBEN subset) and `06_linear_probe.py` trains
  the frozen linear probe (micro/macro-mAP per precision, Δ vs fp32). This mirrors
  the paper-1 harness probe recipe exactly (`pipeline/reben.py` carries the shared
  19-class / transform copy); the encoder is **not** retrained.

The only thing you supply are the two harness products in `input/`: `encoder.pt`
and `norm_stats.json`. Everything else the pipeline builds.

---

## 8. Layout
```
config.py            central configuration (imported by host + container)
docker/
  Dockerfile         lean benchmark image, self-detecting (avx2/noavx2 at build)
  requirements-dev.txt        dev image deps (model prep + data prep)
  requirements-{avx2,noavx2}.txt  lean image deps per CPU profile
  entrypoint.sh      command dispatcher
pipeline/            dev-box steps run in the dev container; lean steps on target
  00_test.py         environment gate (AVX2 / OpenVINO SIGILL / iGPU / RAPL)
  00_verify.py       provenance gate (artifact sha256 vs manifest + chain check)
  prepare_data.py    download reBEN, stream-filter, pack calib + bench sample (dev box)
  reben.py           shared reBEN loader: 19 classes + transform + labels (harness copy)
  encoder_infer.py   ViT-S/16 encoder, no masking (loads encoder.pt)
  01_export_onnx.py  PyTorch -> ONNX  (needs torch+timm, dev box)
  02_quantize.py     fp32 / fp16 / int8  (dev box)
  03_infer_benchmark.py  inference + timing, R repeats (+ metadata snapshot)
  05_extract_features.py features per precision via the IR (accuracy axis, dev box)
  06_linear_probe.py linear probe -> micro/macro-mAP per precision (dev box)
  manifest.py        provenance manifest writer (state/<id>/manifest.json)
  metadata.py        system metadata collector + sha256
  stats.py           mean / std / CI95 across repeats (Student-t, no deps)
  jsonio.py          ujson with json fallback
  logsetup.py        shared logging (console + logs/ file)
host/                runs on the HOST
  measure_power.py   RAPL + Tasmota sampler (host clock)
  run_experiment.sh  orchestrator
  04_merge_stats.py  cross-reference latency + energy -> summary.csv
tools/               diagnostics
  probe_gpu_compile.py   compile the fp32 IR on the iGPU (reproducer)
  Dockerfile.gpu-probe   probe image, parametrised by GPU OpenCL runtime
  gpu_recipes.sh         candidate NEO runtimes to sweep (edit this)
deploy.sh            build on dev, ship, run on target, collect (+ gpu-probe)
input/   encoder + norm_stats + data (calib, bench_sample, reben_subset, features)
state/   provenance manifests (model + data; local source of truth; gitignored)
output/  runs/<run_id>/ per PC-run (bench+power+host_env+summary); <model_id>/ per model (features + probe_accuracy.json)
```

---

## 9. Measurement rigour (protocol)
- batch = 1, warmup discarded, N iterations per repeat.
- **R independent repeats** per (precision × device), each with a fresh
  `compile_model` so the spread reflects real run-to-run variance. The paper
  reports **mean ± CI95** across repeats (Student-t, `stats.py`); per-repeat
  detail is kept in the JSON for auditability. Set with `RSEI_REPEATS`
  (default 5) or `--repeats`.
- energy per inference divides by `n_iter × repeats` over the device window
  (so it includes the small per-repeat recompile overhead - conservative).
- idle baseline measured and subtracted (net power).
- pin the CPU governor. Thermal readings are attempted before/after each device
  run, but the container cannot read `/sys/class/thermal`, so they are `null` in
  the reported runs (stated as a limitation in the paper). The N5105 runs the DDR4-3200 modules
  at 2667 MT/s -> document "DDR4-3200 @ 2667, dual channel".
- `00_test` is the gate: if OpenVINO crashes with SIGILL on the N5105, decide
  there whether to change version or runtime, not mid-experiment.
- Every result embeds a full config + system-metadata snapshot for
  reproducibility.


---

## Citation

If you use this code, please cite the accompanying paper:

> C. Sanchez-Leon and D. Chen, "Quantized Edge Inference of a Self-Supervised
> SAR–Optical Encoder on Commodity x86 Hardware," submitted to *IEEE Geoscience
> and Remote Sensing Letters*, 2026. arXiv ID to be added.

```bibtex
@misc{sanchezleon2026rsedge,
  author = {Sanchez-Leon, Christian and Chen, Dongming},
  title  = {Quantized Edge Inference of a Self-Supervised {SAR}--Optical Encoder
            on Commodity x86 Hardware},
  year   = {2026},
  note   = {Submitted to IEEE Geoscience and Remote Sensing Letters; arXiv ID to be added}
}
```

A `CITATION.cff` is included, so GitHub shows a "Cite this repository" button.

---

## License

Released under the MIT License - see [`LICENSE`](LICENSE). © 2026 Christian Sanchez-Leon.
