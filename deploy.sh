#!/usr/bin/env bash
# =============================================================================
#  deploy.sh  -  build the image here, ship the project to a target node, run
#               it there, and collect the results back.  Runs on the dev box.
# -----------------------------------------------------------------------------
#  The dev box is the single source of truth: it produces the inputs AND the
#  image, so a target node needs no build toolchain or package index of its own.
#  The target's CPU profile is baked into the image at build time, and the image
#  travels as a tarball; the target only loads it and runs.
#
#  Modes (per target):
#    test     build + ship, then run the 00_test environment gate (live output)
#    run      build + ship, then run the full experiment in the FOREGROUND,
#             streaming live (you watch it; Ctrl-C stops it). Short runs / debug.
#    run-bg   same, but inside a tmux session on the target, so it survives an
#             SSH disconnect. Long unattended runs (reattach later).
#    collect  pull the target's output/, logs/ AND state/ into results/<label>/
#    gpu-probe  sweep candidate GPU OpenCL runtimes (NEO): build each on the dev
#             box, run it on the target iGPU, and record which one compiles the
#             model. Writes state/gpu_runtime_probe.json (the paper's datum for
#             which GPU runtime is used and why). Edit tools/gpu_recipes.sh.
#
#  Usage:
#    ./deploy.sh <user@host> <noavx2|avx2> [test|run|run-bg|collect|gpu-probe]
#
#  Env:
#    REMOTE_DIR   remote project path      (default: experiment/rs-edge-infer)
#    RESULTS_DIR  local collect root        (default: output)
#    LABEL        machine label subfolder   (default: the profile name)
#    TASMOTA_IP   wall-plug IP for a run    (default: taken from config.py)
#    REBUILD=1    rebuild the image even if it already exists locally
#    RERUN=1      re-run even if the target already holds results for this profile
#                 (without it, run/run-bg stop early so a measurement is not redone)
#    IR_DIR       model dir (fp32/fp16/int8) for gpu-probe (default: input/encoder)
#    OPENVINO_VER openvino version in the probe image (default: 2025.3.0)
#    GPU_PROBE_FORCE=1  re-run every gpu-probe candidate (default: skip ones
#                 already in state/gpu_runtime_probe.json, run only new recipes)
#
#  Example (one node, end to end):
#    LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 test      # check it runs
#    LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 run       # watch it live
#    LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 run-bg    # leave it in tmux
#    LABEL=n5105 ./deploy.sh user@<target-ip> noavx2 collect   # bring results back
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"      # place this script at the repo root

DEST="${1:?usage: deploy.sh <user@host> <noavx2|avx2> [test|run|run-bg|collect]}"
PROFILE="${2:?profile required: noavx2 | avx2}"
MODE="${3:-test}"
REMOTE_DIR="${REMOTE_DIR:-experiment/rs-edge-infer}"
RESULTS_DIR="${RESULTS_DIR:-output}"   # ONE tree: output/<label>/ (cost, by machine)
LABEL="${LABEL:-$PROFILE}"
IMAGE="rs-edge-infer:${PROFILE}"
TARBALL="${TMPDIR:-/tmp}/rsei-${PROFILE}.tar.zst"   # local build artifact, kept OUT of the repo
RTAR="rsei-${PROFILE}.tar.zst"                       # basename used on the target
IR_DIR="${IR_DIR:-input/encoder}"                    # model dir (fp32/fp16/int8) for gpu-probe
OPENVINO_VER="${OPENVINO_VER:-2025.3.0}"             # openvino version in the probe image

case "$PROFILE" in noavx2|avx2) ;; *) echo "profile must be noavx2 or avx2"; exit 2 ;; esac
case "$MODE" in
  test|run|run-bg|collect|gpu-probe) ;;
  *) echo "mode must be test, run, run-bg, collect or gpu-probe"; exit 2 ;;
esac

# --- collect: pull results back, then stop -----------------------------------
# output/ = benchmark JSON + energy CSV + summary; logs/ = per-run logs; state/ =
# provenance (sha256 manifests + the gpu_runtime_probe record). Bringing all three
# back leaves results/<label>/ self-contained for the paper.
if [ "$MODE" = "collect" ]; then
    # ONE output tree. The target's benchmarks (cost axis) land under output/<label>/,
    # labelled by machine - next to the per-model accuracy under output/<model_id>/.
    # No separate results/ tree.
    DEST_DIR="$RESULTS_DIR/$LABEL"
    mkdir -p "$DEST_DIR"
    echo "==> collecting runs, logs and state -> $DEST_DIR/"
    # --omit-dir-times: the run folders on the target are root-owned (the measurement
    # runs under sudo for RAPL), so setting directory mtimes on a local copy owned by
    # another user - or on a drvfs mount like WSL /mnt/c - fails with "Operation not
    # permitted" and aborts the sync. Skipping dir times side-steps it; file contents,
    # times and the summary still transfer.
    RSYNC_OPTS=(-az --omit-dir-times --info=progress2)
    rsync "${RSYNC_OPTS[@]}" "$DEST:$REMOTE_DIR/output/runs/" "$DEST_DIR/runs/"
    rsync "${RSYNC_OPTS[@]}" "$DEST:$REMOTE_DIR/logs/"        "$DEST_DIR/logs/"
    rsync "${RSYNC_OPTS[@]}" "$DEST:$REMOTE_DIR/state/"       "$DEST_DIR/state/" || true
    echo "==> done. Collected in $DEST_DIR/ (runs, logs, state)"
    exit 0
fi

# --- gpu-probe: find a GPU OpenCL runtime (NEO) that compiles the model -------
# Build each candidate probe image HERE (dev = source of truth), ship it, run it
# on the target iGPU with the fp32 IR, and record which one compiles. The record
# is written to the target's state/ (provenance) so `collect` brings it with the
# rest, and a copy is kept locally in results/<label>/.
if [ "$MODE" = "gpu-probe" ]; then
    [ -f "$IR_DIR/fp32/model.xml" ] || { echo "!! no fp32 IR at $IR_DIR/fp32/model.xml (build it first, or set IR_DIR)"; exit 1; }
    [ -f tools/gpu_recipes.sh ] || { echo "!! tools/gpu_recipes.sh missing"; exit 1; }
    . tools/gpu_recipes.sh
    if [ -f nexus.env ]; then set -a; . ./nexus.env; set +a; fi

    echo "==> checking SSH + docker on $DEST"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$DEST" true \
        || { echo "!! cannot reach $DEST over SSH (is key auth set up?)"; exit 1; }
    ssh "$DEST" 'docker info >/dev/null 2>&1' \
        || { echo "!! docker not usable by this user on $DEST (socket denied?)."; exit 1; }

    ssh "$DEST" "mkdir -p '$REMOTE_DIR/gpu_probe/ir' '$REMOTE_DIR/state'"
    echo "==> [dev->target] shipping IR (fp32/fp16/int8) for the probe"
    rsync -az --exclude 'onnx/' "$IR_DIR/" "$DEST:$REMOTE_DIR/gpu_probe/ir/"

    mkdir -p "$RESULTS_DIR/$LABEL"
    PTSV="$(mktemp)"; trap 'rm -f "$PTSV"' EXIT

    # Cumulative record: results accumulate across runs, so a candidate already in
    # the record is skipped (only NEW recipes are built/run). GPU_PROBE_FORCE=1
    # re-runs everything in the order regardless.
    REC_LOCAL="$RESULTS_DIR/$LABEL/gpu_runtime_probe.json"
    DONE=""
    if [ -z "${GPU_PROBE_FORCE:-}" ] && [ -f "$REC_LOCAL" ]; then
        DONE="$(python3 -c "import json,sys;print(' '.join(c.get('recipe','') for c in json.load(open(sys.argv[1])).get('candidates',[])))" "$REC_LOCAL" 2>/dev/null)" || DONE=""
    fi

    # sha256 of each IR under test (fp32/fp16/int8): same chain of custody as the
    # manifests, so the probe result is tied to the precise model bytes compiled
    # (cross-checkable against state/<model>/manifest.json).
    ir_sha() { [ -f "$1" ] && sha256sum "$1" | cut -d' ' -f1 || echo ''; }
    IR_SHA_FP32="$(ir_sha "$IR_DIR/fp32/model.xml")"
    IR_SHA_FP16="$(ir_sha "$IR_DIR/fp16/model.xml")"
    IR_SHA_INT8="$(ir_sha "$IR_DIR/int8/model.xml")"

    echo "== GPU OpenCL runtime probe: each candidate is BUILT ON THIS DEV BOX,"
    echo "   then run on the target iGPU ($DEST). Already-recorded candidates are"
    echo "   skipped; build output is silent (tail the per-recipe log, path below)."
    printf '%-20s %-9s %s\n' RECIPE RESULT "PRECISIONS | DRIVER"
    for name in "${GPU_RECIPE_ORDER[@]}"; do
        case " $DONE " in
          *" $name "*) echo "==> [cached] '$name' already in the record, skipping (GPU_PROBE_FORCE=1 to redo)"; continue ;;
        esac
        img="rsei-gpuprobe:${name}"
        ptar="${TMPDIR:-/tmp}/rsei-gpuprobe-${name}.tar.zst"
        blog="$RESULTS_DIR/$LABEL/gpuprobe_${name}_build.log"
        ovver="${GPU_RECIPE_OV[$name]:-$OPENVINO_VER}"
        echo "==> [dev] building '$name' (openvino $ovver; silent; watch: tail -f $blog)"
        if ! docker build -f tools/Dockerfile.gpu-probe \
                --build-arg OPENVINO_VER="$ovver" \
                --build-arg NEO_INSTALL="${GPU_RECIPES[$name]}" \
                ${PIP_INDEX_URL:+--build-arg PIP_INDEX_URL="$PIP_INDEX_URL"} \
                ${PIP_EXTRA_INDEX_URL:+--build-arg PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL"} \
                ${PIP_TRUSTED_HOST:+--build-arg PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST"} \
                -t "$img" . >"$blog" 2>&1; then
            printf '%-20s %-9s %s\n' "$name" BUILD_ERR "see $blog"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" BUILD_ERR "$ovver" "" "" "" "build failed (dev)" >>"$PTSV"
            continue
        fi
        echo "==> [dev->target] shipping + loading probe image '$name'"
        docker save "$img" | zstd -T0 -f -o "$ptar"
        rsync -az "$ptar" "$DEST:$REMOTE_DIR/gpu_probe/probe.tar.zst"
        ssh "$DEST" "cd '$REMOTE_DIR' && zstd -d -c gpu_probe/probe.tar.zst | docker load && rm -f gpu_probe/probe.tar.zst" >/dev/null 2>&1 || true
        rlog="$RESULTS_DIR/$LABEL/gpuprobe_${name}.log"
        echo "==> [target] compiling the fp32 IR on the iGPU with '$name'"
        # docker -v needs an absolute host path: cd into REMOTE_DIR and use $(pwd).
        ssh "$DEST" "cd '$REMOTE_DIR' && docker run --rm --device /dev/dri -v \"\$(pwd)/gpu_probe/ir:/ir:ro\" '$img'" >"$rlog" 2>&1 || true
        res="$(sed -n 's/^RESULT: //p' "$rlog" | head -1)"
        drv="$(sed -n 's/^driver: //p' "$rlog" | head -1)"
        oclc="$(sed -n 's/^oclc: //p' "$rlog" | head -1)"
        prc="$(sed -n 's/^precisions: //p' "$rlog" | head -1)"
        ocl="$(sed -n 's/^opencl: //p' "$rlog" | head -1)"
        printf '%-20s %-9s %s\n' "$name" "${res%% *}" "${prc:-} | ${drv:-}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${res:-?}" "$ovver" "${drv:-}" "${oclc:-}" "${prc:-}" "${ocl:-}" >>"$PTSV"
        rm -f "$ptar"
    done

    REC_FILE="docker/requirements-gpu.recommended.txt"
    python3 - "$PTSV" "$REC_LOCAL" "$OPENVINO_VER" "$DEST" "$PROFILE" "$LABEL" "$IR_DIR" "$IR_SHA_FP32" "$IR_SHA_FP16" "$IR_SHA_INT8" "$REC_FILE" <<'PY'
import json, os, sys
from datetime import datetime, timezone  # runs on the dev box: keep portable

tsv, rec, ov_ver, dest, profile, label, ir_dir, sha32, sha16, sha8, rec_file = sys.argv[1:12]
now = datetime.now(timezone.utc).isoformat()

# Merge: keep every candidate already in the record, add/overwrite with this run's.
by_name = {}
if os.path.exists(rec):
    try:
        for c in json.load(open(rec)).get("candidates", []):
            if c.get("recipe"):
                by_name[c["recipe"]] = c
    except (OSError, ValueError):
        pass
with open(tsv) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) == 7:
            by_name[p[0]] = {"recipe": p[0], "result": p[1], "openvino": p[2],
                             "driver": p[3], "oclc": p[4], "precisions": p[5], "opencl": p[6]}
rows = list(by_name.values())
first = next((r for r in rows if str(r.get("result", "")).startswith("PASS")), None)
doc = {
    "kind": "gpu_runtime_probe",
    "generated_by": "deploy.sh gpu-probe",
    "updated_utc": now,
    "label": label,
    "target": dest,
    "profile": profile,
    "openvino": ov_ver,
    "ir": {"model_dir": ir_dir, "sha256": {"fp32": sha32, "fp16": sha16, "int8": sha8}},
    "candidates": rows,
    "recommended_pin": first,
}
with open(rec, "w") as f:
    json.dump(doc, f, indent=2)

# Advisory recommendation file next to the requirements. NOT the authoritative pin:
# the human sets the pin deliberately in requirements-{noavx2,avx2}.txt (and the
# Dockerfile if a non-stock NEO is needed). This just records what the probe found.
adv = [
    "# =============================================================================",
    "#  requirements-gpu.recommended.txt  -  ADVISORY (generated by deploy.sh gpu-probe)",
    "#  NOT the authoritative pin. Set the pin yourself in requirements-{noavx2,avx2}.txt",
    "#  (deliberately, with a commit). This file only records what the probe found,",
    "#  so the recommended library/version is documented next to the deps.",
    "#  Recommended by: deploy.sh gpu-probe (automated sweep).",
    f"#  Run label: {label}   target: {dest}   profile: {profile}   updated: {now}",
    f"#  Full sweep + sha256: state/gpu_runtime_probe.json  (fp32 IR sha256={sha32})",
    "# =============================================================================",
]
if first:
    adv.append(f"# iGPU PASS with recipe '{first['recipe']}' "
               f"(driver {first.get('driver', '?')}, {first.get('oclc', '?')}; "
               f"precisions {first.get('precisions', '?')}).")
    ov = first.get("openvino", "")
    adv.append("# Recommended OpenVINO pin for the lean image:")
    adv.append(f"openvino=={ov}" if ov else "# (openvino version unknown for the winner)")
    if not first["recipe"].startswith("stock"):
        adv.append(f"# This winner uses a NON-STOCK GPU runtime ('{first['recipe']}'):")
        adv.append("# also transplant its recipe from tools/gpu_recipes.sh into docker/Dockerfile.")
    else:
        adv.append("# GPU runtime: stock intel-opencl-icd (no Dockerfile change needed).")
else:
    adv.append("# No candidate compiled the iGPU kernel on this target.")
    adv.append("# iGPU = documented limitation; keep the paper on CPU.")
with open(rec_file, "w") as f:
    f.write("\n".join(adv) + "\n")

print()
print(f"candidates in record: {len(rows)}")
if first:
    ov = first.get("openvino", "?")
    print(f"PIN (recommended): openvino=={ov}  (recipe '{first['recipe']}', "
          f"precisions {first.get('precisions', '?')})")
    print(f"  -> set it in docker/requirements-{{noavx2,avx2}}.txt. Advisory written to {rec_file}")
else:
    print("No candidate compiled on the GPU. Add recipes in tools/gpu_recipes.sh,")
    print(f"or document the iGPU as a limitation (advisory written to {rec_file}).")
print(f"Record: {rec}")
PY

    echo "==> pushing record to target state/ (so collect brings it)"
    rsync -az "$REC_LOCAL" "$DEST:$REMOTE_DIR/state/gpu_runtime_probe.json" || true
    echo "==> done. Bring everything back:  LABEL=$LABEL ./deploy.sh $DEST $PROFILE collect"
    exit 0
fi

# --- guard: don't redo a finished measurement (run/run-bg only) --------------
# A completed run leaves per-precision bench JSON under output/runs/<run>/, the
# run folder named with the profile. If that is already there and RERUN is unset,
# stop BEFORE building or shipping anything, so a measurement is never redone by
# accident. 'test' writes no benchmark results (exempt); 'collect' only pulls.
if { [ "$MODE" = "run" ] || [ "$MODE" = "run-bg" ]; } && [ -z "${RERUN:-}" ]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=8 "$DEST" \
         "ls '$REMOTE_DIR'/output/runs/*_${PROFILE}_*/bench_*.json >/dev/null 2>&1"; then
        echo "==> $DEST already holds results for profile '$PROFILE' (output/runs/)."
        echo "    Nothing was rebuilt or shipped. Choose one:"
        echo "      collect them:   LABEL=$LABEL ./deploy.sh $DEST $PROFILE collect"
        echo "      force a re-run:  RERUN=1 LABEL=$LABEL ./deploy.sh $DEST $PROFILE $MODE"
        exit 0
    fi
fi

EXCLUDES=(
    --exclude '.git/' --exclude '__pycache__/'
    --exclude 'output/' --exclude 'logs/'
    --exclude '*.tar.zst' --exclude 'reben_subset/' --exclude 'features/'
    --exclude 'input/*/onnx/' --exclude 'input/encoder.pt' --exclude '*calib_int8.npz'
)

# --- 1. Build the image for this profile + save a tarball --------------------
if [ -n "${REBUILD:-}" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    # Optional private package index (e.g. a local Nexus) via nexus.env, so the
    # build pulls Python wheels from there instead of the public index.
    if [ -f nexus.env ]; then
        set -a; . ./nexus.env; set +a
        echo "==> nexus.env loaded (pip index: ${PIP_INDEX_URL:-unset})"
    fi
    echo "==> building $IMAGE"
    docker build -f docker/Dockerfile --build-arg FORCE_TARGET="$PROFILE" \
        ${PIP_INDEX_URL:+--build-arg PIP_INDEX_URL="$PIP_INDEX_URL"} \
        ${PIP_EXTRA_INDEX_URL:+--build-arg PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL"} \
        ${PIP_TRUSTED_HOST:+--build-arg PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST"} \
        -t "$IMAGE" .
else
    echo "==> reusing local $IMAGE  (REBUILD=1 to force a rebuild)"
fi
echo "==> saving $IMAGE -> $TARBALL"
docker save "$IMAGE" | zstd -T0 -f -o "$TARBALL"

# --- 2. Preflight: SSH, required tools, working directory --------------------
# The target needs docker (from bring-up) plus rsync + zstd + tmux for this deploy.
# docker is a bring-up prerequisite and is NOT auto-installed here; rsync, zstd and
# tmux are installed on demand so the deploy is self-sufficient (never assume the
# target has them). tmux is always installed even though only run-bg uses it, so the
# unattended mode is always available without a second trip. Versions don't matter
# for these: presence does; the result-critical versions are pinned in the image.
echo "==> checking SSH to $DEST"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$DEST" true \
    || { echo "!! cannot reach $DEST over SSH (is key auth set up?)"; exit 1; }

missing=$(ssh "$DEST" 'for c in docker rsync zstd tmux; do command -v "$c" >/dev/null 2>&1 || echo "$c"; done') || true
if echo "$missing" | grep -qx docker; then
    echo "!! docker is not installed on $DEST. Complete the target bring-up first"
    echo "   (install docker and add the user to the docker group), then re-run."
    exit 1
fi
tools=$(echo "$missing" | grep -xE 'rsync|zstd|tmux' | tr '\n' ' ') || true
if [ -n "${tools// }" ]; then
    echo "==> installing missing tools on target: $tools"
    ssh -t "$DEST" "sudo apt-get update && sudo apt-get install -y $tools" \
        || { echo "!! could not install $tools on $DEST"; exit 1; }
fi

# docker must be USABLE by this user, not just installed: a denied socket means
# the user is not in the docker group (a bring-up step, not auto-fixed here).
if ! ssh "$DEST" 'docker info >/dev/null 2>&1'; then
    echo "!! docker is installed on $DEST but not usable by this user (socket denied)."
    echo "   Add the user to the docker group, then reconnect once:"
    echo "     ssh $DEST 'sudo usermod -aG docker \$USER'"
    exit 1
fi

echo "==> ensuring remote dir $REMOTE_DIR"
ssh "$DEST" "mkdir -p '$REMOTE_DIR'"

# --- 3. Ship code + inputs, then the image -----------------------------------
echo "==> rsync project -> $DEST:$REMOTE_DIR"
rsync -az --info=progress2 "${EXCLUDES[@]}" ./ "$DEST:$REMOTE_DIR/"
echo "==> rsync image -> $DEST:$REMOTE_DIR/$RTAR"
rsync -az --info=progress2 "$TARBALL" "$DEST:$REMOTE_DIR/$RTAR"

# --- 4. Load the image on the target (then remove the shipped tarball) --------
echo "==> loading image on target"
ssh "$DEST" "cd '$REMOTE_DIR' && zstd -d -c '$RTAR' | docker load && rm -f '$RTAR'"

# --- 5. Run ------------------------------------------------------------------
RUN_ENV="RSEI_IMAGE='$IMAGE' SKIP_BUILD=1 ${TASMOTA_IP:+TASMOTA_IP='$TASMOTA_IP'}"
case "$MODE" in
  test)
    echo "==> readiness gate on target (00_test + energy sources; live)"
    echo "    (enter your sudo password when prompted - RAPL needs root to verify)"
    ssh -t "$DEST" "cd '$REMOTE_DIR' && $RUN_ENV RSEI_TEST_ONLY=1 sudo -E bash host/run_experiment.sh"
    ;;
  run)
    echo "==> full experiment on target (foreground, live - Ctrl-C stops it)"
    echo "    (enter your sudo password when prompted)"
    ssh -t "$DEST" "cd '$REMOTE_DIR' && $RUN_ENV sudo -E bash host/run_experiment.sh"
    echo "==> bring the results back:  LABEL=$LABEL ./deploy.sh $DEST $PROFILE collect"
    ;;
  run-bg)
    echo "==> full experiment in tmux session 'rsei' on target"
    echo "    (enter your sudo password when prompted; detach with Ctrl-b then d - it keeps running)"
    ssh -t "$DEST" "cd '$REMOTE_DIR' && tmux new-session -s rsei \"$RUN_ENV sudo -E bash host/run_experiment.sh\""
    echo "==> reattach:  ssh -t $DEST 'tmux attach -t rsei'"
    echo "==> when done, collect:  LABEL=$LABEL ./deploy.sh $DEST $PROFILE collect"
    ;;
esac