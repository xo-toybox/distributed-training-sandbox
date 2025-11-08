#!/usr/bin/env bash
set -euo pipefail

# Utility entrypoint for launching DDP training on Modal, downloading the
# resulting profiler traces, and viewing them locally. Each command writes to or
# reads from these locations:
#   ./path/to/profile.sh run          pushes traces into ${MODAL_TRACE_VOLUME}/<run_id> on Modal.
#   ./path/to/profile.sh sync         pulls the entire volume (or run) into ${TRACE_OUTPUT_DIR}.
#   ./path/to/profile.sh view         points TensorBoard at the downloaded ${TRACE_OUTPUT_DIR}.
#   ./path/to/profile.sh all          run → sync → view using the paths above.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DDP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${DDP_ROOT}/.env.public"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

die() {
    echo "${SCRIPT_NAME}: $*" >&2
    exit 1
}

if [[ ! -f "${ENV_FILE}" ]]; then
    die "Missing config: ${ENV_FILE}."
fi

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

required_vars=(MODAL_TRACE_VOLUME MODAL_GPU_SPEC TRACE_OUTPUT_DIR TENSORBOARD_PORT)
missing=()
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        missing+=("${var}")
    fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
    die "Missing required variable(s) in ${ENV_FILE}: ${missing[*]}"
fi

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        die "Missing required command '${cmd}'. Please install it and ensure it is in PATH."
    fi
}

resolve_path() {
    local path="$1"
    case "${path}" in
        "~"|"~/"*)
            path="${HOME}${path:1}"
            ;;
    esac
    if [[ "${path}" = /* ]]; then
        printf "%s\n" "${path}"
    else
        printf "%s\n" "${DDP_ROOT}/${path#./}"
    fi
}

TRACE_DIR="$(resolve_path "${TRACE_OUTPUT_DIR}")"
VOLUME_NAME="${MODAL_TRACE_VOLUME}"
PORT="${TENSORBOARD_PORT}"

usage() {
    cat <<USAGE
Usage: ${SCRIPT_NAME} [command] [options]

Commands:
  run   [--run-name NAME]                           Launch training on Modal.
  sync  [destination_dir]                           Pull trace volume to a local dir.
  view  [--port PORT] [--logdir DIR]                Start TensorBoard for downloaded traces.
  all   (default) combines run -> sync -> view      End-to-end helper.

Options (global):
  --run-name NAME         Optional identifier appended to the Modal run_id.
  --port PORT             Port for TensorBoard (view/all commands).
  --logdir DIR            TensorBoard logdir (view/all commands).
  --dest DIR              Destination for downloaded traces (sync/all commands).
  -h, --help              Show this help and exit.
USAGE
}

run_modal_job() {
    local run_name="$1"
    local modal_app="${DDP_ROOT}/modal_app.py"
    local cmd=(modal run "${modal_app}::launch")
    if [[ -n "${run_name}" ]]; then
        cmd+=("--run-name" "${run_name}")
    fi
    echo "▶️  Launching Modal job: ${cmd[*]}"
    "${cmd[@]}"
}

sync_traces() {
    local dest="$1"
    mkdir -p "${dest}"
    echo "⬇️  Syncing volume '${VOLUME_NAME}' into ${dest}"
    modal volume get "${VOLUME_NAME}" / "${dest}" --force
}

launch_tensorboard() {
    local logdir="$1"
    local port="$2"
    echo "🚀 Starting TensorBoard on port ${port}, logdir=${logdir}"
    tensorboard --logdir "${logdir}" --port "${port}"
}

COMMAND="${1:-all}"
if [[ $# -gt 0 ]]; then
    case "${COMMAND}" in
        run|sync|view|all)
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            die "Unknown option: ${COMMAND}"
            ;;
        *)
            die "Unknown command: ${COMMAND}"
            ;;
    esac
fi

RUN_NAME="${RUN_NAME:-}"
VIEW_PORT="${PORT}"
VIEW_LOGDIR="${TRACE_DIR}"
SYNC_DEST="${TRACE_DIR}"
VIEW_LOGDIR_SET=0
SYNC_DEST_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)
            RUN_NAME="$2"; shift 2;;
        --port)
            VIEW_PORT="$2"; shift 2;;
        --logdir)
            VIEW_LOGDIR="$(resolve_path "$2")"; VIEW_LOGDIR_SET=1; shift 2;;
        --dest|--destination)
            SYNC_DEST="$(resolve_path "$2")"; SYNC_DEST_SET=1; shift 2;;
        --help|-h)
            usage; exit 0;;
        --)
            shift; break;;
        -*)
            die "Unknown option: $1";;
        *)
            if [[ "${COMMAND}" == "sync" && "${SYNC_DEST_SET}" -eq 0 ]]; then
                SYNC_DEST="$(resolve_path "$1")"; SYNC_DEST_SET=1; shift 1;
            else
                die "Unexpected argument: $1"
            fi
            ;;
    esac
done

if [[ "${COMMAND}" == "all" && "${VIEW_LOGDIR_SET}" -eq 0 ]]; then
    VIEW_LOGDIR="${SYNC_DEST}"
fi

case "${COMMAND}" in
    run|sync|all)
        require_command modal
        ;;
esac

case "${COMMAND}" in
    view|all)
        require_command tensorboard
        ;;
esac

case "${COMMAND}" in
    run)
        run_modal_job "${RUN_NAME}"
        ;;
    sync)
        sync_traces "${SYNC_DEST}"
        ;;
    view)
        launch_tensorboard "${VIEW_LOGDIR}" "${VIEW_PORT}"
        ;;
    all)
        echo "Using trace directory: ${SYNC_DEST}"
        run_modal_job "${RUN_NAME}"
        sync_traces "${SYNC_DEST}"
        launch_tensorboard "${VIEW_LOGDIR}" "${VIEW_PORT}"
        ;;
    *)
        usage
        exit 1
        ;;
esac
