#!/usr/bin/env bash
# run_fft_tier_sweep.sh
#
# Sweeps sample rate from 2.5 MHz to 10 MHz in 500 kHz steps with
# a fixed num_codes=10, to expose the power-of-two jumps in fftSize (N).
# Produces one CSV per operating point and a master CSV.
#
# Usage:
#   bash scripts/run_fft_tier_sweep.sh [--blocks N] [--codes N] [--outdir DIR]
#
# The benchmark binary must be built first:  make benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_BIN="${REPO_ROOT}/build/correlator_bench"

# ---------- defaults ---------------------------------------------------------
BLOCKS=1000
NUM_CODES=10
OUTDIR="${REPO_ROOT}/timing_fft_tier_sweep"

# ---------- argument parsing -------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --blocks) BLOCKS="$2";    shift 2 ;;
        --codes)  NUM_CODES="$2"; shift 2 ;;
        --outdir) OUTDIR="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------- build if needed --------------------------------------------------
if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "[sweep] benchmark binary not found — building with BENCHMARK_BLOCKS=${BLOCKS} ..."
    make -C "${REPO_ROOT}" benchmark BENCHMARK_BLOCKS="${BLOCKS}"
fi

mkdir -p "${OUTDIR}"

# ---------- generate sample rate list (2.5 to 10.0 MHz, step 0.5 MHz) -------
SAMPLE_RATES=()
for step in $(seq 0 15); do
    SR=$(( 2500000 + step * 500000 ))
    SAMPLE_RATES+=("${SR}")
done

# ---------- master CSV -------------------------------------------------------
MASTER="${OUTDIR}/timing_fft_tier_master.csv"
HEADER_WRITTEN=0

TOTAL=${#SAMPLE_RATES[@]}
RUN=0

echo "[sweep] ${TOTAL} sample rates  |  ${NUM_CODES} codes fixed  |  ${BLOCKS} blocks each"
echo "[sweep] output: ${OUTDIR}"
echo ""

for SR in "${SAMPLE_RATES[@]}"; do
    RUN=$(( RUN + 1 ))
    SR_MHZ=$(echo "scale=1; ${SR}/1000000" | bc)
    TAG="${SR_MHZ}MHz_${NUM_CODES}codes"
    CSV_OUT="${OUTDIR}/timing_${TAG}.csv"

    printf "[%2d/%d]  %5s MHz  %d codes  ->  %s\n" \
           "${RUN}" "${TOTAL}" "${SR_MHZ}" "${NUM_CODES}" "$(basename "${CSV_OUT}")"

    pushd "${OUTDIR}" > /dev/null
    "${BENCH_BIN}" "${SR}" "${NUM_CODES}" 2>&1 | tail -12
    popd > /dev/null

    if [[ -f "${OUTDIR}/timing.csv" ]]; then
        mv "${OUTDIR}/timing.csv" "${CSV_OUT}"
    else
        echo "  WARNING: timing.csv missing — skipping" >&2
        continue
    fi

    if [[ "${HEADER_WRITTEN}" -eq 0 ]]; then
        head -1 "${CSV_OUT}" > "${MASTER}"
        HEADER_WRITTEN=1
    fi
    tail -n +2 "${CSV_OUT}" >> "${MASTER}"

    echo ""
done

echo "[sweep] Done.  Master CSV: ${MASTER}"