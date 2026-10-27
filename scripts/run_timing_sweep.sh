#!/usr/bin/env bash
# run_timing_sweep.sh
#
# Sweeps over (sample_rate, num_codes) operating points, running the
# TIMING_BENCHMARK binary at each point and collecting the timing CSV.
#
# Usage:
#   ./scripts/run_timing_sweep.sh [--blocks N] [--outdir DIR]
#
# Options:
#   --blocks N    Override BENCHMARK_BLOCKS (default: 1000).
#   --outdir DIR  Where to write per-run CSVs and the master CSV
#                 (default: timing_sweep/).
#
# The benchmark binary (build/correlator_bench) must be built first:
#   make benchmark                         # default 1000 blocks
#   make benchmark BENCHMARK_BLOCKS=500    # or a shorter run
# If the binary is not found this script will attempt to build it.
#
# Output:
#   <outdir>/timing_<Msps>Msps_<numCodes>codes.csv  — per-run CSV
#   <outdir>/timing_master.csv                       — all runs concatenated

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_BIN="${REPO_ROOT}/build/correlator_bench"

# ---------- defaults ---------------------------------------------------------
BLOCKS=1000
OUTDIR="${REPO_ROOT}/timing_sweep"

# ---------- argument parsing -------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --blocks)   BLOCKS="$2";  shift 2 ;;
        --outdir)   OUTDIR="$2";  shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------- build if needed --------------------------------------------------
if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "[sweep] benchmark binary not found — building with BENCHMARK_BLOCKS=${BLOCKS} ..."
    make -C "${REPO_ROOT}" benchmark BENCHMARK_BLOCKS="${BLOCKS}"
fi

mkdir -p "${OUTDIR}"

# ---------- sweep parameters -------------------------------------------------
# Sample rates in Hz.  The corresponding M (template length) is derived at
# runtime by the binary as round(24552 * fs / 6e6).
SAMPLE_RATES=(4000000 6000000 8000000 10000000 12000000)

# Number of codes to correlate against.
NUM_CODES=(1 5 10 15 20 25 30 35)

# ---------- master CSV header ------------------------------------------------
MASTER="${OUTDIR}/timing_master.csv"
HEADER_WRITTEN=0

TOTAL=$(( ${#SAMPLE_RATES[@]} * ${#NUM_CODES[@]} ))
RUN=0

echo "[sweep] ${TOTAL} operating points  (BENCHMARK_BLOCKS=${BLOCKS})"
echo "[sweep] output directory: ${OUTDIR}"
echo ""

for SR in "${SAMPLE_RATES[@]}"; do
    for NC in "${NUM_CODES[@]}"; do
        RUN=$(( RUN + 1 ))
        SR_MSPS=$(( SR / 1000000 ))
        TAG="${SR_MSPS}Msps_${NC}codes"
        CSV_OUT="${OUTDIR}/timing_${TAG}.csv"

        printf "[%2d/%d] %s Msps  %2d codes  -> %s\n" \
               "${RUN}" "${TOTAL}" "${SR_MSPS}" "${NC}" "$(basename "${CSV_OUT}")"

        # Run the benchmark from OUTDIR so timing.csv lands there.
        pushd "${OUTDIR}" > /dev/null
        "${BENCH_BIN}" "${SR}" "${NC}" 2>&1 | tail -20
        popd > /dev/null

        # Rename the per-run CSV.
        if [[ -f "${OUTDIR}/timing.csv" ]]; then
            mv "${OUTDIR}/timing.csv" "${CSV_OUT}"
        else
            echo "  WARNING: timing.csv not found after run — skipping" >&2
            continue
        fi

        # Append to master CSV (header once, data rows thereafter).
        if [[ "${HEADER_WRITTEN}" -eq 0 ]]; then
            head -1 "${CSV_OUT}" > "${MASTER}"
            HEADER_WRITTEN=1
        fi
        # Skip the header line of each per-run file.
        tail -n +2 "${CSV_OUT}" >> "${MASTER}"

        echo ""
    done
done

echo "[sweep] Done.  Master CSV: ${MASTER}"
echo "[sweep] Per-run CSVs in:  ${OUTDIR}/"
