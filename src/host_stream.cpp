// host_stream.cpp
//
// Streams complex samples from a Nuand bladeRF A4 (bladeRF 2.0 micro xA4)
// over USB 3.0, via libbladeRF's synchronous interface, and feeds them
// into the GPU matched-filter bank (correlator.hpp) using overlap-save.
//
// libbladeRF's native wire format is SC16_Q11: each I and Q sample is a
// 16-bit signed integer representing a Q11 fixed-point value (12-bit ADC
// headroom-scaled into 16 bits). Unlike UHD, there is no host-side "give
// me float32" stream format — conversion to float is done here on the
// CPU after each receive call, scaling by 1/2048 (2^11).
//
// Orin Nano has unified physical memory, but CPU writes to mapped pinned
// memory can sit in the CPU L1/L2 cache and not be visible to the GPU via
// a DeviceToDevice copy. The ring buffer is therefore allocated as plain
// pinned memory (cudaHostAllocDefault) and processBlock() uses an explicit
// cudaMemcpyHostToDevice, which forces a cache-coherent DMA transfer.
// The SC16Q11->float32 conversion is an unavoidable small CPU-side pass
// (bladeRF doesn't hand you floats directly the way UHD's "fc32" stream
// args do), but it's a couple thousand samples per block — negligible.
//
// Build: see ../Makefile. Requires libbladeRF (libbladerf-dev) and the
// CUDA toolkit.

#include <libbladeRF.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <memory>
#include <chrono>

#include "../include/correlator.hpp"

// =============================================================================
// Compile-time capture switch
//
//   Defined   → streams to raw_capture.bin + corr_c##_capture.bin for
//               CAPTURE_DURATION_S seconds, then exits.  Useful for offline
//               verification of the GPU correlator against MATLAB / Python.
//
//   Undefined → runs continuously until SIGINT (Ctrl-C), writing nothing to
//               disk.  This is the normal real-time operating mode.
//
// Toggle by adding/removing -DWRITE_CAPTURE_FILES to CXXFLAGS in the Makefile,
// or by un/commenting the line below.
// =============================================================================
// #define WRITE_CAPTURE_FILES

static volatile sig_atomic_t g_stop = 0;
static void handleSigint(int) { g_stop = 1; }
// Circular IQ history buffer.  After each block's M new samples are
// converted from SC16Q11 they are pushed here so that on detection we can
// extract ±60,000 samples around the peak lag from the continuous stream,
// regardless of where the peak falls within the FFT window.
static constexpr int     HIST_SAMPLES = 131072;   // 2^17 > 2*60000+1
static cufftComplex      g_iqHistory[HIST_SAMPLES];
static int               g_histHead  = 0;    // next write slot (circular)
static int64_t           g_histTotal = 0;    // total samples pushed

static bool checkBladerf(int status, const char* what) {
    if (status != 0) {
        fprintf(stderr, "bladeRF error (%s): %s\n", what, bladerf_strerror(status));
        return false;
    }
    return true;
}

#include "../include/prn_templates.hpp"

// Converts one row of interleaved I/Q floats into the cufftComplex array
// loadCode() expects.
static std::vector<cufftComplex> templateToComplex(const float* interleavedIQ, int len) {
    std::vector<cufftComplex> out(len);
    for (int i = 0; i < len; ++i) {
        out[i].x = interleavedIQ[2 * i];
        out[i].y = interleavedIQ[2 * i + 1];
    }
    return out;
}

// Smallest power of two >= n. Used to size the FFT from whatever
// TEMPLATE_LEN turns out to be, since it's no longer a fixed 127 chips —
// it depends on your upsampling factor and pulse length.
static int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int main(int argc, char** argv) {
    // ---------------- Configuration ----------------
    // Usage: correlator <sample_rate_hz> <num_codes>
    //   sample_rate_hz — bladeRF RX sample rate in samples/second.
    //                    The RX bandwidth is set to the same value.
    //   num_codes      — number of PN codes to correlate against.
    //                    In normal mode: must be <= numCodesInTable (from prn_templates.hpp).
    //                    In TIMING_BENCHMARK mode: any positive integer; zero-filled templates.
    //   Example: ./correlator 6000000 35    (6 Msps, 35 codes)
    //            ./correlator 10000000 10   (10 Msps, 10 codes)
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <sample_rate_hz> <num_codes>\n", argv[0]);
        fprintf(stderr, "  example: %s 6000000 35\n", argv[0]);
        return 1;
    }
    char* endptr = nullptr;
    const long long sampleRateArg = std::strtoll(argv[1], &endptr, 10);
    if (endptr == argv[1] || *endptr != '\0' || sampleRateArg <= 0) {
        fprintf(stderr, "Error: sample_rate_hz must be a positive integer, got '%s'\n", argv[1]);
        return 1;
    }
    const unsigned int sampleRate = static_cast<unsigned int>(sampleRateArg);

    char* endptr2 = nullptr;
    const long long numCodesArg = std::strtoll(argv[2], &endptr2, 10);
    if (endptr2 == argv[2] || *endptr2 != '\0' || numCodesArg <= 0) {
        fprintf(stderr, "Error: num_codes must be a positive integer, got '%s'\n", argv[2]);
        return 1;
    }

    const uint64_t     centerFreq = 435000000ULL; // set to your band
    const int          rxGainDb   = 10;
    const bladerf_channel rxChan  = BLADERF_CHANNEL_RX(0);

    // codeLen: in TIMING_BENCHMARK mode, compute from sample rate so N and L
    // scale correctly without recompiling.  In normal mode, use the compile-time
    // TEMPLATE_LEN from prn_templates.hpp (which must match the hardware sample rate).
    // Reference: 24552 samples at 6 Msps (the baseline rate used when generating
    // the pulse-shaped templates); scale linearly for other rates.
#ifdef TIMING_BENCHMARK
    const int codeLen  = static_cast<int>(std::round(24552.0 * sampleRate / 6000000.0));
#else
    const int codeLen  = TEMPLATE_LEN; // upsampled, pulse-shaped template length
#endif
    // fftSize: smallest power-of-two >= 2*codeLen-1 (overlap-save requirement).
    // Previously hardcoded to 65536; now derived from codeLen so benchmark sweeps
    // across sample rates automatically pick the correct FFT size.
    const int fftSize  = nextPow2(2 * codeLen - 1);

    // numCodes: from command-line argument.  In normal mode validate against the
    // table size so we never overrun prnTemplateTable[].
    const int numCodes = static_cast<int>(numCodesArg);
#ifndef TIMING_BENCHMARK
    if (numCodes > numCodesInTable) {
        fprintf(stderr, "Error: num_codes=%d exceeds numCodesInTable=%d in prn_templates.hpp\n",
                numCodes, numCodesInTable);
        return 1;
    }
#endif

    std::signal(SIGINT, handleSigint);

    // ---------------- bladeRF setup ----------------
    struct bladerf* dev = nullptr;
    int status = bladerf_open(&dev, nullptr); // nullptr = first available device
    if (!checkBladerf(status, "bladerf_open")) return 1;

    // Lock the onboard VCTCXO to an external 10 MHz reference fed into the
    // bladeRF's CLKIN SMA port.  The ADF4002 reference PLL disciplines the
    // VCTCXO so the LO and sample clock are steered by the external reference
    // rather than free-running.  The VCTCXO remains the actual clock source;
    // the PLL just phase-locks it.  Must be called before set_sample_rate /
    // set_frequency so the synthesizer programs against the locked oscillator.
    status = bladerf_set_pll_enable(dev, true);
    if (!checkBladerf(status, "set_pll_enable")) return 1;

    status = bladerf_set_pll_refclk(dev, 10000000UL);   // 10 MHz reference
    if (!checkBladerf(status, "set_pll_refclk")) return 1;

    status = bladerf_set_sample_rate(dev, rxChan, sampleRate, nullptr);
    if (!checkBladerf(status, "set_sample_rate")) return 1;

    // Set RX bandwidth equal to the sample rate.  The bladeRF will round to
    // the nearest supported value and write the actual bandwidth back through
    // the last argument (nullptr = don't need it here).
    status = bladerf_set_bandwidth(dev, rxChan, sampleRate, nullptr);
    if (!checkBladerf(status, "set_bandwidth")) return 1;

    status = bladerf_set_frequency(dev, rxChan, centerFreq);
    if (!checkBladerf(status, "set_frequency")) return 1;

    status = bladerf_set_gain_mode(dev, rxChan, BLADERF_GAIN_MGC); // manual gain control
    if (!checkBladerf(status, "set_gain_mode")) return 1;

    status = bladerf_set_gain(dev, rxChan, rxGainDb);
    if (!checkBladerf(status, "set_gain")) return 1;

    // Sync interface config. buffer_size must be a multiple of 1024 and
    // divide evenly into num_buffers' worth of streaming; these values
    // are reasonable defaults for a few-Msps stream — tune for latency
    // vs. USB overrun margin if you push the sample rate up.
    const unsigned int numBuffers   = 16;
    const unsigned int bufferSize   = 8192;
    const unsigned int numTransfers = 8;
    const unsigned int timeoutMs    = 3500;

    status = bladerf_sync_config(dev, BLADERF_RX_X1, BLADERF_FORMAT_SC16_Q11,
                                  numBuffers, bufferSize, numTransfers, timeoutMs);
    if (!checkBladerf(status, "sync_config")) return 1;

    status = bladerf_enable_module(dev, rxChan, true);
    if (!checkBladerf(status, "enable_module (rx)")) return 1;

    // ---------------- GPU correlator bank ----------------
    GpuCorrelatorBank bank(codeLen, fftSize, numCodes);

    // Load the code bank.  loadCode() zero-pads each template to fftSize,
    // forward FFTs it, conjugates it, and stores it — once at startup.
    // In TIMING_BENCHMARK mode all templates are zero-filled (timing is
    // independent of template content); in normal mode, the first numCodes
    // rows of prnTemplateTable[] (pulse-shaped waveforms) are used.
#ifdef TIMING_BENCHMARK
    {
        std::vector<cufftComplex> zeroCode(codeLen, {0.f, 0.f});
        for (int i = 0; i < numCodes; ++i)
            bank.loadCode(i, zeroCode.data());
    }
#else
    for (int i = 0; i < numCodes; ++i) {
        std::vector<cufftComplex> code = templateToComplex(prnTemplateTable[i], codeLen);
        bank.loadCode(i, code.data());
    }
#endif

    const int L = bank.blockLen(); // new data points per block (Proakis L; L = N-M+1)
    const int M = bank.codeLen(); // filter duration / code length (Proakis M)
    const int N = bank.fftSize();

    // Ring buffer holding N samples: [M-1 history | L new].  After each
    // block we shift the last M-1 samples to the front for the next
    // block's history (classic overlap-save bookkeeping, Proakis §7.3.2).
    cufftComplex* h_ring = nullptr;
    // Plain pinned memory — processBlock() uses cudaMemcpyHostToDevice so
    // the GPU sees the CPU's freshly written samples through the coherent
    // DMA path rather than potentially stale CPU-cache data.
    cudaError_t err = cudaHostAlloc((void**)&h_ring, sizeof(cufftComplex) * N, cudaHostAllocDefault);
    memset(h_ring, 0, sizeof(cufftComplex)*N);
    if (err != cudaSuccess) { fprintf(stderr, "cudaHostAlloc failed: %s\n", cudaGetErrorString(err)); return 1; }

    for (int i = 0; i < N; ++i) { h_ring[i].x = 0.f; h_ring[i].y = 0.f; }

    // SC16Q11 staging buffer for one block's worth of new samples
    // (interleaved I/Q, 2 int16 per complex sample).
    std::vector<int16_t> sc16Buf((size_t)L * 2);

    std::vector<float> peakMag(numCodes);
    std::vector<int>   peakLag(numCodes);

    // Correlation output buffers — allocated once, reused every block.
    // corrBuf: full N-sample IFFT output for one code (from getLastCorrelation).
    // f32CorrBuf: the M valid samples packed as interleaved float32 I/Q.
    // invN: the normalization factor; cuFFT C2C inverse does NOT divide by N.
    const float invN = 1.0f / static_cast<float>(N);
    std::vector<cufftComplex> corrBuf((size_t)N);
    std::vector<float>        f32CorrBuf((size_t)L * 2);

    // -----------------------------------------------------------------
    // Binary capture files for offline verification of the GPU correlator.
    //
    // raw_capture.bin        — raw SC16_Q11 samples exactly as received
    //                          from libbladeRF.  M complex samples
    //                          (2 × int16_t each) are appended per block.
    //                          The format is identical to a bladeRF
    //                          rx_samples file, so any tool that reads
    //                          SC16_Q11 (e.g. numpy, GNU Radio, bladeRF
    //                          CLI) can load it directly.
    //
    // corr_c##_capture.bin   — GPU correlator output for code ##, written
    //                          as complex float32 (NOT SC16_Q11).  For
    //                          each block the L valid lags [0, L) of
    //                          the IFFT output are normalised by N and
    //                          written as pairs of IEEE-754 float32
    //                          [I0, Q0, I1, Q1, ...].
    //                          Lag k of block b is at byte offset
    //                          (b*L + k) * 8 in the file.
    //
    //                          NOTE: the templates in prnTemplateTable are
    //                          at SC16_Q11 scale (values up to ~±1000),
    //                          while the input was divided by 2048 to
    //                          reach float32 scale.  The correlation
    //                          output therefore carries a 2048× gain
    //                          relative to what scipy would produce with
    //                          float32 templates.  Divide this file's
    //                          values by 2048 (or the scipy output by
    //                          1/2048) before comparing.
    // -----------------------------------------------------------------
#ifdef WRITE_CAPTURE_FILES
    // ---- capture mode: open output files and set the time limit -----------
    static constexpr double CAPTURE_DURATION_S = 5.0;

    FILE* rawFile = fopen("raw_capture.bin", "wb");
    if (!rawFile) { fprintf(stderr, "Cannot open raw_capture.bin\n"); return 1; }

    std::vector<FILE*> corrFiles(numCodes, nullptr);
    for (int c = 0; c < numCodes; ++c) {
        char cfname[64];
        snprintf(cfname, sizeof(cfname), "corr_c%02d_capture.bin", c);
        corrFiles[c] = fopen(cfname, "wb");
        if (!corrFiles[c]) {
            fprintf(stderr, "Cannot open %s\n", cfname);
            fclose(rawFile);
            return 1;
        }
    }

    auto captureStart = std::chrono::steady_clock::now();

    printf("Capture mode: recording %.0f s of data to SC16_Q11 binary files.\n"
           "  N=%d  M(codeLen)=%d  L(blockLen)=%d  bank=%d codes\n"
           "  raw_capture.bin + corr_c##_capture.bin (one per code)\n",
           CAPTURE_DURATION_S, N, M, L, numCodes);
#else
    // ---- continuous mode: no files, run until SIGINT ---------------------
#ifdef TIMING_BENCHMARK
#  ifndef BENCHMARK_BLOCKS
#    define BENCHMARK_BLOCKS 1000
#  endif
    printf("TIMING BENCHMARK MODE. N=%d  M(codeLen)=%d  "
           "L(blockLen)=%d  bank=%d codes  blocks=%d\n"
           "  budget=%.3f ms  Running...\n",
           N, M, L, numCodes, BENCHMARK_BLOCKS,
           1000.0 * L / sampleRate);
#else
    printf("Continuous mode (no file output). N=%d  M(codeLen)=%d  "
           "L(blockLen)=%d  bank=%d codes\n"
           "  Press Ctrl-C to stop.\n",
           N, M, L, numCodes);
#endif
#endif

    // Buffer count since the last timestamp grab (the one above, or the
    // most recent overflow re-grab below).
    uint32_t bufferNum = 0;

    // ------------------------------------------------------------------
    // Real-time characterization: timing stats collected every STAT_INTERVAL
    // blocks.  Reports show mean and worst-case latency for each pipeline
    // stage so you can determine the headroom at this (sampleRate, numCodes).
    // ------------------------------------------------------------------
    {
        size_t gpuFree = 0, gpuTotal = 0;
        cudaMemGetInfo(&gpuFree, &gpuTotal);
        printf("GPU memory: %.0f MB free / %.0f MB total\n",
               gpuFree / 1e6, gpuTotal / 1e6);
        size_t gpuUsed = (size_t)N * (size_t)(numCodes + 3) * sizeof(cufftComplex)
                       + (size_t)numCodes * (sizeof(float) + sizeof(int));
        printf("  correlator bank allocated: ~%.1f MB  (%d codes × N=%d)\n",
               gpuUsed / 1e6, numCodes, N);
    }

    static constexpr int STAT_INTERVAL = 100; // blocks between reports
    struct {
        // Per-interval accumulators — reset every STAT_INTERVAL blocks.
        // rx_ms:    time bladerf_sync_rx blocked (≈ budget when keeping up)
        // gpu_ms:   time for processBlock() — forward FFT, multiply, IFFT, peak search
        // cpu_ms:   SC16→float32 conversion + detection loop + memmove
        // total_ms: gpu_ms + cpu_ms  (processing work that must fit within budget)
        double rx_ms = 0, gpu_ms = 0, cpu_ms = 0, total_ms = 0;
        double rx_max = 0, gpu_max = 0, cpu_max = 0, total_max = 0;
        int    count  = 0;
        // Lifetime accumulators — never reset; used for benchmark summary.
        double lt_rx_ms = 0, lt_gpu_ms = 0, lt_cpu_ms = 0, lt_total_ms = 0;
        double lt_rx_max = 0, lt_gpu_max = 0, lt_cpu_max = 0, lt_total_max = 0;
        int    lt_count = 0;
        int    bladerf_errors = 0; // cumulative SDR rx failures
        FILE*  csv    = nullptr; // kept across reset()
        void reset() {
            // Preserve lifetime fields, csv pointer, and error count.
            FILE*  f    = csv;
            double lrx  = lt_rx_ms,   lgpu  = lt_gpu_ms,
                   lcpu = lt_cpu_ms,  ltot  = lt_total_ms;
            double lrxmx = lt_rx_max, lgmx  = lt_gpu_max,
                   lcmx = lt_cpu_max, ltmx  = lt_total_max;
            int    lc   = lt_count,   le    = bladerf_errors;
            *this = {};
            csv = f;
            lt_rx_ms  = lrx;   lt_gpu_ms  = lgpu;
            lt_cpu_ms = lcpu;  lt_total_ms = ltot;
            lt_rx_max = lrxmx; lt_gpu_max  = lgmx;
            lt_cpu_max = lcmx; lt_total_max = ltmx;
            lt_count = lc;  bladerf_errors = le;
        }
        void accumulate(double rx, double gpu, double cpu) {
            double tot = gpu + cpu;
            // Per-interval
            rx_ms    += rx;  gpu_ms   += gpu;
            cpu_ms   += cpu; total_ms += tot;
            if (rx  > rx_max)    rx_max    = rx;
            if (gpu > gpu_max)   gpu_max   = gpu;
            if (cpu > cpu_max)   cpu_max   = cpu;
            if (tot > total_max) total_max = tot;
            ++count;
            // Lifetime
            lt_rx_ms  += rx;  lt_gpu_ms  += gpu;
            lt_cpu_ms += cpu; lt_total_ms += tot;
            if (rx  > lt_rx_max)    lt_rx_max    = rx;
            if (gpu > lt_gpu_max)   lt_gpu_max   = gpu;
            if (cpu > lt_cpu_max)   lt_cpu_max   = cpu;
            if (tot > lt_total_max) lt_total_max = tot;
            ++lt_count;
        }
        // Write one CSV row for a bladeRF rx failure.  Timing fields are
        // empty; the error string goes in the note column so the row is
        // identifiable by a parser (bladerf_errors > 0, note non-empty).
        void write_error(double elapsed_s, unsigned sr, int nc, int N_val,
                         double budget_ms, const char* msg) {
            if (!csv) return;
            fprintf(csv,
                "%.1f,0,%.2f,%d,%d"  // elapsed + sentinel block count
                ",%.4f"
                ",,,,,,,,,,"       // rx/gpu/cpu/total/margin all blank
                "%d"               // bladerf_errors = 1 for this row
                ",\"bladerf_error: %s\"\n",
                elapsed_s, (double)sr / 1e6, nc, N_val,
                budget_ms,
                1, msg);
            fflush(csv);
            fprintf(stderr, "bladeRF error logged to CSV: %s\n", msg);
        }
        // Write one CSV row for a periodic timing report.
        // elapsed_s is wall-clock seconds since the main loop started.
        void report(double budget_ms, int L_val, int N_val,
                    unsigned sr, int nc, double elapsed_s) const {
            if (!csv) return;
            double n      = count ? count : 1;
            double margin = budget_ms - total_ms / n;
            fprintf(csv,
                "%.1f,%d,%.2f,%d,%d"
                ",%.4f"
                ",%.4f,%.4f"
                ",%.4f,%.4f"
                ",%.4f,%.4f"
                ",%.4f,%.4f"
                ",%.4f,%.2f"
                ",%d,"             // bladerf_errors, empty note
                "\n",
                elapsed_s, count, (double)sr / 1e6, nc, N_val,
                budget_ms,
                rx_ms / n, rx_max,
                gpu_ms / n, gpu_max,
                cpu_ms / n, cpu_max,
                total_ms / n, total_max,
                margin, 100.0 * margin / budget_ms,
                bladerf_errors);
            fflush(csv);
            // Brief terminal line so the operator knows it's alive.
            const char* flag = (margin < 0)           ? " *** OVERRUN ***"
                             : (total_max > budget_ms) ? " (worst-case overrun)"
                             : (bladerf_errors > 0)    ? " (bladeRF errors)"
                             : "";
            printf("[t=%6.0fs] margin %+.2f ms (%.1f%% of %.2f ms budget)%s\n",
                   elapsed_s, margin, 100.0 * margin / budget_ms,
                   budget_ms, flag);
        }
#ifdef TIMING_BENCHMARK
        // Write a final summary row tagged "BENCHMARK_SUMMARY" and print
        // a clean terminal report averaged over all lt_count blocks.
        void report_benchmark(double budget_ms, int N_val,
                              unsigned sr, int nc, double elapsed_s) const {
            double n      = lt_count ? (double)lt_count : 1.0;
            double margin = budget_ms - lt_total_ms / n;
            if (csv) {
                fprintf(csv,
                    "%.1f,%d,%.2f,%d,%d"
                    ",%.4f"
                    ",%.4f,%.4f"
                    ",%.4f,%.4f"
                    ",%.4f,%.4f"
                    ",%.4f,%.4f"
                    ",%.4f,%.2f"
                    ",%d,\"BENCHMARK_SUMMARY\""
                    "\n",
                    elapsed_s, lt_count, (double)sr / 1e6, nc, N_val,
                    budget_ms,
                    lt_rx_ms  / n, lt_rx_max,
                    lt_gpu_ms / n, lt_gpu_max,
                    lt_cpu_ms / n, lt_cpu_max,
                    lt_total_ms / n, lt_total_max,
                    margin, 100.0 * margin / budget_ms,
                    bladerf_errors);
                fflush(csv);
            }
            const char* verdict = (margin < 0)               ? "OVERRUN"
                                : (lt_total_max > budget_ms) ? "MARGINAL (worst-case overrun)"
                                : "OK";
            printf("\n=== TIMING BENCHMARK SUMMARY (%d blocks) ===\n", lt_count);
            printf("  config      : %u MSPS  %d codes  N=%d\n",
                   sr / 1000000u, nc, N_val);
            printf("  budget      : %.3f ms\n", budget_ms);
            printf("  rx   mean   : %.3f ms  (max %.3f ms)\n",
                   lt_rx_ms / n, lt_rx_max);
            printf("  gpu  mean   : %.3f ms  (max %.3f ms)\n",
                   lt_gpu_ms / n, lt_gpu_max);
            printf("  cpu  mean   : %.3f ms  (max %.3f ms)\n",
                   lt_cpu_ms / n, lt_cpu_max);
            printf("  total mean  : %.3f ms  (max %.3f ms)\n",
                   lt_total_ms / n, lt_total_max);
            printf("  margin      : %+.3f ms  (%.1f%% of budget)\n",
                   margin, 100.0 * margin / budget_ms);
            printf("  bladeRF err : %d\n", bladerf_errors);
            printf("  verdict     : %s\n", verdict);
            printf("============================================\n\n");
        }
#endif // TIMING_BENCHMARK
    } stats;

    const double block_budget_ms = 1000.0 * L / sampleRate;

    // Detection threshold for correlation peaks.
    // peakMag[c] = max(|corr[k]| / N) over the L valid lags.
    // Set this empirically from observed noise-floor values in
    // detections.csv; 0.10 is a reasonable starting point.
    constexpr float DETECTION_THRESHOLD = 20000.0f;

    // Open the CSV and write the header row.
    {
        stats.csv = fopen("timing.csv", "w");
        if (!stats.csv) {
            fprintf(stderr, "Warning: cannot open timing.csv — timing will not be recorded\n");
        } else {
            fprintf(stats.csv,
                "elapsed_s,blocks,sample_rate_msps,num_codes,fft_size"
                ",block_budget_ms"
                ",rx_mean_ms,rx_max_ms"
                ",gpu_mean_ms,gpu_max_ms"
                ",cpu_mean_ms,cpu_max_ms"
                ",total_mean_ms,total_max_ms"
                ",margin_ms,margin_pct"
                ",bladerf_errors,note"
                "\n");
            fflush(stats.csv);
        }
    }

    // Open the detections CSV.
    FILE* detFile = fopen("detections.csv", "w");
    if (!detFile) {
        fprintf(stderr, "Warning: cannot open detections.csv — detections will not be recorded\n");
    } else {
        fprintf(detFile,
            "elapsed_s,block,code,peak_mag,lag_in_block,stream_sample\n");
        fflush(detFile);
    }

    auto loopEpoch = std::chrono::steady_clock::now(); // t=0 for elapsed_s

    while (!g_stop) {
        auto tLoopStart = std::chrono::steady_clock::now();
        status = bladerf_sync_rx(dev, sc16Buf.data(), (unsigned int)L, nullptr, timeoutMs);
        if (status != 0) {
            // Log the failure to the CSV immediately (don't wait for the next
            // periodic report) so the exact block and timestamp are recorded.
            double err_elapsed = std::chrono::duration<double>(
                tLoopStart - loopEpoch).count();
            stats.bladerf_errors++;
            stats.write_error(err_elapsed, sampleRate, numCodes, N,
                              block_budget_ms, bladerf_strerror(status));
            // Zero the overlap history so the next block does not correlate
            // new samples against stale, temporally discontinuous data.
            // One block of correlation output will be unreliable (the first
            // block after recovery), but the algorithm reconverges immediately
            // because the history is now consistent with the incoming stream.
            std::memset(h_ring, 0, sizeof(cufftComplex) * N);
            continue;
        }
        auto tAfterRx = std::chrono::steady_clock::now();

#ifdef WRITE_CAPTURE_FILES
        // Write raw SC16_Q11 samples to the capture file before conversion.
        // sc16Buf holds exactly M interleaved int16_t I/Q pairs straight
        // from libbladeRF — this is the byte-identical wire format.
        fwrite(sc16Buf.data(), sizeof(int16_t), (size_t)L * 2, rawFile);
#endif

        // Convert SC16Q11 -> float32 directly into the mapped ring buffer
        // (the GPU-visible "new samples" region, after the M-1 history).
        cufftComplex* newSamples = h_ring + (M - 1);
        const float scale = 1.0f / 2048.0f; // Q11
        for (int n = 0; n < L; ++n) {
            newSamples[n].x = sc16Buf[2 * n]     * scale;
            newSamples[n].y = sc16Buf[2 * n + 1] * scale;
        }

        // Push the L new float32 samples into the IQ history ring buffer
        // before processBlock so detection logic can read them by absolute
        // stream position.
        for (int n = 0; n < L; ++n) {
            g_iqHistory[g_histHead] = newSamples[n];
            if (++g_histHead == HIST_SAMPLES) g_histHead = 0;
        }
        g_histTotal += L;
        auto tAfterConv = std::chrono::steady_clock::now();
#ifdef WRITE_CAPTURE_FILES
        // DIAGNOSTIC: dump h_ring for first few blocks, compare with MATLAB
        if (bufferNum < 5) {
            char fname[64];
            snprintf(fname, sizeof(fname), "ring_block_%d.bin", bufferNum);
            FILE* f = fopen(fname, "wb");
            fwrite(h_ring, sizeof(cufftComplex), N, f);
            fclose(f);
        }
#endif
        // Pass h_ring (host pointer) so processBlock() uses H2D memcpy,
        // guaranteeing the GPU sees the CPU's SC16Q11->float32 writes.
        bank.processBlock(h_ring, peakMag.data(), peakLag.data());
        auto tAfterProc = std::chrono::steady_clock::now();

        // Detection: check each code's peak against the threshold.
        // stream_sample is the absolute 0-based sample index of the peak
        // in the full incoming stream: g_histTotal was incremented by L
        // above, so the current block's L samples live at
        // [g_histTotal-L, g_histTotal-1] and the peak at lag k is at
        // position g_histTotal - L + k.
        if (detFile) {
            double det_elapsed = std::chrono::duration<double>(
                tAfterProc - loopEpoch).count();
            for (int c = 0; c < numCodes; ++c) {
                if (peakMag[c] > DETECTION_THRESHOLD) {
                    long long streamPos =
                        static_cast<long long>(g_histTotal) - L + peakLag[c];
                    fprintf(detFile,
                        "%.3f,%u,%d,%.6f,%d,%lld\n",
                        det_elapsed, bufferNum, c,
                        peakMag[c], peakLag[c], streamPos);
                    fflush(detFile);
                    printf("[DETECT t=%.1fs] block=%u code=%d "
                           "mag=%.6f lag=%d stream_sample=%lld\n",
                           det_elapsed, bufferNum, c,
                           peakMag[c], peakLag[c], streamPos);
                }
            }
        }

#ifdef WRITE_CAPTURE_FILES
        // Write GPU correlator output for every code as complex float32.
        // (corrBuf and f32CorrBuf are allocated once before the loop.)
        for (int c = 0; c < numCodes; ++c) {
            // Copy the full IFFT row for this code to the host.
            // In overlap-save the VALID region is [M-1, N-1] (= the last L
            // samples, Proakis eq. 7.3.9).  The first M-1 = codeLen-1
            // samples are circular-wraparound artefacts and must be skipped.
            bank.getLastCorrelation(c, corrBuf.data());

            for (int k = 0; k < L; ++k) {
                const cufftComplex& s = corrBuf[(size_t)(M - 1 + k)];
                f32CorrBuf[2 * k]     = s.x * invN;
                f32CorrBuf[2 * k + 1] = s.y * invN;
            }
            fwrite(f32CorrBuf.data(), sizeof(float), (size_t)L * 2, corrFiles[c]);
        }
#endif

        // Overlap-save bookkeeping: shift the last M-1 samples of this
        // block to the front of the ring buffer as history for the next
        // block's forward FFT window (Proakis §7.3.2).
        std::memmove(h_ring, h_ring + L, sizeof(cufftComplex) * (M - 1));
        auto tAfterMemmove = std::chrono::steady_clock::now();

        // Accumulate timing for this block.
        //   gpu_ms  = processBlock (FFT bank + multiply + IFFT + peak search)
        //   cpu_ms  = SC16→float32 conversion + detection loop + memmove
        //   total   = gpu + cpu  (must fit within block_budget_ms)
        //   margin  = block_budget_ms - total
        {
            using ms = std::chrono::duration<double, std::milli>;
            double rx_ms  = ms(tAfterRx      - tLoopStart).count();
            double gpu_ms = ms(tAfterProc    - tAfterConv).count();
            double cpu_ms = ms(tAfterConv    - tAfterRx).count()      // SC16 convert
                          + ms(tAfterMemmove - tAfterProc).count();   // detect + memmove
            stats.accumulate(rx_ms, gpu_ms, cpu_ms);
            if (stats.count >= STAT_INTERVAL) {
                double elapsed_s = std::chrono::duration<double>(
                    tAfterMemmove - loopEpoch).count();
                stats.report(block_budget_ms, L, N, sampleRate, numCodes, elapsed_s);
                stats.reset();
            }
        }

        // Progress display.  peakMag[] is filled by peakSearchKernel and
        // represents max(|corr[k]| / N) over the L valid lags.
#ifdef WRITE_CAPTURE_FILES
        // Capture mode: print every 50 blocks and stop after CAPTURE_DURATION_S.
        if (bufferNum % 50 == 0) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - captureStart).count();
            printf("  block %u  t=%.2fs  peak_mag:", bufferNum + 1, elapsed);
            for (int c = 0; c < numCodes; ++c)
                printf("  c%d=%.4f", c, peakMag[c]);
            printf("\n");
        }
        {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - captureStart).count();
            if (elapsed >= CAPTURE_DURATION_S) g_stop = 1;
        }
#else
        // Continuous mode: print every 500 blocks (~indefinite).
        if (bufferNum % 500 == 0) {
            printf("  block %u  peak_mag:", bufferNum + 1);
            for (int c = 0; c < numCodes; ++c)
                printf("  c%d=%.4f", c, peakMag[c]);
            printf("\n");
        }
#endif

        ++bufferNum;
#ifdef TIMING_BENCHMARK
        if (bufferNum >= BENCHMARK_BLOCKS) break;
#endif
    } //End of main while loop

#ifdef TIMING_BENCHMARK
    {
        double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - loopEpoch).count();
        stats.report_benchmark(block_budget_ms, N, sampleRate,
                               numCodes, elapsed_s);
    }
#endif

    if (stats.csv) { fclose(stats.csv); stats.csv = nullptr; }
    if (detFile)   { fclose(detFile);   detFile   = nullptr; }
    bladerf_enable_module(dev, rxChan, false);
    bladerf_close(dev);
    cudaFreeHost(h_ring);
#ifdef WRITE_CAPTURE_FILES
    fclose(rawFile);
    for (FILE* f : corrFiles) if (f) fclose(f);
    printf("Capture complete: %u blocks, %.1f s of data written.\n",
           bufferNum,
           std::chrono::duration<double>(
               std::chrono::steady_clock::now() - captureStart).count());
#else
#ifdef TIMING_BENCHMARK
    printf("Benchmark complete: %u blocks.\n", bufferNum);
#else
    printf("Stopped after %u blocks.\n", bufferNum);
#endif
#endif
    return 0;
}
