# GPU Real-Time Matched-Filter Correlator — Jetson Orin Nano

A real-time matched-filter bank correlator for a low-rate (<10 Msps)
stream from a Nuand bladeRF A4 (bladeRF 2.0 micro xA4, USB 3.0, via
libbladeRF), searching many reference codes (e.g. PN/spreading codes)
against the incoming signal simultaneously, running on the Orin Nano's
integrated GPU.

## Why FFT-based (overlap-save), not direct time-domain convolution

With a **bank** of codes, the expensive part isn't correlating against one
code — it's that direct time-domain correlation costs `O(N * L)` **per
code** (N = block length, L = code length). With, say, 32 codes of length
1023, that's ~33k multiply-accumulates per output sample, times however
many samples per second you're processing.

FFT correlation flips this: the forward FFT of the incoming block is
computed **once**, and reused against every code via a batched
frequency-domain multiply + batched inverse FFT. Cost becomes
`O(N log N)` once, plus `O(N)` per code for the multiply and a batched
inverse FFT. This wins decisively once you have more than a handful of
codes or codes longer than a couple hundred taps — which is exactly the
"many reference waveforms" case here.

## Algorithm: overlap-save

- Each reference code (length `L`) is zero-padded to FFT size `N` and its
  FFT is precomputed **once at startup** (`GpuCorrelatorBank::loadCode`),
  conjugated and stored.
- `N` must satisfy `N >= 2*L - 1`. Pick a fast cuFFT size (power of two is
  simplest).
- Each processing block consumes `M = N - L + 1` **new** input samples,
  preceded by the last `L-1` samples from the previous block (the
  "overlap"). This N-sample window is forward-FFT'd once.
- That one spectrum is broadcast-multiplied against every code's
  conjugated spectrum (`batchedComplexMultiplyKernel`, grid.y = numCodes),
  then a single batched inverse FFT (`batch = numCodes`) produces every
  code's correlation output for this block in one shot.
- Output samples `[L-1, N)` in each code's row are the valid
  (non-circular-wraparound) linear-correlation outputs — that's what
  `peakSearchKernel` scans, doing a per-code parallel reduction to find
  peak magnitude + lag.

This is standard block-convolution/correlation theory (the same method
used for FIR filtering long streams via FFT); see any DSP reference on
"overlap-save" for the derivation if you want to double check the indexing.

## Files

- `include/correlator.hpp` / `src/correlator.cu` — `GpuCorrelatorBank`:
  the CUDA/cuFFT correlator core. Self-contained; no UHD dependency, so
  you can unit-test it against a file source before wiring up hardware.
- `src/host_stream.cpp` — bladeRF streaming loop (libbladeRF sync
  interface): configures the radio, receives SC16Q11 samples
  continuously, converts them to float32, maintains the overlap-save
  ring buffer, and calls `processBlock()` once per block.
- `Makefile` — builds for Orin's SM 8.7 (Ampere) by default.

## Unified memory on Orin Nano

Discrete-GPU designs need `cudaMemcpyAsync` to move samples from host to
device across PCIe. Orin Nano's GPU shares physical DRAM with the CPU, so
`host_stream.cpp` allocates the ring buffer with `cudaHostAlloc(...,
cudaHostAllocMapped)` and gets a device pointer to the *same* memory via
`cudaHostGetDevicePointer`. `processBlock()` reads that buffer as device
memory with no explicit H2D copy in the critical path (there's still one
cheap device-to-device copy inside `processBlock` to stage into the FFT
input buffer — you can remove that too and cuFFT directly on the mapped
pointer if you want to shave the last bit of latency).

One wrinkle specific to bladeRF: libbladeRF's `bladerf_sync_rx` only
hands you samples in its native **SC16Q11** wire format (16-bit signed
ints, Q11 fixed point) — unlike UHD, there's no "give me float32
directly" stream option. So `host_stream.cpp` receives into a small
`int16_t` staging buffer, then does a CPU-side conversion pass
(`* 1/2048.0f`) straight into the mapped ring buffer. That conversion is
a few thousand samples per block at these rates and is not a bottleneck,
but it does mean the zero-copy property only applies from "after
conversion" onward, not all the way from the USB transfer.

`cudaSetDeviceFlags(cudaDeviceMapHost)` must be called before the first
CUDA context-creating call, which is why it's the first CUDA call in
`main()`.

## Tunable parameters (top of `host_stream.cpp`)

| Parameter | Meaning |
|---|---|
| `sampleRate` | bladeRF RX sample rate, Hz |
| `centerFreq` | bladeRF RX center frequency, Hz |
| `rxGainDb` | manual RX gain, dB (gain mode is set to `BLADERF_GAIN_MGC`) |
| `numBuffers` / `bufferSize` / `numTransfers` / `timeoutMs` | libbladeRF sync-interface buffering — affects latency vs. USB overrun margin |
| `TEMPLATE_LEN` (top of file, above `main()`) | number of complex samples in one correlation template — set this to match your generator's actual output length |
| `prnTemplateTable` (top of file) | your pulse-shaped, upsampled correlation templates — one row per code, interleaved I/Q floats |
| `fftSize` (`N`) | computed automatically from `TEMPLATE_LEN` via `nextPow2(2*L-1)` — no longer a manual setting |
| `numCodes` | computed automatically from the number of rows in `prnTemplateTable` |
| `threshold` | detection threshold on peak magnitude — currently a placeholder scaled off `M`; needs replacing with something scaled to your templates' actual energy and a measured noise floor |

## What's stubbed out / next steps

1. **Correlation templates.** `prnTemplateTable` (top of `host_stream.cpp`,
   above `main()`) currently has empty placeholder rows — paste in your
   generator's actual pulse-shaped, upsampled I/Q samples, one row per
   code, and set `TEMPLATE_LEN` to match their length. Nothing in this
   file synthesizes chips or applies pulse shaping anymore; the loader
   just FFTs whatever samples you give it.
2. **Detection threshold.** The `0.5f * M` placeholder needs replacing
   with something threshold based on noise floor / desired false-alarm
   rate (e.g. CFAR, or normalize by input block energy).
3. **Overflow handling.** The bladeRF receive loop currently just logs
   and continues on any `bladerf_sync_rx` error; production code should
   distinguish `BLADERF_ERR_TIMEOUT` (often indicates an upstream
   overrun on the sync-buffer queue) from other errors and track/log
   overruns explicitly (USB streaming at even a few Msps can overrun if
   the host loop stalls or `numBuffers`/`bufferSize` are set too tight).
4. **Doppler search**, if your application needs it (common in GPS-style
   acquisition): extend the code bank dimension to (code × Doppler bin)
   pairs, each with its own frequency-shifted, precomputed FFT — the
   batched-multiply/batched-IFFT structure here extends directly, it just
   multiplies the batch size.
5. **Double-buffered block pipeline.** Right now `processBlock()` is
   called and synchronized every block (simple, correct, and at <10 Msps
   almost certainly fast enough). If you later push sample rate up, you
   can overlap the next UHD `recv()` with the current block's GPU work
   using two ring buffers and cuFFT's async execution on a non-blocking
   stream — the `cudaStreamSynchronize` at the end of `processBlock` is
   the thing to remove/restructure for that.

## Build

```sh
make            # builds build/correlator
make run        # builds and runs
```

Requires the CUDA toolkit (present in JetPack) and `libbladerf-dev`. The
`bladerf_open(&dev, nullptr)` call opens the first available device; pass
a device identifier string instead of `nullptr` if you need to target a
specific unit (e.g. multiple bladeRFs on one host).

The Makefile links against libbladeRF using `-l:libbladeRF.so.2` (the
linker's explicit-filename syntax) plus `-L` for its directory, rather
than plain `-lbladeRF` or a raw path. Two things it's working around:
some installs ship only the versioned `.so.2` with no unversioned
`libbladeRF.so` dev symlink for `-lbladeRF` to find; and nvcc dispatches
link inputs by file extension, so a raw path ending in `.2` can get
silently dropped instead of forwarded to the host linker. `-l:exact-name`
sidesteps both, since it's an ordinary `-l`-style flag nvcc always
forwards, and it searches for that literal filename rather than the
usual `lib<name>.so` pattern.

Similarly, the Makefile assumes `libbladeRF.h` is in `/usr/include`. If
your header or library live elsewhere, override any of these:

```sh
find / -name 'libbladeRF.h' 2>/dev/null      # locate the header
make BLADERF_INCLUDE=/path/to/header/dir \
     BLADERF_LIB_DIR=/path/to/lib/dir BLADERF_LIB_NAME=libbladeRF.so.2
```
