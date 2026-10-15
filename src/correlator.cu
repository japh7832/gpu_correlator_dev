// correlator.cu
//
// GPU matched-filter bank correlator using overlap-save FFT correlation.
//
// Design:
//   - N reference codes (PN codes, etc.), each of length L samples, are
//     zero-padded to FFT size and their FFTs precomputed ONCE at startup
//     (conjugated, ready for multiply).
//   - Each incoming block of input samples (length M = FFT_SIZE - L + 1
//     "new" samples, overlapped with the previous L-1 samples) is forward
//     FFT'd ONCE.
//   - That single spectrum is broadcast-multiplied against every code's
//     conjugated spectrum (batched complex multiply), then a single batched
//     inverse FFT (batch = numCodes) produces the correlation output for
//     every code in one shot.
//   - A per-code parallel reduction finds the peak magnitude + lag.
//
// This amortizes the input FFT cost across the whole code bank, which is
// the point of doing this in frequency domain instead of direct convolution.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "../include/correlator.hpp"

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error("CUDA error"); \
    } \
} while (0)

#define CUFFT_CHECK(x) do { \
    cufftResult err = (x); \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        throw std::runtime_error("cuFFT error"); \
    } \
} while (0)

// ---------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------

// Broadcast-multiply the (single) input spectrum against every code's
// conjugated spectrum. Layout: codeSpectra is [numCodes][fftSize]
// row-major; inputSpectrum is [fftSize]. Output overwrites codeSpectra
// in place (we don't need the original code spectra again this block).
__global__ void batchedComplexMultiplyKernel(
    const cufftComplex* __restrict__ inputSpectrum,
    const cufftComplex* __restrict__ codeSpectraConj,
    cufftComplex* __restrict__ productOut,
    int fftSize,
    int numCodes)
{
    int bin  = blockIdx.x * blockDim.x + threadIdx.x;
    int code = blockIdx.y;
    if (bin >= fftSize || code >= numCodes) return;

    cufftComplex a = inputSpectrum[bin];
    cufftComplex b = codeSpectraConj[code * fftSize + bin];

    cufftComplex prod;
    prod.x = a.x * b.x - a.y * b.y;
    prod.y = a.x * b.y + a.y * b.x;

    productOut[code * fftSize + bin] = prod;
}

// Per-code peak-magnitude search over the "valid" (non-wraparound)
// portion of the overlap-save output: first M samples [0, M) are valid
// for cross-correlation (conj(FFT(h)) form). One block per code; block-wide
// reduction in shared memory.
__global__ void peakSearchKernel(
    const cufftComplex* __restrict__ corrTime, // [numCodes][fftSize], unnormalized IFFT output
    int fftSize,
    int validStart,   // = 0  (cross-correlation: valid = FIRST M samples)
    int validLen,      // = M  (= fftSize - codeLen + 1)
    float* __restrict__ peakMag,
    int* __restrict__ peakLag,
    float normFactor)  // 1/fftSize (cuFFT does not normalize the inverse transform)
{
    extern __shared__ float sdata[];
    float* smag = sdata;
    int*   slag = (int*)&sdata[blockDim.x];

    int code = blockIdx.x;
    const cufftComplex* row = corrTime + (size_t)code * fftSize;

    float bestMag = -1.0f;
    int   bestLag = -1;

    for (int n = threadIdx.x; n < validLen; n += blockDim.x) {
        cufftComplex v = row[validStart + n];
        float re = v.x * normFactor;
        float im = v.y * normFactor;
        float mag = re * re + im * im; // magnitude-squared is enough for peak comparison
        if (mag > bestMag) {
            bestMag = mag;
            bestLag = n;
        }
    }

    smag[threadIdx.x] = bestMag;
    slag[threadIdx.x] = bestLag;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smag[threadIdx.x + stride] > smag[threadIdx.x]) {
                smag[threadIdx.x] = smag[threadIdx.x + stride];
                slag[threadIdx.x] = slag[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        peakMag[code] = sqrtf(smag[0]); // report linear magnitude, not squared
        peakLag[code] = slag[0];
    }
}

// ---------------------------------------------------------------------
// GpuCorrelatorBank
// ---------------------------------------------------------------------

GpuCorrelatorBank::GpuCorrelatorBank(int codeLen, int fftSize, int numCodesIn)
    : L(codeLen), N(fftSize), numCodes(numCodesIn)
{
    if (N < 2 * L - 1) {
        throw std::runtime_error("fftSize must be >= 2*codeLen-1 for overlap-save");
    }
    M = N - L + 1; // valid new samples per block

    CUDA_CHECK(cudaMalloc(&d_codeSpectraConj, sizeof(cufftComplex) * (size_t)N * numCodes));
    CUDA_CHECK(cudaMalloc(&d_inputTime,       sizeof(cufftComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_inputSpectrum,   sizeof(cufftComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_product,         sizeof(cufftComplex) * (size_t)N * numCodes));
    CUDA_CHECK(cudaMalloc(&d_peakMag,         sizeof(float) * numCodes));
    CUDA_CHECK(cudaMalloc(&d_peakLag,         sizeof(int)   * numCodes));

    // Forward FFT plan: single transform, batch 1, reused every block.
    CUFFT_CHECK(cufftPlan1d(&fwdPlan, N, CUFFT_C2C, 1));

    // Inverse FFT plan: batched over numCodes.
    int n[1] = { N };
    CUFFT_CHECK(cufftPlanMany(&invPlan, 1, n,
                               nullptr, 1, N,   // inembed/istride/idist
                               nullptr, 1, N,   // onembed/ostride/odist
                               CUFFT_C2C, numCodes));

    CUDA_CHECK(cudaStreamCreate(&stream));
    cufftSetStream(fwdPlan, stream);
    cufftSetStream(invPlan, stream);
}

GpuCorrelatorBank::~GpuCorrelatorBank() {
    cufftDestroy(fwdPlan);
    cufftDestroy(invPlan);
    cudaFree(d_codeSpectraConj);
    cudaFree(d_inputTime);
    cudaFree(d_inputSpectrum);
    cudaFree(d_product);
    cudaFree(d_peakMag);
    cudaFree(d_peakLag);
    cudaStreamDestroy(stream);
}

int GpuCorrelatorBank::fftSize() const { return N; }
int GpuCorrelatorBank::validLen() const { return M; }
int GpuCorrelatorBank::codeLen() const { return L; }
int GpuCorrelatorBank::bankSize() const { return numCodes; }

// Load one reference code (host-side complex float array of length L).
// Zero-pads to N, forward-FFTs it, conjugates it, and stores it in the
// code bank at slot `idx`. Call once per code at startup.
void GpuCorrelatorBank::loadCode(int idx, const cufftComplex* hostCode) {
    cufftComplex* d_scratch;
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(cufftComplex) * N));
    CUDA_CHECK(cudaMemset(d_scratch, 0, sizeof(cufftComplex) * N));
    CUDA_CHECK(cudaMemcpy(d_scratch, hostCode, sizeof(cufftComplex) * L, cudaMemcpyHostToDevice));

    cufftHandle tmpPlan;
    CUFFT_CHECK(cufftPlan1d(&tmpPlan, N, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftExecC2C(tmpPlan, d_scratch, d_scratch, CUFFT_FORWARD));
    cufftDestroy(tmpPlan);

    conjugateInPlace(d_scratch, N);

    CUDA_CHECK(cudaMemcpy(d_codeSpectraConj + (size_t)idx * N, d_scratch,
                           sizeof(cufftComplex) * N, cudaMemcpyDeviceToDevice));
    cudaFree(d_scratch);
}

// Process one overlap-save block. `h_blockInput` must point to N
// time-domain samples in HOST pinned memory (the last L-1 samples
// of the previous block followed by M new samples) — see host loop
// for how the ring buffer maintains this overlap.
// Using cudaMemcpyHostToDevice (not DeviceToDevice) ensures the GPU
// sees the CPU's freshly written samples through the cache-coherent
// DMA path — required on Orin Nano's unified memory to avoid reading
// stale CPU-cached data and producing ghost correlation peaks.
// Results land in host arrays outMag/outLag, length numCodes.
void GpuCorrelatorBank::processBlock(const cufftComplex* h_blockInput, float* outMag, int* outLag) {
    CUDA_CHECK(cudaMemcpyAsync(d_inputTime, h_blockInput, sizeof(cufftComplex) * N,
                                cudaMemcpyHostToDevice, stream));

    CUFFT_CHECK(cufftExecC2C(fwdPlan, d_inputTime, d_inputSpectrum, CUFFT_FORWARD));

    dim3 mulBlock(256);
    dim3 mulGrid((N + mulBlock.x - 1) / mulBlock.x, numCodes);
    batchedComplexMultiplyKernel<<<mulGrid, mulBlock, 0, stream>>>(
        d_inputSpectrum, d_codeSpectraConj, d_product, N, numCodes);

    // In-place batched inverse FFT over d_product ([numCodes][N]).
    CUFFT_CHECK(cufftExecC2C(invPlan, d_product, d_product, CUFFT_INVERSE));

    int threads = 256;
    size_t shmem = threads * (sizeof(float) + sizeof(int));
    peakSearchKernel<<<numCodes, threads, shmem, stream>>>(
        d_product, N, 0, M, d_peakMag, d_peakLag, 1.0f / N);

    CUDA_CHECK(cudaMemcpyAsync(outMag, d_peakMag, sizeof(float) * numCodes,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(outLag, d_peakLag, sizeof(int) * numCodes,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void conjugateKernel(cufftComplex* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i].y = -d[i].y;
}

void GpuCorrelatorBank::conjugateInPlace(cufftComplex* d_data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    conjugateKernel<<<blocks, threads>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Diagnostic readback: copy the full N-sample IFFT output (d_product row
// for codeIdx) to a host buffer.  Call after processBlock(); the data is
// valid until the next processBlock() call overwrites d_product.
//
// The values are the unnormalized cuFFT IFFT output (each true correlation
// value is hostBuf[i] / N).  The valid linear-correlation region is
// samples [0, M) (0-indexed, M = N-L+1); samples [M, N) are circular-wraparound
// artefacts from overlap-save (code uses conj(FFT(h)) = cross-correlation form).
void GpuCorrelatorBank::getLastCorrelation(int codeIdx, cufftComplex* hostBuf) const {
    if (codeIdx < 0 || codeIdx >= numCodes) {
        throw std::out_of_range("getLastCorrelation: codeIdx out of range");
    }
    // d_product layout: [numCodes][N], row-major.
    CUDA_CHECK(cudaMemcpy(
        hostBuf,
        d_product + (size_t)codeIdx * N,
        sizeof(cufftComplex) * N,
        cudaMemcpyDeviceToHost));
}
