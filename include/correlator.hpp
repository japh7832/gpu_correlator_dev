#pragma once
#include <cufft.h>

// See src/correlator.cu for full documentation of the algorithm
// (overlap-save FFT correlation, batched across a bank of codes).
class GpuCorrelatorBank {
public:
    GpuCorrelatorBank(int codeLen, int fftSize, int numCodes);
    ~GpuCorrelatorBank();

    int fftSize() const;
    int validLen() const;  // M: new input samples consumed per block
    int codeLen() const;   // L: reference code length
    int bankSize() const;  // number of codes in the bank

    void loadCode(int idx, const cufftComplex* hostCode);

    // h_blockInput: N complex samples in HOST (pinned) memory
    // (last L-1 samples from previous block + M new samples).
    // Uses cudaMemcpyHostToDevice so the GPU sees the CPU's freshly
    // written samples via the cache-coherent DMA path on Orin Nano.
    // outMag/outLag: host arrays, length bankSize(), receive the
    // peak correlation magnitude and lag (0..M-1) for each code.
    void processBlock(const cufftComplex* h_blockInput, float* outMag, int* outLag);

    // Diagnostic: copy the full N-sample IFFT output for one code back to
    // the host. Must be called AFTER processBlock() and BEFORE the next
    // processBlock() call (d_product is overwritten each block).
    //
    // hostBuf must hold at least fftSize() cufftComplex elements.
    // The correlation is the unnormalized cuFFT IFFT output — divide each
    // sample by fftSize() to get the true linear cross-correlation value.
    // The valid (non-wraparound) region is samples [codeLen()-1 .. fftSize()-1],
    // i.e. indices L-1 through N-1 (0-indexed). Those M = N-L+1 samples are
    // what peakSearchKernel scans, and lag k (0-indexed) corresponds to
    // hostBuf[L-1+k] after normalization.
    void getLastCorrelation(int codeIdx, cufftComplex* hostBuf) const;

private:
    static void conjugateInPlace(cufftComplex* d_data, int n);
    int L, N, M, numCodes;
    cufftComplex* d_codeSpectraConj = nullptr;
    cufftComplex* d_inputTime = nullptr;
    cufftComplex* d_inputSpectrum = nullptr;
    cufftComplex* d_product = nullptr;
    float* d_peakMag = nullptr;
    int*   d_peakLag = nullptr;
    cufftHandle fwdPlan {};
    cufftHandle invPlan {};
    cudaStream_t stream {};
};
