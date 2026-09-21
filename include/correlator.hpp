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

    // d_blockInput: N complex samples already on the GPU
    // (last L-1 samples from previous block + M new samples).
    // outMag/outLag: host arrays, length bankSize(), receive the
    // peak correlation magnitude and lag (0..M-1) for each code.
    void processBlock(const cufftComplex* d_blockInput, float* outMag, int* outLag);

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
