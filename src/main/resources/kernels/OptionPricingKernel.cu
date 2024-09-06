extern "C"

#define MS_IN_YEAR 31536000000L;

__global__ void fairPx(
unsigned int n,
unsigned long long int timeMs,
float fwdPx,
void* optInputs,
float *g_odata)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int strikeOffset = 0;
        int expiryOffset = strikeOffset + n * sizeof(float);
        int isCallOffset = expiryOffset + n * sizeof(unsigned long long int);
        int volOffset = isCallOffset + n * sizeof(char);
        int rateOffset = volOffset + n * sizeof(float);

//         printf("i %d n %d [%d %d %d %d %d]\n", i, n, strikeOffset, expiryOffset, isCallOffset, volOffset, rateOffset);

        float fairPx = 0;
        float s = ((float*)optInputs)[i];
        unsigned long long int expiry = ((unsigned long long int*)((char*)optInputs + expiryOffset))[i];
        char isC = ((char*)optInputs + isCallOffset)[i];
        float v = ((float* )((char*)optInputs + volOffset))[i];
        float r = ((float* )((char*)optInputs + rateOffset))[i];

//         printf("i %d n %d [%f %f %f %d %llu]\n", i, n, v, s, r, isC, expiry);

        for(int j = 0; j < 80; ++j) {
            float tte = (expiry - timeMs) * 1.0 / MS_IN_YEAR;
            float fwdOverStrike = fwdPx / s;
            float scaledVol = v * sqrt(tte);
            float d1 = logf(fwdOverStrike) / scaledVol + scaledVol / 2;
            float d2 = d1 - scaledVol;
            float discount = expf(-1 * r * tte);
            if(isC == 1) {
                fairPx = discount * (normcdff(d1) - normcdff(d2) / fwdOverStrike);
            } else {
                fairPx = discount * (normcdff(-d2) / fwdOverStrike - normcdff(-d1));
            }
        }

        // write result for this block to global mem
        g_odata[i] = fairPx;
    }
//*/
}