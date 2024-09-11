extern "C"

#define MS_IN_YEAR 31536000000L;
#define rateOffset 4;
#define volOffset 8;
#define expiryOffset 12;
#define isCallOffset 20;
#define unitSize 24;

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
//         printf("i %d n %d [%d %d %d %d %d] %d\n", i, n, strikeOffset, expiryOffset, isCallOffset, volOffset, rateOffset, unitSize);

        float fairPx = 0;
        char* optionBytes = (char*)optInputs + i * unitSize;

        float* floatOffset = (float*)optionBytes;
        float s = floatOffset[0];
        float r = floatOffset[1];
        float v = floatOffset[2];

//         printf("i %d n %d [vol %f strike %f rate %f] \n", i, n, v, s, r);

        char* expOffset = optionBytes + expiryOffset;
        unsigned long long int expiry = *((unsigned long long int*)expOffset);
        printf("hello\n");
        char* cpOffset = optionBytes + isCallOffset;
        char isC = ((char*)cpOffset)[0];

        printf("i %d n %d [%f %f %f %d %llu]\n", i, n, v, s, r, isC, expiry);

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