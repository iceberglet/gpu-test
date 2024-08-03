extern "C"

#define MS_IN_YEAR 31536000000L;

__global__ void fairPx(
unsigned int n,
unsigned long long int timeMs,
unsigned long long int* expiryMs,
float* vol, float* rate, float* strike, char* isCall, float fwdPx,
float *g_odata)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
//         printf("%d*%d+%d=%d - timeMs %llu expiryMs %llu \n", blockIdx.x, blockDim.x, threadIdx.x, i, timeMs, expiryMs[i]);
//         printf("vol %f fwdPx %f strike %f isCall %d \n", vol[i], fwdPx, strike[i], isCall[i]);
        float fairPx = 0;
        unsigned long long int expiry = expiryMs[i];
        float s = strike[i];
        float v = vol[i];
        char isC = isCall[i];
        float r = rate[i];
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