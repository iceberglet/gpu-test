extern "C"
__global__ void fairPx(unsigned int n,
unsigned long long int timeMs, unsigned long long int* expiryMs,
double* vol, double* rate, double* strike, char* isCall, double fwdPx, double *g_odata)
{
    const unsigned long long int MS_IN_YEAR = 31536000000L;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
//         printf("%d*%d+%d=%d - timeMs %llu expiryMs %llu \n", blockIdx.x, blockDim.x, threadIdx.x, i, timeMs, expiryMs[i]);
//         printf("vol %f fwdPx %f strike %f isCall %d \n", vol[i], fwdPx[i], strike[i], isCall[i]);
        double tte = (expiryMs[i] - timeMs) * 1.0 / MS_IN_YEAR;
        double fwdOverStrike = fwdPx / strike[i];
        double scaledVol = vol[i] * sqrt(tte);
        double d1 = logf(fwdOverStrike) / scaledVol + scaledVol / 2;
        double d2 = d1 - scaledVol;
        double discount = expf(-1 * rate[i] * tte);
        double fairPx = 0;
        if(isCall[i] == 1) {
            fairPx = discount * (normcdff(d1) - normcdff(d2) / fwdOverStrike);
        } else {
            fairPx = discount * (normcdff(-d2) / fwdOverStrike - normcdff(-d1));
        }

        // write result for this block to global mem
        g_odata[i] = fairPx;
    }

}
