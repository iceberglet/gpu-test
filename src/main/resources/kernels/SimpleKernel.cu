extern "C"

#define MS_IN_YEAR 31536000000L;

__global__ void fairPx(
unsigned int n,
int* input,
int* g_odata)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // write result for this block to global mem
        g_odata[i] = input[i];
    }
//*/
}