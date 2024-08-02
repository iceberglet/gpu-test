package benchmark;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.samples.utils.JCudaSamplesUtils;

import java.util.ArrayList;
import java.util.List;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

public class CudaOptionPricer implements OptionPricer {

    private final CUfunction function = new CUfunction();
    //max computed at the same time
    private final int maxOptions = 4096;
    private final CUdeviceptr expiryInput = new CUdeviceptr();
    private final CUdeviceptr volInput = new CUdeviceptr();
//    private final CUdeviceptr fwdPxInput = new CUdeviceptr();
    private final CUdeviceptr rateInput = new CUdeviceptr();
    private final CUdeviceptr strikeInput = new CUdeviceptr();
    private final CUdeviceptr isCallInput = new CUdeviceptr();
    private final CUdeviceptr fairPxOutput = new CUdeviceptr();

    private final long[] expiries = new long[maxOptions];
    private final double[] vols = new double[maxOptions];
//    private final double[] fwdPxs = new double[maxOptions];
    private final double[] rates = new double[maxOptions];
    private final double[] strikes = new double[maxOptions];
    private final byte[] isCall = new byte[maxOptions];

    private final double[] fairPxs = new double[maxOptions];

    @Override
    public void init() {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaSamplesUtils.preparePtxFile(
                "src/main/resources/kernels/OptionPricingKernel.cu");

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "add" function.
        cuModuleGetFunction(function, module, "fairPx");

        cuMemAlloc(expiryInput, maxOptions * Sizeof.LONG);
        cuMemAlloc(volInput, maxOptions * Sizeof.DOUBLE);
//        cuMemAlloc(fwdPxInput, maxOptions * Sizeof.DOUBLE);
        cuMemAlloc(rateInput, maxOptions * Sizeof.DOUBLE);
        cuMemAlloc(strikeInput, maxOptions * Sizeof.DOUBLE);
        cuMemAlloc(fairPxOutput, maxOptions * Sizeof.DOUBLE);
        cuMemAlloc(isCallInput, maxOptions * Sizeof.BYTE);
    }

    private List<OptionInst> options;

    @Override
    public void loadOptions(List<OptionInst> options, final double vol, final double rate) {
        this.options = options;
        final int numOptions = options.size();
        if(numOptions > maxOptions) {
            throw new IllegalArgumentException("Too many options! need to re-allocate memory!");
        }

        for(int i = 0; i < options.size(); i++)
        {
            final var inst = options.get(i);
            expiries[i] = inst.expiryMs;
            strikes[i] = inst.strike;
            isCall[i] = (byte)(inst.isCall? 1 : 0);
            vols[i] = vol;
            rates[i] = rate;
        }

        cuMemcpyHtoD(expiryInput, Pointer.to(expiries), numOptions * Sizeof.LONG);
        cuMemcpyHtoD(strikeInput, Pointer.to(strikes), numOptions * Sizeof.DOUBLE);
        cuMemcpyHtoD(isCallInput, Pointer.to(isCall), numOptions * Sizeof.BYTE);
        cuMemcpyHtoD(volInput, Pointer.to(vols), numOptions * Sizeof.DOUBLE);
        cuMemcpyHtoD(rateInput, Pointer.to(rates), numOptions * Sizeof.DOUBLE);
    }

    @Override
    public List<Double> price(double fwdPx, long timeMs) {
        final int numOptions = options.size();
        final List<Double> result = new ArrayList<>(numOptions);

//        for(int i = 0; i < options.size(); i++)
//        {
//            fwdPxs[i] = fwdPx;
//        }

        // Allocate the device input data, and copy the
        // host input data to the device

//        cuMemcpyHtoD(fwdPxInput, Pointer.to(fwdPxs), numOptions * Sizeof.DOUBLE);

        // Allocate device output memory

        //int n, long int timeMs
        //long int* expiryMs, float vol*, float* fwdPx, float* rate
        //float* strike, float *g_odata
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{numOptions}),
                Pointer.to(new long[]{timeMs}),
                Pointer.to(expiryInput),
                Pointer.to(volInput),
                Pointer.to(rateInput),
                Pointer.to(strikeInput),
                Pointer.to(isCallInput),
                Pointer.to(new double[]{fwdPx}),
                Pointer.to(fairPxOutput)
        );

        // Call the kernel function.
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)numOptions / blockSizeX);
        cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        cuMemcpyDtoH(Pointer.to(fairPxs), fairPxOutput, numOptions * Sizeof.DOUBLE);

        // Verify the result
        for(int i = 0; i < numOptions; i++)
        {
            result.add(fairPxs[i]);
        }

        // Clean up.
//        cuMemFree(expiryInput);
//        cuMemFree(volInput);
//        cuMemFree(fwdPxInput);
//        cuMemFree(rateInput);
//        cuMemFree(strikeInput);
//        cuMemFree(isCallInput);
//        cuMemFree(fairPxOutput);
        return result;
    }
}
