package benchmark;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.JCudaDriver;
import org.lwjgl.PointerBuffer;
import org.lwjgl.cuda.CUDA;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.cuda.CU.CUDA_SUCCESS;
import static org.lwjgl.cuda.CU.cuCtxCreate;
import static org.lwjgl.cuda.CU.cuCtxDetach;
import static org.lwjgl.cuda.CU.cuDeviceComputeCapability;
import static org.lwjgl.cuda.CU.cuDeviceGet;
import static org.lwjgl.cuda.CU.cuDeviceGetCount;
import static org.lwjgl.cuda.CU.cuDeviceGetName;
import static org.lwjgl.cuda.CU.cuDeviceTotalMem;
import static org.lwjgl.cuda.CU.cuInit;
import static org.lwjgl.cuda.CU.cuLaunchKernel;
import static org.lwjgl.cuda.CU.cuMemAlloc;
import static org.lwjgl.cuda.CU.cuMemFree;
import static org.lwjgl.cuda.CU.cuMemcpyDtoH;
import static org.lwjgl.cuda.CU.cuMemcpyHtoD;
import static org.lwjgl.cuda.CU.cuModuleGetFunction;
import static org.lwjgl.cuda.CU.cuModuleLoadData;
import static org.lwjgl.cuda.NVRTC.NVRTC_SUCCESS;
import static org.lwjgl.cuda.NVRTC.nvrtcCompileProgram;
import static org.lwjgl.cuda.NVRTC.nvrtcCreateProgram;
import static org.lwjgl.cuda.NVRTC.nvrtcGetErrorString;
import static org.lwjgl.cuda.NVRTC.nvrtcGetPTX;
import static org.lwjgl.cuda.NVRTC.nvrtcGetPTXSize;
import static org.lwjgl.cuda.NVRTC.nvrtcGetProgramLog;
import static org.lwjgl.cuda.NVRTC.nvrtcGetProgramLogSize;
import static org.lwjgl.cuda.NVRTC.nvrtcVersion;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memASCII;
import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memAlloc;
import static org.lwjgl.system.MemoryUtil.memAllocDouble;
import static org.lwjgl.system.MemoryUtil.memAllocInt;
import static org.lwjgl.system.MemoryUtil.memAllocLong;


public class CudaLWJGLOptionPricer implements OptionPricer {

    private static final String KERNEL_NAME = "fairPx";

    private static long ctx;

    private PointerBuffer pp;

    private long function;

    @Override
    public void init() {
        try (MemoryStack stack = stackPush()) {
            IntBuffer major = stack.mallocInt(1);
            IntBuffer minor = stack.mallocInt(1);

            checkNVRTC(nvrtcVersion(major, minor));

            System.out.println("Compiling kernel with NVRTC v" + major.get(0) + "." + minor.get(0));

            pp = stack.mallocPointer(1);

            final String cu = Files.readString(Path.of("src/main/resources/kernels/OptionPricingKernel.cu"), StandardCharsets.UTF_8);

            checkNVRTC(nvrtcCreateProgram(pp, cu, "OptionPricingKernel.cu", null, null));
            long program = pp.get(0);

            int compilationStatus = nvrtcCompileProgram(program, null);
            {
                checkNVRTC(nvrtcGetProgramLogSize(program, pp));
                if (1L < pp.get(0)) {
                    ByteBuffer log = stack.malloc((int)pp.get(0) - 1);

                    checkNVRTC(nvrtcGetProgramLog(program, log));

//                    StringBuilder builder = new StringBuilder();
//                    while(log.hasRemaining()) {
//                        builder.append(log.getChar());
//                    }
//                    System.out.println(builder);
                }
            }
            checkNVRTC(compilationStatus);

            checkNVRTC(nvrtcGetPTXSize(program, pp));
            final ByteBuffer PTX = memAlloc((int)pp.get(0));
            checkNVRTC(nvrtcGetPTX(program, PTX));

            //////////////////////

//            PointerBuffer pp = stack.mallocPointer(1);
            IntBuffer     pi = stack.mallocInt(1);

            // initialize
            if (CUDA.isPerThreadDefaultStreamSupported()) {
                Configuration.CUDA_API_PER_THREAD_DEFAULT_STREAM.set(true);
            }

            System.out.format("- Initializing...\n");
            check(cuInit(0));

            check(cuDeviceGetCount(pi));
            if (pi.get(0) == 0) {
                throw new IllegalStateException("Error: no devices supporting CUDA");
            }

            // get first CUDA device
            check(cuDeviceGet(pi, 0));
            int device = pi.get(0);

            // get device name
            ByteBuffer pb = stack.malloc(100);
            check(cuDeviceGetName(pb, device));
            System.out.format("> Using device 0: %s\n", memASCII(memAddress(pb)));

            // get compute capabilities and the device name
//            IntBuffer minor = stack.mallocInt(1);
            check(cuDeviceComputeCapability(pi, minor, device));
            System.out.format("> GPU Device has SM %d.%d compute capability\n", pi.get(0), minor.get(0));

            // get memory size
            check(cuDeviceTotalMem(pp, device));
            System.out.format("  Total amount of global memory:   %d bytes\n", pp.get(0));
            System.out.format("  64-bit Memory Address:           %s\n", (pp.get(0) > 4 * 1024 * 1024 * 1024L) ? "YES" : "NO");

            // create context
            check(cuCtxCreate(pp, 0, device));
            ctx = pp.get(0);

            // load kernel
            check(cuModuleLoadData(pp, PTX));
            long module = pp.get(0);

            check(cuModuleGetFunction(pp, module, KERNEL_NAME));
            function = pp.get(0);
        } catch (final Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    List<OptionInst> options;
    long cudaExpiries;
    long cudaIsCalls;
    long cudaStrikes;
    long cudaVols;
    long cudaRates;

    long cudaNowMs;
    long cudaNumOpts;
    long cudaFwdPx;

    long cudaFairPxOut;

    LongBuffer nowTimeMs;
    DoubleBuffer fwdPxBuffer;
    IntBuffer numOptions;
    DoubleBuffer fairPxOut;


    @Override
    public void loadOptions(List<OptionInst> options, double vol, double rate) {
        this.options = options;
        final int size = options.size();

        LongBuffer expiries = memAllocLong(size);
        ByteBuffer isCalls = memAlloc(size);
        DoubleBuffer strikes = memAllocDouble(size);
        DoubleBuffer vols = memAllocDouble(size);
        DoubleBuffer rates = memAllocDouble(size);
        fairPxOut = memAllocDouble(size);
        nowTimeMs = memAllocLong(1);
        fwdPxBuffer = memAllocDouble(1);
        numOptions = memAllocInt(1);

        numOptions.put(options.size());
        for (final OptionInst inst : options) {
            expiries.put(inst.expiryMs);
            strikes.put(inst.strike);
            isCalls.put((byte) (inst.isCall ? 1 : 0));
            vols.put(vol);
            rates.put(rate);
        }

        check(cuMemAlloc(pp, Long.BYTES * size));
        cudaExpiries = pp.get(0);
        check(cuMemAlloc(pp, Byte.BYTES * size));
        cudaIsCalls = pp.get(0);
        check(cuMemAlloc(pp, Double.BYTES * size));
        cudaStrikes = pp.get(0);
        check(cuMemAlloc(pp, Double.BYTES * size));
        cudaVols = pp.get(0);
        check(cuMemAlloc(pp, Double.BYTES * size));
        cudaRates = pp.get(0);
        check(cuMemAlloc(pp, Double.BYTES * size));
        cudaFairPxOut = pp.get(0);


        check(cuMemAlloc(pp, Long.BYTES * 1));
        cudaNowMs = pp.get(0);
        check(cuMemAlloc(pp, Double.BYTES * 1));
        cudaFwdPx = pp.get(0);
        check(cuMemAlloc(pp, Integer.BYTES * 1));
        cudaNumOpts = pp.get(0);

        check(cuMemcpyHtoD(cudaExpiries, expiries));
        check(cuMemcpyHtoD(cudaIsCalls, isCalls));
        check(cuMemcpyHtoD(cudaStrikes, strikes));
        check(cuMemcpyHtoD(cudaVols, vols));
        check(cuMemcpyHtoD(cudaRates, rates));
        check(cuMemcpyHtoD(cudaNumOpts, numOptions));
    }

    @Override
    public List<Double> price(double fwdPx, long timeMs) {

        nowTimeMs.clear().put(timeMs);
        fwdPxBuffer.clear().put(fwdPx);
        check(cuMemcpyHtoD(cudaNowMs, nowTimeMs));
        check(cuMemcpyHtoD(cudaFwdPx, fwdPxBuffer));

        try (MemoryStack stack = stackPush()) {
            // grid for kernel: <<<N, 1>>>
            int blockSizeX = 256;
            int gridSizeX = (int)Math.ceil((double)options.size() / blockSizeX);
            check(cuLaunchKernel(function,
                    gridSizeX, 1, 1,  // Nx1x1 blocks
                    blockSizeX, 1, 1,            // 1x1x1 threads
                    0, 0,
                    // method 1: unpacked (simple, no alignment requirements)
                    stack.pointers(
                            memAddress(stack.longs(cudaNumOpts)),
                            memAddress(stack.longs(cudaNowMs)),
                            memAddress(stack.longs(cudaExpiries)),
                            memAddress(stack.longs(cudaVols)),
                            memAddress(stack.longs(cudaRates)),
                            memAddress(stack.longs(cudaStrikes)),
                            memAddress(stack.longs(cudaIsCalls)),
                            memAddress(stack.longs(cudaFwdPx)),
                            memAddress(stack.longs(cudaFairPxOut))
                    ),
                    null/*,
                // method 2: packed (user is responsible for correct argument alignment)
                stack.pointers(
                    CU_LAUNCH_PARAM_BUFFER_POINTER, memAddress(stack.longs(
                        deviceA,
                        deviceB,
                        deviceC
                    )),
                    CU_LAUNCH_PARAM_BUFFER_SIZE, memAddress(stack.pointers(3 * Long.BYTES)),
                    CU_LAUNCH_PARAM_END
                )*/));
        }

        List<Double> result = new ArrayList<>(options.size());
        // copy results to host and report
        fairPxOut.clear();
        check(cuMemcpyDtoH(fairPxOut, cudaFairPxOut));
        for (int i = 0; i < options.size(); ++i) {
            result.add(fairPxOut.get());
        }

        // finish
//        check(cuMemFree(deviceA));
//        check(cuMemFree(deviceB));
//        check(cuMemFree(deviceC));
//        check(cuCtxDetach(ctx));

        return result;
    }

    private static void checkNVRTC(int err) {
        if (err != NVRTC_SUCCESS) {
            throw new IllegalStateException(nvrtcGetErrorString(err));
        }
    }

    private static void check(int err) {
        if (err != CUDA_SUCCESS) {
            if (ctx != NULL) {
                cuCtxDetach(ctx);
                ctx = NULL;
            }
            throw new IllegalStateException(Integer.toString(err));
        }
    }
}
