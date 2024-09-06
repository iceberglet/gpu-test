package benchmark;

import org.lwjgl.PointerBuffer;
import org.lwjgl.cuda.CUDA;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static benchmark.OptionLevelInput.bytesNeededFor;
import static org.lwjgl.cuda.CU.CUDA_SUCCESS;
import static org.lwjgl.cuda.CU.CU_CTX_SCHED_SPIN;
import static org.lwjgl.cuda.CU.cuCtxCreate;
import static org.lwjgl.cuda.CU.cuCtxDetach;
import static org.lwjgl.cuda.CU.cuDeviceGet;
import static org.lwjgl.cuda.CU.cuDeviceGetCount;
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
import static org.lwjgl.system.MemoryUtil.memAllocFloat;
import static org.lwjgl.system.MemoryUtil.memFree;


public class CudaLWJGLOptionPricer implements OptionPricer {

    private static final String KERNEL_NAME = "fairPx";

    private static long ctx;

    private PointerBuffer pp;

    private long function;

    @Override
    public void init() {
        try (MemoryStack stack = stackPush()) {
            //allocate 2 integer buffer to read from nvrtc
            IntBuffer major = stack.mallocInt(1);
            IntBuffer minor = stack.mallocInt(1);

            checkNVRTC(nvrtcVersion(major, minor));

            System.out.println("Compiling kernel with NVRTC v" + major.get(0) + "." + minor.get(0));

            //allocate a main memory pointer address
            pp = stack.mallocPointer(1);

            //read cu file content
            final String cu = Files.readString(Path.of("src/main/resources/kernels/OptionPricingKernel.cu"), StandardCharsets.UTF_8);

            //create a cu program, now pp contains pointer address to the program, read it
            checkNVRTC(nvrtcCreateProgram(pp, cu, "OptionPricingKernel.cu", null, null));
            long program = pp.get(0);

            //invoke nvrtc to compile cu file into ptx content
            int compilationStatus = nvrtcCompileProgram(program, null);

            //check compilation results
            checkNVRTC(nvrtcGetProgramLogSize(program, pp));
            if (1L < pp.get(0)) {
                ByteBuffer log = stack.malloc((int)pp.get(0) - 1);

                checkNVRTC(nvrtcGetProgramLog(program, log));
                System.err.println("Compilation log:");
                System.err.println("----------------");
                System.err.println(memASCII(log));
            }
            checkNVRTC(compilationStatus);

            //load ptx results
            checkNVRTC(nvrtcGetPTXSize(program, pp));
            final ByteBuffer PTX = memAlloc((int)pp.get(0));
            checkNVRTC(nvrtcGetPTX(program, PTX));

            // initialize CUDA device
            IntBuffer pi = stack.mallocInt(1);
            if (CUDA.isPerThreadDefaultStreamSupported()) {
                Configuration.CUDA_API_PER_THREAD_DEFAULT_STREAM.set(true);
            }
            check(cuInit(0));
            check(cuDeviceGetCount(pi));
            if (pi.get(0) == 0) {
                throw new IllegalStateException("Error: no devices supporting CUDA");
            }

            // get first CUDA device
            check(cuDeviceGet(pi, 0));
            int device = pi.get(0);

//            // get device name
//            ByteBuffer pb = stack.malloc(100);
//            check(cuDeviceGetName(pb, device));
//            System.out.format("> Using device 0: %s\n", memASCII(memAddress(pb)));
//
//            // get compute capabilities and the device name
//            check(cuDeviceComputeCapability(pi, minor, device));
//            System.out.format("> GPU Device has SM %d.%d compute capability\n", pi.get(0), minor.get(0));
//
//            // get memory size
//            check(cuDeviceTotalMem(pp, device));
//            System.out.format("  Total amount of global memory:   %d bytes\n", pp.get(0));
//            System.out.format("  64-bit Memory Address:           %s\n", (pp.get(0) > 4 * 1024 * 1024 * 1024L) ? "YES" : "NO");

            // create context
            check(cuCtxCreate(pp, CU_CTX_SCHED_SPIN, device));
            ctx = pp.get(0);

            // prepare kernel with compiled ptx data
            check(cuModuleLoadData(pp, PTX));
            long module = pp.get(0);

            check(cuModuleGetFunction(pp, module, KERNEL_NAME));
            function = pp.get(0);
        } catch (final Exception ex) {
            throw new RuntimeException(ex);
        }
    }

    List<OptionInst> options;
    long cudaOptInputs;
    long cudaFairPxOut;
    double[] result;
    ByteBuffer optInputs;
    FloatBuffer fairPxOut;

    @Override
    public void loadOptions(List<OptionInst> options, double vol, double rate) {
        this.options = options;
        final int size = options.size();
        result = new double[size];

        optInputs = memAlloc(bytesNeededFor(size));
        fairPxOut = memAllocFloat(size);

        OptionLevelInput input = new OptionLevelInput(optInputs, size);

        for (int i = 0; i < options.size(); ++i) {
            final var inst = options.get(i);
            input.setExpiryMs(i, inst.expiryMs);
            input.setStrike(i, (float)inst.strike);
            input.setIsCall(i, (byte) (inst.isCall ? 1 : 0));
            input.setVol(i, (float)vol);
            input.setRate(i, (float)rate);
        }

        check(cuMemAlloc(pp, bytesNeededFor(size)));
        cudaOptInputs = pp.get(0);
        check(cuMemAlloc(pp, Float.BYTES * size));
        cudaFairPxOut = pp.get(0);
    }

    @Override
    public void clear() {
        memFree(fairPxOut);
        cuMemFree(cudaFairPxOut);
    }

    @Override
    public double[] price(double fwdPx, long timeMs) {

        //NOTE: this is just for illustration on the latency for option level inputs
        //and to illustrate we can cramp different data type into the same level of inputs
        //in actual impl, we probably will separate runtime and realtime option level inputs
        //for expiry it should be entirely runtime inputs only
        check(cuMemcpyHtoD(cudaOptInputs, optInputs));

        try (MemoryStack stack = stackPush()) {
            // grid for kernel: <<<N, 1>>>
            // block size is ideally multiples of 32 (a warp). Here we use fewer so more SM can be used
            int blockSizeX = 32;
            int gridSizeX = (int)Math.ceil((double)options.size() / blockSizeX);
            check(cuLaunchKernel(function,
                    gridSizeX, 1, 1,  // Nx1x1 blocks
                    blockSizeX, 1, 1, // 1x1x1 threads
                    0, 0,
                    // method 1: unpacked (simple, no alignment requirements)
                    stack.pointers(
                            memAddress(stack.ints(options.size())),
                            memAddress(stack.longs(timeMs)),
                            memAddress(stack.floats((float) fwdPx)),
                            memAddress(stack.longs(cudaOptInputs)),
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

        // copy results to host and report
        fairPxOut.clear();
        check(cuMemcpyDtoH(fairPxOut, cudaFairPxOut));
        for (int i = 0; i < options.size(); ++i) {
            result[i] = fairPxOut.get();
        }

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
