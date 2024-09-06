package benchmark;

import org.agrona.IoUtil;
import org.lwjgl.PointerBuffer;
import org.lwjgl.cuda.CUDA;
import org.lwjgl.cuda.CUIPCMemHandle;
import org.lwjgl.cuda.CUctxCreateParams;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.lwjgl.cuda.CU.CUDA_SUCCESS;
import static org.lwjgl.cuda.CU.CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED;
import static org.lwjgl.cuda.CU.cuCtxCreate;
import static org.lwjgl.cuda.CU.cuCtxCreate_v4;
import static org.lwjgl.cuda.CU.cuCtxDetach;
import static org.lwjgl.cuda.CU.cuDeviceGet;
import static org.lwjgl.cuda.CU.cuDeviceGetAttribute;
import static org.lwjgl.cuda.CU.cuDeviceGetCount;
import static org.lwjgl.cuda.CU.cuInit;
import static org.lwjgl.cuda.CU.cuIpcGetMemHandle;
import static org.lwjgl.cuda.CU.cuIpcOpenMemHandle;
import static org.lwjgl.cuda.CU.cuLaunchKernel;
import static org.lwjgl.cuda.CU.cuMemAlloc;
import static org.lwjgl.cuda.CU.cuMemcpyDtoH;
import static org.lwjgl.cuda.CU.cuMemcpyDtoHAsync;
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
import static org.lwjgl.system.MemoryUtil.memAllocInt;


public class Test2 {

    private static final String KERNEL_NAME = "fairPx";

    private static long ctx;

    private PointerBuffer pp;

    private long function;

    long cudaInput;
    long cudaOutput;
    FloatBuffer output;
    int device;

    public static void main(String[] args) throws Exception {
        new Test2().init();
    }

    public void init() throws Exception {
        try (MemoryStack stack = stackPush()) {
            //allocate 2 integer buffer to read from nvrtc
            IntBuffer major = stack.mallocInt(1);
            IntBuffer minor = stack.mallocInt(1);

            checkNVRTC(nvrtcVersion(major, minor));

            System.out.println("Compiling kernel with NVRTC v" + major.get(0) + "." + minor.get(0));

            //allocate a main memory pointer address
            pp = stack.mallocPointer(1);

            //read cu file content
            final String cu = Files.readString(Path.of("src/main/resources/kernels/SimpleKernel.cu"), StandardCharsets.UTF_8);

            //create a cu program, now pp contains pointer address to the program, read it
            checkNVRTC(nvrtcCreateProgram(pp, cu, "OptionPricingKernel.cu", null, null));
            long program = pp.get(0);

            //invoke nvcc to compile cu file into ptx content
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
            device = pi.get(0);

            // create context
            check(cuCtxCreate(pp, 0, device));
            ctx = pp.get(0);

            // prepare kernel with compiled ptx data
            check(cuModuleLoadData(pp, PTX));
            long module = pp.get(0);

            check(cuModuleGetFunction(pp, module, KERNEL_NAME));
            function = pp.get(0);
        } catch (final Exception ex) {
            throw new RuntimeException(ex);
        }

        int size = 2;
        output = memAllocFloat(size);

        check(cuMemAlloc(pp, size * Float.BYTES));
        cudaInput = pp.get(0);
        check(cuMemAlloc(pp, size * Float.BYTES));
        cudaOutput = pp.get(0);

        final MappedByteBuffer buf = IoUtil.mapExistingFile(new File("D:\\hello"),
                FileChannel.MapMode.READ_ONLY, "D:\\hello", 0, size * 4);

        final var buffer = buf.order(ByteOrder.nativeOrder()).asFloatBuffer();

        check(cuMemcpyHtoD(cudaInput, buffer));
//        System.out.println(memAddress(buffer) + " " + buffer.get(0) + " " + buffer.get(1));
//        try (MemoryStack stack = stackPush()) {
//            // grid for kernel: <<<N, 1>>>
//            // block size is ideally multiples of 32 (a warp). Here we use fewer so more SM can be used
//            int blockSizeX = 32;
//            int gridSizeX = 1;
//            check(cuLaunchKernel(function,
//                    gridSizeX, 1, 1,  // Nx1x1 blocks
//                    blockSizeX, 1, 1, // 1x1x1 threads
//                    0, 0,
//                    // method 1: unpacked (simple, no alignment requirements)
//                    stack.pointers(
//                            memAddress(stack.ints(1)),
//                            memAddress(stack.longs(cudaInput)),
//                            memAddress(stack.longs(cudaOutput))
//                    ),
//                    null));
//        }

        IntBuffer flagBuf = memAllocInt(1);
        cuDeviceGetAttribute(flagBuf, CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED, device);
        System.out.println(flagBuf.get(0)); //prints "1"

        final int bufSz = CUIPCMemHandle.SIZEOF;
        final var handleBuf = IoUtil.mapExistingFile(new File("D:\\handleBuf"),  FileChannel.MapMode.READ_WRITE, "D:\\handleBuf", 0, bufSz);
//        final var handleBuf = IoUtil.mapNewFile(new File("D:\\handleBuf"),  bufSz, true);


        for(int i = 0; i < CUIPCMemHandle.SIZEOF; ++i) {
            System.out.print(handleBuf.get(i));
            System.out.print(",");
        }
        System.out.println();


        var handle = CUIPCMemHandle.create(memAddress(handleBuf));
        check(cuIpcGetMemHandle(handle, cudaInput));

        for(int i = 0; i < CUIPCMemHandle.SIZEOF; ++i) {
            System.out.print(handleBuf.get(i));
            System.out.print(",");
        }
        System.out.println();
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
