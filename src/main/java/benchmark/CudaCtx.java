package benchmark;

import org.agrona.IoUtil;
import org.lwjgl.PointerBuffer;
import org.lwjgl.cuda.CUDA;
import org.lwjgl.cuda.CUIPCMemHandle;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.lwjgl.cuda.CU.CUDA_SUCCESS;
import static org.lwjgl.cuda.CU.CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED;
import static org.lwjgl.cuda.CU.cuCtxCreate;
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
import static org.lwjgl.system.MemoryStack.stackPop;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memASCII;
import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memAlloc;
import static org.lwjgl.system.MemoryUtil.memAllocFloat;

public class CudaCtx {

    private long ctx;

    private PointerBuffer pp;

    int device;

    public long getCtx() {
        return ctx;
    }

    public PointerBuffer getPp() {
        return pp;
    }

    public int getDevice() {
        return device;
    }

    public CudaCtx() {

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

            final var handleBuf = IoUtil.mapExistingFile(new File("D:\\handleBuf"),  FileChannel.MapMode.READ_WRITE,
                    "D:\\handleBuf", 0, CUIPCMemHandle.SIZEOF);
            final var buf = memAlloc(CUIPCMemHandle.SIZEOF);
            buf.put(handleBuf);
//        for(int i = 0; i < CUIPCMemHandle.SIZEOF; ++i) {
//            System.out.print(buf.get(i));
//            System.out.print(",");
//        }
            final var handleAddr = memAddress(buf);
            final var handle = CUIPCMemHandle.create(handleAddr);
            final var CUdeviceptr = stack.callocPointer(1);
            check(cuIpcOpenMemHandle(CUdeviceptr, handle, 0));
            final long addr = CUdeviceptr.get(0);
            final var fBuf = memAllocFloat(4);
            check(cuMemcpyDtoH(fBuf, addr));
            System.out.println(fBuf.get(0));
        } catch (final Exception ex) {
            throw new RuntimeException(ex);
        }
    }


    private static void checkNVRTC(int err) {
        if (err != NVRTC_SUCCESS) {
            throw new IllegalStateException(nvrtcGetErrorString(err));
        }
    }

    private void check(int err) {
        if (err != CUDA_SUCCESS) {
            if (ctx != NULL) {
                cuCtxDetach(ctx);
                ctx = NULL;
            }
            throw new IllegalStateException(Integer.toString(err));
        }
    }
}
