package benchmark;

import org.agrona.IoUtil;
import org.lwjgl.cuda.CUIPCMemHandle;
import org.lwjgl.system.MemoryStack;

import java.io.File;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

import static org.lwjgl.cuda.CU.CUDA_SUCCESS;
import static org.lwjgl.cuda.CU.cuCtxDetach;
import static org.lwjgl.cuda.CU.cuIpcOpenMemHandle;
import static org.lwjgl.cuda.CU.cuMemcpyDtoH;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memAlloc;
import static org.lwjgl.system.MemoryUtil.memAllocFloat;

public class SideKick {

    static CudaCtx context;

    public static void main(String[] args) {
        context = new CudaCtx();
    }


    private static void check(int err) {
        if (err != CUDA_SUCCESS) {
            if (context.getCtx() != NULL) {
                cuCtxDetach(context.getCtx());
            }
            throw new IllegalStateException(Integer.toString(err));
        }
    }
}
