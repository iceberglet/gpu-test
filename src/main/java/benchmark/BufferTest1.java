package benchmark;

import org.agrona.IoUtil;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memAlloc;

public class BufferTest1 {

    public static void main(String[] args) throws Exception {

        final int size = 2;
        final var buffer = IoUtil.mapExistingFile(new File("D:\\hello"), FileChannel.MapMode.READ_WRITE, "D:\\hello", 0, 8)
//        final var buffer = IoUtil.mapNewFile(new File("D:\\hello"), 8)
                .order(ByteOrder.nativeOrder());
        final var fBuffer = buffer.asFloatBuffer();

//        System.out.println(fBuffer.get(0) + " " + fBuffer.get(1));
        int i = 0;
        while(true) {
//            for(int k = 0; k < size; ++k){
//                fBuffer.put(k, i);
//            }
            fBuffer.put(0, i + 0.2f);
            fBuffer.put(1, i * 2 + 0.2f);
            System.out.println(fBuffer.get(0) + " " + fBuffer.get(1));
            ++i;
            Thread.sleep(1000L);
        }
    }
}
