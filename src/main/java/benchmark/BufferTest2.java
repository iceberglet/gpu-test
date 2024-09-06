package benchmark;

import org.agrona.IoUtil;

import java.io.File;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

import static org.lwjgl.system.MemoryUtil.memAddress;

public class BufferTest2 {

    public static void main(String[] args) throws Exception {

        final var buffer = IoUtil.mapExistingFile(new File("D:\\hello"), FileChannel.MapMode.READ_WRITE, "D:\\hello", 0, 8)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        int i = 98;
        while(true) {
            buffer.put(1, i + 0.2f);
            i++;
            System.out.println(memAddress(buffer) + " " + buffer.get(0) + " " + buffer.get(1));
            Thread.sleep(1000L);
        }
    }
}
