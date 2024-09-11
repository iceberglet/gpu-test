package benchmark;

import java.nio.ByteBuffer;

public class OptionLevelInput {

    private static final int strikeOffset;
    private static final int expiryOffset;
    private static final int isCallOffset;
    private static final int volOffset;
    private static final int rateOffset;
    private static final int unitSize;

    static {
        strikeOffset = 0;
        rateOffset = strikeOffset + Float.BYTES;
        volOffset = rateOffset + Float.BYTES;
        expiryOffset = volOffset + Float.BYTES;
        isCallOffset = expiryOffset + Long.BYTES;

        //note: pad to multiple of 4 bytes! CUDA cannot tolerate float point address of odd sizes
        unitSize = isCallOffset + Float.BYTES;
    }

    public static int bytesNeededFor(int size) {
        final int res = size * unitSize;

        if(res % 4 != 0) {
            throw new IllegalArgumentException("CUDA buffer must be multiple of 4");
        }

        return res;
    }

    private final int size;
    private final ByteBuffer buffer;

    public OptionLevelInput(ByteBuffer buffer, int size) {
        this.buffer = buffer;
        this.size = size;
    }

    public void setStrike(int idx, float strike) {
        buffer.putFloat(strikeOffset + idx * unitSize, strike);
    }

    public void setExpiryMs(int idx, long expiryMs) {
        buffer.putLong(expiryOffset + idx * unitSize, expiryMs);
    }

    public void setIsCall(int idx, byte isCall) {
        buffer.put(isCallOffset + idx * unitSize, isCall);
    }

    public void setVol(int idx, float vol) {
        buffer.putFloat(volOffset + idx * unitSize, vol);
    }

    public void setRate(int idx, float rate) {
        buffer.putFloat(rateOffset + idx * unitSize, rate);
    }
}
