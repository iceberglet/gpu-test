package benchmark;

import java.nio.ByteBuffer;

public class OptionLevelInput {

    private final int size;
    private final ByteBuffer buffer;
    private final int strikeOffset;
    private final int expiryOffset;
    private final int isCallOffset;
    private final int volOffset;
    private final int rateOffset;

    public static int bytesNeededFor(int size) {
        final int res = size * (Float.BYTES * 3 + Byte.BYTES + Long.BYTES);

        if(res % 4 != 0) {
            throw new IllegalArgumentException("CUDA buffer must be multiple of 4");
        }

        return res;
    }

    public OptionLevelInput(ByteBuffer buffer, int size) {
        this.buffer = buffer;
        this.size = size;
        strikeOffset = 0;
        expiryOffset = strikeOffset + size * Float.BYTES;
        isCallOffset = expiryOffset + size * Long.BYTES;
        volOffset = isCallOffset + size * Byte.BYTES;
        rateOffset = volOffset + size * Float.BYTES;
    }

    public void setStrike(int idx, float strike) {
        buffer.putFloat(strikeOffset + idx * Float.BYTES, strike);
    }

    public void setExpiryMs(int idx, long expiryMs) {
        buffer.putLong(expiryOffset + idx * Long.BYTES, expiryMs);
    }

    public void setIsCall(int idx, byte isCall) {
        buffer.put(isCallOffset + idx * Byte.BYTES, isCall);
    }

    public void setVol(int idx, float vol) {
        buffer.putFloat(volOffset + idx * Float.BYTES, vol);
    }

    public void setRate(int idx, float rate) {
        buffer.putFloat(rateOffset + idx * Float.BYTES, rate);
    }
}
