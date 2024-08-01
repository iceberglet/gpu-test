package benchmark;

public class OptionInst {

    public boolean isCall;
    public long expiryMs;
    public double strike;

    public OptionInst(boolean isCall, long expiryMs, double strike) {
        this.isCall = isCall;
        this.expiryMs = expiryMs;
        this.strike = strike;
    }
}
