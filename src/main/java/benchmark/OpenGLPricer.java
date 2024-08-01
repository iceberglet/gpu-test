package benchmark;

import java.util.List;

public class OpenGLPricer implements OptionPricer {

    @Override
    public void init() {

    }

    @Override
    public void loadOptions(List<OptionInst> options, double vol, double rate) {

    }

    @Override
    public List<Double> price(double fwdPx, long timeMs) {
        return List.of();
    }
}
