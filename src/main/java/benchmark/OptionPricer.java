package benchmark;

import java.util.List;

public interface OptionPricer {

    void init();

    void loadOptions(List<OptionInst> options, double vol, double rate);

    double[] price(double fwdPx, long timeMs);

}
