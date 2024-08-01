package benchmark;

import java.util.List;

public interface OptionPricer {

    void init();

    void loadOptions(List<OptionInst> options, double vol, double rate);

    List<Double> price(double fwdPx, long timeMs);


}
