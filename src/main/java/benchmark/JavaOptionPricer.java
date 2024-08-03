package benchmark;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;

public class JavaOptionPricer implements OptionPricer {

    private static final long MS_IN_YEAR = 365 * 24 * 3600 * 1000L;
    private static final RealDistribution STD_NORM = new NormalDistribution(0, 1d);

    private static double getCdfStd(double x) {
        return STD_NORM.cumulativeProbability(x);
    }

    @Override
    public void init() {
        //nothing
    }

    private List<OptionInst> options;
    private double vol;
    private double rate;
    private double[] result;

    @Override
    public void loadOptions(List<OptionInst> options, double vol, double rate) {
        this.options = options;
        this.vol = vol;
        this.rate = rate;
        result = new double[options.size()];
    }

    @Override
    public double[] price(double fwdPx, long timeMs) {
        for(int x = 0; x < result.length; ++x) {
            final var option = options.get(x);
            double fairPx = Double.NaN;
            for(int i = 0; i < 130; ++i) {
                final double tte = (double)(option.expiryMs - timeMs) / MS_IN_YEAR;
                final double scaledVol = vol * FastMath.sqrt(tte);
                final double d1 = FastMath.log(fwdPx / option.strike) / scaledVol + scaledVol / 2;
                final double d2 = d1 - scaledVol;
                final double discount = FastMath.exp(-1 * rate * tte);
                if(option.isCall) {
                    fairPx = discount * (getCdfStd(d1) - getCdfStd(d2) * option.strike / fwdPx);
                } else {
                    fairPx = discount * (getCdfStd(-d2) * option.strike / fwdPx - getCdfStd(-d1));
                }
            }
            result[x] = fairPx;
        }
        return result;
    }
}
