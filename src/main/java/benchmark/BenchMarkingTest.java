package benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BenchMarkingTest {
    private static final long MS_IN_YEAR = 365 * 24 * 3600L * 1000L;


    private static final double currPrice = 60000;
    private static final double priceStep = 1000;
    private static final long expiryMs = MS_IN_YEAR;
    private static final long nowMs = MS_IN_YEAR / 2;

    static final OptionPricer cuda = new CudaOptionPricer();
    static final OptionPricer java = new JavaOptionPricer();

    private static double[] genFwdPx(int size) {
        final double[] res = new double[size];
        for(int i = 0; i < size; ++i) {
            res[i] = currPrice + 1000 + Math.random() * 5500d;
        }
        return res;
    }

    public static void main(String[] args) {

        cuda.init();
        java.init();
        doOneRound(generateOptions(256), true);
        doOneRound(generateOptions(512), true);
        doOneRound(generateOptions(64), true);
        doOneRound(generateOptions(128), true);


        doOneRound(generateOptions(1), false);
        doOneRound(generateOptions(2), false);
        doOneRound(generateOptions(4), false);
        doOneRound(generateOptions(8), false);
        doOneRound(generateOptions(16), false);
        doOneRound(generateOptions(32), false);
        doOneRound(generateOptions(64), false);
        doOneRound(generateOptions(128), false);
        doOneRound(generateOptions(256), false);
        doOneRound(generateOptions(512), false);
        doOneRound(generateOptions(1024), false);
    }

    private static void doOneRound(final List<OptionInst> options, final boolean warmup) {
        java.loadOptions(options, 0.34, 0.001);
        cuda.loadOptions(options, 0.34, 0.001);
        final int rounds = 3000;
        final long start = System.nanoTime();
        List<Double> res1 = Collections.emptyList();
        final var fwds = genFwdPx(rounds);
        for(int i = 0; i < rounds; ++i) {
            res1 = java.price(fwds[i], nowMs);
        }
        final long javaDone = System.nanoTime();
        List<Double> res2 = Collections.emptyList();
        for(int i = 0; i < rounds; ++i) {
            res2 = cuda.price(fwds[i], nowMs);
        }
        final long end = System.nanoTime();

        assert res1.size() == res2.size();
        assert res1.size() == options.size();
        for(int i = 0; i < options.size(); ++i) {
            final var javaRes = res1.get(i);
            final var cudaRes = res2.get(i);
            if(javaRes > 0.000000001 && Math.abs((javaRes - cudaRes) / cudaRes) > 0.001d) {
                System.out.printf("Significant Error! Java %f Cuda %f\n", javaRes, cudaRes);
            }
        }

        if(!warmup) {
            System.out.printf("------ Option Size %d ------\n", options.size());
            System.out.println("Java (nano per task): " + (javaDone - start) / rounds);
            System.out.println("Cuda (nano per task): " + (end - javaDone) / rounds);
        }
    }

    private static List<OptionInst> generateOptions(int size) {
        List<OptionInst> res = new ArrayList<>();
        final double priceBase = currPrice - (double) size / 2 * priceStep;
        for(int i = 0; i < size; ++i) {
            final var rand = Math.random();
            res.add(new OptionInst(rand > 0.5, expiryMs, priceBase + i * priceStep));
        }
        return res;
    }
}
