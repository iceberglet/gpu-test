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

    static final OptionPricer cuda = new CudaLWJGLOptionPricer();
    static final OptionPricer java = new JavaOptionPricer();

    private static double[] genFwdPx(int size) {
        final double[] res = new double[size];
        for(int i = 0; i < size; ++i) {
            res[i] = currPrice + 1000 + Math.random() * 5500d;
        }
        return res;
    }

    public static void main(String[] args) throws Exception {

//        Thread.sleep(20_000L);
        cuda.init();
        java.init();
        doOneRound(generateOptions(32), true);
//
//
        doOneRound(generateOptions(1), false);
        doOneRound(generateOptions(2), false);
        doOneRound(generateOptions(4), false);
        doOneRound(generateOptions(8), false);
        doOneRound(generateOptions(16), false);
        doOneRound(generateOptions(32), false);
        doOneRound(generateOptions(64), false);
        doOneRound(generateOptions(96), false);
        doOneRound(generateOptions(128), false);
        doOneRound(generateOptions(160), false);
        doOneRound(generateOptions(192), false);
        doOneRound(generateOptions(256), false);
        doOneRound(generateOptions(384), false);
        doOneRound(generateOptions(512), false);
        doOneRound(generateOptions(640), false);
        doOneRound(generateOptions(768), false);
        doOneRound(generateOptions(896), false);
        doOneRound(generateOptions(1024), false);
    }

    private static void doOneRound(final List<OptionInst> options, final boolean warmup) {
        java.loadOptions(options, 0.34, 0.001);
        cuda.loadOptions(options, 0.34, 0.001);
        final int rounds = 100;
        final long start = System.nanoTime();
        double[] res1 = new double[0];
        final var fwds = genFwdPx(rounds);
        for(int i = 0; i < rounds; ++i) {
            res1 = java.price(fwds[i], nowMs);
        }
        final long javaDone = System.nanoTime();
        double[] res2 = new double[0];
        for(int i = 0; i < rounds; ++i) {
//            res2 = cuda.price(fwds[i], nowMs);
        }
        final long end = System.nanoTime();

//        assert res1.length == res2.length;
//        assert res1.length == options.size();
//        for(int i = 0; i < options.size(); ++i) {
//            final var javaRes = res1[i];
//            final var cudaRes = res2[i];
//            if(javaRes > 0.000000001 && Math.abs((javaRes - cudaRes) / cudaRes) > 0.001d) {
//                System.out.printf("Significant Error! Java %f Cuda %f\n", javaRes, cudaRes);
//            }
//        }

        if(!warmup) {
//            System.out.printf("%d,%d,%d\n", options.size(), (javaDone - start) / rounds, (end - javaDone) / rounds);
            System.out.printf("%d\n", (javaDone - start) / rounds);
//            System.out.printf("%d\n", (end - javaDone) / rounds);

//            if(options.size() == 256 || options.size() == 8) {
//                for(var t : cuda.getTime()){
//                    System.out.println(t);
//                }
//            }


        }
    }

    private static List<OptionInst> generateOptions(int size) {
        List<OptionInst> res = new ArrayList<>();
        int callPutCutoff = Math.max(32, size / 2 / 32 * 32);
        final double priceBase = currPrice - (double) size / 2 * priceStep;
        for(int i = 0; i < size; ++i) {
            res.add(new OptionInst(
//                    Math.random() < 0.5
                    i < callPutCutoff
                    , expiryMs, priceBase + i * priceStep));
        }
        return res;
    }
}
