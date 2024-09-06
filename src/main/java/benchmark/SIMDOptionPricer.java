//package benchmark;
//
//import jdk.incubator.vector.DoubleVector;
//import jdk.incubator.vector.FloatVector;
//import jdk.incubator.vector.IntVector;
//import jdk.incubator.vector.LongVector;
//import jdk.incubator.vector.Vector;
//import jdk.incubator.vector.VectorSpecies;
//import org.apache.commons.math3.exception.ConvergenceException;
//import org.apache.commons.math3.exception.util.LocalizedFormats;
//import org.apache.commons.math3.util.FastMath;
//import org.apache.commons.math3.util.Precision;
//
//import java.util.Arrays;
//import java.util.List;
//import java.util.stream.Collectors;
//import java.util.stream.Stream;
//
//public class SIMDOptionPricer implements OptionPricer {
//
//    static final VectorSpecies<Long> LONG_VECTOR_SPECIES = LongVector.SPECIES_PREFERRED;
//    static final VectorSpecies<Double> DOUBLE_VECTOR_SPECIES = DoubleVector.SPECIES_PREFERRED;
//    static final VectorSpecies<Float> FLOAT_VECTOR_SPECIES = FloatVector.SPECIES_PREFERRED;
//
//
//    @Override
//    public void init() {
//
//    }
//
//    Vector<Long> expiryMs;
//
//    @Override
//    public void loadOptions(List<OptionInst> options, double vol, double rate) {
//        expiryMs = fromList(options.stream().map(o -> o.expiryMs));
//    }
//
//    Vector<Long> fromList(Stream<Long> longStream) {
//        final var list = longStream.toList();
//        final var arr = new long[list.size()];
//        for(int i = 0; i < arr.length; ++i) {
//            arr[i] = list.get(i);
//        }
//        return LongVector.fromArray(LONG_VECTOR_SPECIES, arr, 0);
//    }
//
//    public static void main(String[] args) {
//        final var a = LongVector.fromArray(LONG_VECTOR_SPECIES, new long[]{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}, 0);
//        final var c = a.add(3L);
//        System.out.println(Arrays.toString(c.toArray()));
////        final var b = a.broadcast(3);
////        //6, 8, 10, 12
////        final var c = a.add(b);
////        System.out.println(Arrays.toString(c.toArray()));
//    }
//
//    @Override
//    public double[] price(double fwdPx, long timeMs) {
//
////        final var tte = expiryMs.sub(expiryMs.broadcast(timeMs)).div()
//
////        final var option = options.get(x);
////        double fairPx = Double.NaN;
////        for(int i = 0; i < 130; ++i) {
////            final double tte = (double)(option.expiryMs - timeMs) / MS_IN_YEAR;
////            final double scaledVol = vol * FastMath.sqrt(tte);
////            final double d1 = FastMath.log(fwdPx / option.strike) / scaledVol + scaledVol / 2;
////            final double d2 = d1 - scaledVol;
////            final double discount = FastMath.exp(-1 * rate * tte);
////            if(option.isCall) {
////                fairPx = discount * (getCdfStd(d1) - getCdfStd(d2) * option.strike / fwdPx);
////            } else {
////                fairPx = discount * (getCdfStd(-d2) * option.strike / fwdPx - getCdfStd(-d1));
////            }
////        }
////        result[x] = fairPx;
//
//
//        return new double[0];
//    }
//
//
//
//
////    private static FloatVector erf(FloatVector x) {
////
////        FloatVector.fromMemorySegment(FLOAT_VECTOR_SPECIES, );
////        x.div(2f);
////    }
////
////    private static FloatVector regGamma(FloatVector x) {
////        //getA = (2.0 * n) + 0.5 + x;
////        //getB = -0.5 * n
////
////        double hPrev = getA(0, x);
////        int n = 1;
////        double dPrev = 0.0;
////        double cPrev = hPrev;
////        double hN = hPrev;
////
////        while (n < 10_000) {
////            final double a = getA(n, x);
////            final double b = getB(n, x);
////
////            double dN = a + b * dPrev;
////            if (Precision.equals(dN, 0.0, small)) {
////                dN = small;
////            }
////            double cN = a + b / cPrev;
////            if (Precision.equals(cN, 0.0, small)) {
////                cN = small;
////            }
////
////            dN = 1 / dN;
////            final double deltaN = cN * dN;
////            hN = hPrev * deltaN;
////
////            if (Double.isInfinite(hN)) {
////                throw new ConvergenceException(LocalizedFormats.CONTINUED_FRACTION_INFINITY_DIVERGENCE,
////                        x);
////            }
////            if (Double.isNaN(hN)) {
////                throw new ConvergenceException(LocalizedFormats.CONTINUED_FRACTION_NAN_DIVERGENCE,
////                        x);
////            }
////
////            if (FastMath.abs(deltaN - 1.0) < epsilon) {
////                break;
////            }
////
////            dPrev = dN;
////            cPrev = cN;
////            hPrev = hN;
////            n++;
////        }
////    }
//}
