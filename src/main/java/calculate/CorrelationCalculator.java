package calculate;

import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.util.ArithmeticUtils;

import java.math.BigDecimal;
import java.math.MathContext;


public class CorrelationCalculator {

    /*
    SPEARMAN, PEARSON , KENDALL 这三个有什么区别
    Spearman相关系数：Spearman相关系数是一种用于衡量两个变量之间的单调关系的方法。它基于每个变量的秩次（排名），而不是实际的数值。Spearman相关系数适用于任何数据类型，包括连续型、顺序型和分类型数据。
    Pearson相关系数：Pearson相关系数是一种用于衡量两个变量之间线性关系的方法。它基于每个变量的实际数值，而不是秩次。Pearson相关系数适用于连续型数据。
    Kendall相关系数：Kendall相关系数是一种用于衡量两个变量之间的排序一致性的方法。它也基于每个变量的秩次，而不是实际的数值。Kendall相关系数适用于顺序型数据。

    Point-Biserial Correlation coefficient 适用于当一个变量为连续型而另一个变量是二元变量的情况。
    Phi coefficient是一种用于衡量两个二元变量之间的相关性的方法，它的取值范围也是-1到1，其中0表示没有相关性，1表示完全的正相关性，-1表示完全的负相关性。
     */

    public static double calculateCorrelation(double[] x, double[] y, CorrelationTypeEnum correlationType) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Input vectors must have the same length.");
        }
        if (correlationType == CorrelationTypeEnum.PEARSON
//                || correlationType== CorrelationTypeEnum.POINT_BISERIAL
        ) {
            return new PearsonsCorrelation().correlation(x, y);
        } else if (correlationType == CorrelationTypeEnum.SPEARMAN) {
            return new SpearmansCorrelation().correlation(x, y);
        } else if (correlationType == CorrelationTypeEnum.KENDALL) {
            return new KendallsCorrelation().correlation(x, y);
        }
//        else if(correlationType== CorrelationTypeEnum.PHI){
//            // 将x和y转换为矩阵
//            Matrix X = new Matrix(x, x.length);
//            Matrix Y = new Matrix(y, y.length);
////          phi = (n11 * n00 - n10 * n01) / sqrt(n1 * n0 * (n1 + n0) * (n1 + n0))
//            // 计算Phi coefficient
//            double phi = X.transpose().times(Y).get(0, 0) / Math.sqrt(X.transpose().times(X).get(0, 0) * Y.transpose().times(Y).get(0, 0));
//            return calculatePhiCoefficient(x, y);
//        }
        else {
            throw new IllegalArgumentException("Unsupported correlation type: " + correlationType);
        }
    }

    public static double calculateCorrelation(double[][] x, double[][] y, CorrelationTypeEnum correlationType) {
        if (x.length != y.length || x[0].length != y[0].length) {
            throw new IllegalArgumentException("Input vectors must have the same length.");
        }
        int vectorCount = x.length;
        int vectorLength = x[0].length;
        double[] x1D = new double[vectorCount * vectorLength];
        double[] y1D = new double[vectorCount * vectorLength];
        for (int i = 0; i < vectorCount; i++) {
            for (int j = 0; j < vectorLength; j++) {
                x1D[i * vectorLength + j] = x[i][j];
                y1D[i * vectorLength + j] = y[i][j];
            }
        }
        if (correlationType == CorrelationTypeEnum.PEARSON) {
            return new PearsonsCorrelation().correlation(x1D, y1D);
        } else if (correlationType == CorrelationTypeEnum.SPEARMAN) {
            return new SpearmansCorrelation().correlation(x1D, y1D);
        } else if (correlationType == CorrelationTypeEnum.KENDALL) {
            return new KendallsCorrelation().correlation(x1D, y1D);
        } else {
            throw new IllegalArgumentException("Unsupported correlation type: " + correlationType);
        }
    }


    public static double calculatePhiCoefficient(double[] continuousVar, double[] binaryVar) {
        MathContext mc = new MathContext(20);
        int a = 0, b = 0, c = 0, d = 0;
        for (int i = 0; i < continuousVar.length; i++) {
            if (continuousVar[i] > 0 && binaryVar[i] > 0) {
                a++;
            } else if (continuousVar[i] > 0 && binaryVar[i] <= 0) {
                b++;
            } else if (continuousVar[i] <= 0 && binaryVar[i] > 0) {
                c++;
            } else {
                d++;
            }
        }
        BigDecimal denominator = BigDecimal.valueOf(ArithmeticUtils.mulAndCheck(a + b, c + d))
                .multiply(BigDecimal.valueOf(ArithmeticUtils.mulAndCheck(a + c, b + d)))
                .sqrt(mc);

        if (denominator.compareTo(BigDecimal.ZERO) == 0) {
            return 0;
        }

        BigDecimal numerator = BigDecimal.valueOf(ArithmeticUtils.mulAndCheck(a, d))
                .subtract(BigDecimal.valueOf(ArithmeticUtils.mulAndCheck(b, c)));

        return numerator.divide(denominator, 10, BigDecimal.ROUND_HALF_UP).doubleValue();
    }
}
