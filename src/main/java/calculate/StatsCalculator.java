package calculate;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.math.BigDecimal;

public class StatsCalculator {
    private BigDecimal range;
    private BigDecimal variance;

    public StatsCalculator(double[] arr) {
        SummaryStatistics stats = new SummaryStatistics();
        for (int i = 0; i < arr.length; i++) {
            stats.addValue(arr[i]);
        }

        BigDecimal count = new BigDecimal(stats.getN());
        BigDecimal mean = new BigDecimal(stats.getMean());

        BigDecimal minValue = new BigDecimal(stats.getMin());
        BigDecimal maxValue = new BigDecimal(stats.getMax());
        range = maxValue.subtract(minValue).abs();

        BigDecimal sumSquares = new BigDecimal(stats.getSumsq());
        variance = sumSquares.divide(count, 10, BigDecimal.ROUND_HALF_UP).subtract(mean.multiply(mean));
    }

    public BigDecimal getRange() {
        return range;
    }

    public BigDecimal getVariance() {
        return variance;
    }

}

