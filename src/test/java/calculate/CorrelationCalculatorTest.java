package calculate;

import junit.framework.TestCase;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CorrelationCalculatorTest extends TestCase {

    @Test
    public void testPearsonCorrelation_scalar() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 6, 8, 10};
        double correlation = CorrelationCalculator.calculateCorrelation(x, y, CorrelationTypeEnum.PEARSON);
        assertEquals(1.0, correlation, 0.0001);
    }

    @Test
    public void testPearsonCorrelation_vector() {
        double[][] x = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] y = {{3, 6, 9}, {12, 15, 18}, {21, 24, 27}};
        double correlation = CorrelationCalculator.calculateCorrelation(x, y, CorrelationTypeEnum.PEARSON);
        assertEquals(1.0, correlation, 0.0001);
    }

    @Test
    public void testSpearmanCorrelation_scalar() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 6, 8, 10};
        double correlation = CorrelationCalculator.calculateCorrelation(x, y, CorrelationTypeEnum.SPEARMAN);
        assertEquals(1.0, correlation, 0.0001);
    }

    @Test
    public void testSpearmanCorrelation_vector() {
        double[][] x = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] y = {{3, 6, 9}, {12, 15, 18}, {21, 24, 27}};
        double correlation = CorrelationCalculator.calculateCorrelation(x, y, CorrelationTypeEnum.SPEARMAN);
        assertEquals(1.0, correlation, 0.0001);
    }

    @Test
    public void testKendallCorrelation_scalar() {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2, 4, 6, 8, 10};
        double correlation = CorrelationCalculator.calculateCorrelation(x, y, CorrelationTypeEnum.KENDALL);
        assertEquals(1.0, correlation, 0.0001);
    }

    @Test
    public void testKendallCorrelation_vector() {
        double[][] x = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        double[][] y = {{3, 6, 9}, {12, 15, 18}, {21, 24, 27}};
        double correlation = CorrelationCalculator.calculateCorrelation(x, y, CorrelationTypeEnum.KENDALL);
        assertEquals(1.0, correlation, 0.0001);
    }

}