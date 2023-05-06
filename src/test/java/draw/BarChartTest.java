package draw;

import org.junit.Test;

import java.awt.Color;
import java.io.File;
import java.io.IOException;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PiePlot;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.general.PieDataset;

import static org.jfree.chart.ChartUtils.saveChartAsPNG;

public class BarChartTest {
    @Test
    public void barChartTest() {
        // create dataset
        DefaultPieDataset dataset = new DefaultPieDataset();
        dataset.setValue("1", 3357);
        dataset.setValue("2", 8743);
        dataset.setValue("3", 2760);
        dataset.setValue("4", 1077);
        dataset.setValue("5", 772);
        dataset.setValue("6", 327);
        dataset.setValue("7", 182);
        dataset.setValue("8", 153);
        dataset.setValue("9", 75);
        dataset.setValue("10", 44);

        // create chart
        JFreeChart chart = ChartFactory.createPieChart(
                "Pie Chart Example", // chart title
                dataset, // dataset
                true, // legend
                true, // tooltips
                false // urls
        );

        // set colors
        PiePlot plot = (PiePlot) chart.getPlot();
        plot.setSectionPaint("1", new Color(0, 128, 0));
        plot.setSectionPaint("2", new Color(255, 128, 0));
        plot.setSectionPaint("3", new Color(0, 0, 128));
        plot.setSectionPaint("4", new Color(255, 0, 0));
        plot.setSectionPaint("5", new Color(128, 0, 128));
        plot.setSectionPaint("6", new Color(0, 128, 128));
        plot.setSectionPaint("7", new Color(128, 128, 0));
        plot.setSectionPaint("8", new Color(0, 255, 0));
        plot.setSectionPaint("9", new Color(255, 0, 255));
        plot.setSectionPaint("10", new Color(128, 128, 128));

        // save chart as PNG image
        File file = new File("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\pie-chart.png");
        try {
            saveChartAsPNG(file, chart, 400, 300);
            System.out.println("Chart saved as PNG image.");
        } catch (IOException e) {
            System.err.println("Error saving chart as PNG image: " + e.getMessage());
        }
    }

    private static PieDataset createDataset() {
        DefaultPieDataset dataset = new DefaultPieDataset();
        dataset.setValue("1", 3357);
        dataset.setValue("2", 8743);
        dataset.setValue("3", 2760);
        dataset.setValue("4", 1077);
        dataset.setValue("5", 772);
        dataset.setValue("6", 327);
        dataset.setValue("7", 182);
        dataset.setValue("8", 153);
        dataset.setValue("9", 75);
        dataset.setValue("10", 44);
        return dataset;
    }
}