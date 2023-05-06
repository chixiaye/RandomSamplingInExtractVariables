package calculate;

import io.excel.ExcelData;
import io.excel.ExcelReader;
import io.json.EVRecordReader;
import json.EVRecord;
import json.LayoutRelationData;
import json.MetaData;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import sample.Constants;
import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.util.*;

import static calculate.Utils.*;

@Slf4j
public class Correlation {
    ExcelReader excelReader;
    @Getter
    private List<EVRecord> posRecords;
    @Getter
    private List<EVRecord> negRecords;

    public Correlation(ExcelReader excelReader) {
        this.excelReader = excelReader;
        posRecords = new ArrayList<>();
        negRecords = new ArrayList<>();
        try {
            loadInitialData(Constants.POSITIVE_OUTPUT_PATH, Constants.NEGATIVE_OUTPUT_PATH);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadInitialData(String positiveOutputPath, String negativeOutputPath) throws IOException {
        for (int i = 0; i <= excelReader.getExcelDataList().size() - 1; ++i) {
            ExcelData v = excelReader.getExcelDataList().get(i);
            String localName = v.getAccount() + "@" + v.getRepository();
            ArrayList<File> positiveArrayList = new ArrayList<>();
            utils.Utils.getFileList(positiveArrayList, positiveOutputPath
                    + localName, "json");
            ArrayList<File> negtiveArrayList = new ArrayList<>();
            Utils.getFileList(negtiveArrayList, negativeOutputPath
                    + localName, "json");
            for (File f : positiveArrayList) {
                List<EVRecord> evRecordList = EVRecordReader.deserializeAsEVRecordList(f.getAbsolutePath());
                Iterator<EVRecord> it = evRecordList.iterator();
                while (it.hasNext()) {
                    EVRecord evRecord = it.next();
                    evRecord.setOccurrences(evRecord.getOccurrences() - 1);
                }
                posRecords.addAll(evRecordList);
            }
            for (File f : negtiveArrayList) {
                List<EVRecord> evRecordList = EVRecordReader.deserializeAsEVRecordList(f.getAbsolutePath());
                negRecords.addAll(evRecordList);
            }
        }
        System.out.println("posRecords.size() = " + posRecords.size());
        System.out.println("negRecords.size() = " + negRecords.size());
    }

    public static void getCorrelation(List<EVRecord> posRecords, List<EVRecord> negRecords, int size, String metric, Class clazz) throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        List<Double> xList = new ArrayList<>();
        List<Integer> yList = new ArrayList<>();
//        Arrays.stream(clazz.getMethods()).toList().forEach(v -> System.out.println(v.getName()));
        Method method = clazz == LayoutRelationData.class ? null : clazz.getMethod("get" + metric);
        for (int j = 0; j < size; ++j) {
            Object invoke = null;
            Object invoke1 = null;
            if (clazz == EVRecord.class) {
                invoke = method.invoke(posRecords.get(j));
                invoke1 = method.invoke(negRecords.get(j));
            } else if (clazz == NodePosition.class) {
                invoke = method.invoke(posRecords.get(j).getExpressionList().get(0).getNodePosition());
                invoke1 = method.invoke(negRecords.get(j).getExpressionList().get(0).getNodePosition());
            } else if (clazz == MetaData.class) {
                invoke = method.invoke(posRecords.get(j).getExpressionList().get(0));
                invoke1 = method.invoke(negRecords.get(j).getExpressionList().get(0));
            } else if (clazz == LayoutRelationData.class) {
                invoke = getStatus(posRecords.get(j).getLayoutRelationDataList(), metric, true);
                invoke1 = getStatus(negRecords.get(j).getLayoutRelationDataList(), metric, false);
            }
            xList.add(Double.valueOf(invoke.toString()));
            xList.add(Double.valueOf(invoke1.toString()));
            yList.add(1);
            yList.add(0);
        }
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("metric: " + metric + System.lineSeparator());
        for (CorrelationTypeEnum e : CorrelationTypeEnum.values()) {
            double v = CorrelationCalculator.calculateCorrelation(xList.stream().mapToDouble(i -> i).toArray(),
                    yList.stream().mapToDouble(i -> i).toArray(), e);
            stringBuilder.append(e.name() + ": " + String.format("%.3f", v).toString() + System.lineSeparator());
        }
        stringBuilder.append(System.lineSeparator());
        log.info(stringBuilder.toString());

    }

    //RQ1: 是否与表达式的复杂度有关
    public void calculate4RQ1() throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        int size = Math.min(negRecords.size(), posRecords.size());
        log.info("size: {}", size);

//        Map<Integer, Integer> map = getOccurrenceDistribution(negRecords);
//        for (Map.Entry<Integer, Integer> e:map.entrySet()){
//            System.out.println(e.getKey()+", "+ e.getValue());
//        }
//        EVRecord.class


        String s1 = "astNodeNumber";
        getCorrelation(posRecords, negRecords, size,
                preProcess(s1), MetaData.class);

        String s2 = "tokenLength";
        getCorrelation(posRecords, negRecords, size,
                preProcess(s2), MetaData.class);

        String s3 = "astHeight";
        getCorrelation(posRecords, negRecords, size,
                preProcess(s3), MetaData.class);

        String s4 = "charLength";
        getCorrelation(posRecords, negRecords, size,
                preProcess(s4), NodePosition.class);
    }

    public void calculate4RQ2() throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        int size = Math.min(negRecords.size(), posRecords.size());
        log.info("size: {}", size);

        String s1 = "Occurrences";
        getCorrelation(posRecords, negRecords, size,
                preProcess(s1), EVRecord.class);

        Map<Integer, Integer> map = getOccurrenceDistribution(posRecords);
        for (Map.Entry<Integer, Integer> e : map.entrySet()) {
            System.out.println(e.getKey() + ", " + e.getValue());
        }

    }


    public void calculate4RQ3() throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        int size = Math.min(negRecords.size(), posRecords.size());
        log.info("size: {}", size);

        Map<String, int[]> map = getNodeTypeOccurrences(posRecords, negRecords, size);
//        map.forEach((k,v) ->System.out.println(k+": "+ v[0]+", "+ v[1]));
        List<Map.Entry<String, int[]>> list = new ArrayList<>(map.entrySet());

        // 对List中的键值对按值排序
        Collections.sort(list, (o1, o2) -> o1.getValue()[0] + o1.getValue()[1] - o2.getValue()[0] - o2.getValue()[1]);

        // 输出排序后的键值对
        for (Map.Entry<String, int[]> entry : list) {
            System.out.println(entry.getKey() + ": " + entry.getValue()[0]);
        }
        System.out.println();
        for (Map.Entry<String, int[]> entry : list) {
            System.out.println(entry.getKey() + ": " + entry.getValue()[1]);
        }
    }

    private static double getStatus(ArrayList<LayoutRelationData> layoutRelationDataArrayList, String s, boolean flag) {
        double[] arr = new double[layoutRelationDataArrayList.size()];
        int i = 0;
        for (LayoutRelationData data : layoutRelationDataArrayList) {
            if (flag && (data.getFirstKey() == 0 || data.getSecondKey() == 0)) {
                continue;
            }
            arr[i++] = data.getLayout();
        }
        StatsCalculator statsCalculator = new StatsCalculator(arr);
        BigDecimal res = s.equals("Range") ? statsCalculator.getRange() : statsCalculator.getVariance();
        return res.doubleValue();
    }
}
