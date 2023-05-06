package calculate;

import json.EVRecord;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Utils {
    public static double calculatePrecision(int[] predicted, int[] actual) {
        int truePositives = 0;
        int falsePositives = 0;

        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == 1 && actual[i] == 1) {
                truePositives++;
            } else if (predicted[i] == 1 && actual[i] == 0) {
                falsePositives++;
            }
        }

        if (truePositives + falsePositives == 0) {
            return 0.0;
        }

        return (double) truePositives / (truePositives + falsePositives);
    }

    public static double calculateRecall(int[] predicted, int[] actual) {
        int truePositives = 0;
        int falseNegatives = 0;

        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == 1 && actual[i] == 1) {
                truePositives++;
            } else if (predicted[i] == 0 && actual[i] == 1) {
                falseNegatives++;
            }
        }

        if (truePositives + falseNegatives == 0) {
            return 0.0;
        }

        return (double) truePositives / (truePositives + falseNegatives);
    }


    public static String preProcess(String s) {
        return s.substring(0, 1).toUpperCase() + s.substring(1);
    }

    public static Map<String, int[]> getNodeTypeOccurrences(List<EVRecord> posList, List<EVRecord> negList, int size) {
        Map<String, int[]> occurrences = new HashMap<>();

        // 统计正样本中每个字符串出现的次数
        for (EVRecord record : posList) {
            String nodeType = record.getExpressionList().get(0).getNodeType();
            if (!occurrences.containsKey(nodeType)) {
                occurrences.put(nodeType, new int[]{1, 0}); // 该字符串在正样本出现一次，负样本出现0次
            } else {
                occurrences.get(nodeType)[0]++; // 该字符串在正样本中出现次数加1
            }
        }

        // 统计负样本中每个字符串出现的次数
        for (EVRecord record : negList) {
            String nodeType = record.getExpressionList().get(0).getNodeType();
            if (!occurrences.containsKey(nodeType)) {
                occurrences.put(nodeType, new int[]{0, 1}); // 该字符串在正样本出现0次，负样本出现1次
            } else {
                occurrences.get(nodeType)[1]++; // 该字符串在负样本中出现次数加1
            }
        }

        return occurrences;
    }

    public static Map<Integer, Integer> getOccurrenceDistribution(List<EVRecord> posList) {
        Map<Integer, Integer> occurrences = new TreeMap<>();
        for (EVRecord record : posList) {
            int key = record.getOccurrences();
//            if(key>100){
//                System.out.println(record.getExpression());
//            }
            occurrences.put(key, 1 + occurrences.getOrDefault(key, 0));
        }
        return occurrences;
    }
}
