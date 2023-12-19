package sample;


import io.json.JSONReader;
import json.LabelData;
import lombok.extern.slf4j.Slf4j;
import miner.NegativeMinerThread;
import miner.PositiveMinerThread;
import miner.RatioMinerThread;
import refactoringminer.handler.RefactoringMinerThread;
import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertTrue;

@Slf4j
public class Main {

    static final int CORE_POOL_SIZE = Runtime.getRuntime().availableProcessors() / 5;
    static final int MAX_POOL_SIZE = CORE_POOL_SIZE + 1;

    public static void main(String[] args) throws Exception {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(CORE_POOL_SIZE, MAX_POOL_SIZE,
                5L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
        log.info("core pool size: {}, max pool size: {}", CORE_POOL_SIZE, MAX_POOL_SIZE);
//        getRatio(executor);
//        minePositive(executor);
        mineNegative(executor, Constants.RATIO);
//        doRefactoringMiner(executor);
    }


    // 根据正样本的表达式信息定位到方法， 查找方法内有多少个表达式可以被提取
    public static void getRatio(ThreadPoolExecutor executor) throws IOException {
        ArrayList<File> list = new ArrayList<>();
        Utils.getFileList(list, Constants.LABELED_DATA_PATH, "json");
        StringBuffer sb = new StringBuffer();
        //spring-projects@spring-framework-14
        int sum = 0;
        for (int i = list.size() - 1; i >= 0; --i) {
            File file = list.get(i);

            String fileName = file.getName();
            String localName = fileName.substring(0, fileName.lastIndexOf("_"));
            int sampleNumber = Integer.parseInt(fileName.substring(fileName.lastIndexOf("_") + 1, fileName.lastIndexOf(".")));
            LabelData labelData = JSONReader.deserializeAsLabelData(file.getAbsolutePath());
//            executor.execute( );
            if (localName.equals("spring-projects@spring-framework") && labelData.getId() == 14) {
//                System.out.println("spring-projects@spring-framework-14");
            } else {
//                continue;
            }
            //单线程
            RatioMinerThread thread = new RatioMinerThread(localName, 1, labelData);
            thread.run();
            sum += thread.getVisitor().getCount();
            sb.append(localName + "-" + labelData.getId()).append(",").append(thread.getVisitor().getCount()).append("\n");

        }
        System.out.println(sb.toString());
        System.out.println(sum / list.size());
        executor.shutdown();
    }


    public static void minePositive(ThreadPoolExecutor executor) throws IOException {
        ArrayList<File> list = new ArrayList<>();
        Utils.getFileList(list, Constants.LABELED_DATA_PATH, "json");
        for (int i = list.size() - 1; i >= 0; --i) {
            File file = list.get(i);
            String fileName = file.getName();
            if (!file.getName().contains("bumptech@glide_37")) {
                continue;
            } else {
                System.out.println(file.getName());
            }
            String localName = fileName.substring(0, fileName.lastIndexOf("_"));
            int sampleNumber = Integer.parseInt(fileName.substring(fileName.lastIndexOf("_") + 1, fileName.lastIndexOf(".")));
            LabelData labelData = JSONReader.deserializeAsLabelData(file.getAbsolutePath());
//            executor.execute( );
            //单线程
            new PositiveMinerThread(localName, 1, labelData).run();
//            break;
        }
        executor.shutdown();
    }


    /**
     * 根据正样本所在方法 挖掘
     * @param executor
     * @param ratio
     * @throws IOException
     */
    public static void mineNegative(ThreadPoolExecutor executor, int ratio) throws IOException {
        HashMap<String, Integer> map = new HashMap<>();
        ArrayList<File> list = new ArrayList<>();
        Utils.getFileList(list, Constants.LABELED_DATA_PATH, "json");
        for (int i = list.size() - 1; i >= 0; --i) {
            File file = list.get(i);
            String fileName = file.getName();
            String localName = fileName.substring(0, fileName.lastIndexOf("_"));
            map.put(localName, 1 + map.getOrDefault(localName, 0));
        }
        List<File> minedNegList = new ArrayList<>();
        Map<String, Integer> minedNegProjMap = new HashMap<>();
        Utils.getFileList(minedNegList, Constants.NEGATIVE_OUTPUT_PATH, "json");
        minedNegList.forEach(file -> {
            String fileName = file.getName();
            String localName = fileName.substring(0, fileName.lastIndexOf("_"));
            minedNegProjMap.put(localName, 1 + minedNegProjMap.getOrDefault(localName, 0));
        });
        minedNegProjMap.forEach((key, value) -> {
//            System.out.println(key+", " + value+" "+map.get(key)*ratio );
            try {
                assertTrue(key + ", " + value + " " + map.get(key) * ratio, value == map.get(key) * ratio);
            } catch (AssertionError e) {
                e.printStackTrace();
            }
        });
        System.out.println(minedNegProjMap.keySet().size());
        for (String key : map.keySet()) {
//            System.out.println(key+", " + map.get(key)  );
            Integer sampleNumber = map.get(key);
            if (minedNegProjMap.containsKey(key)) {
                continue;
            }

//            break;
        }
        executor.shutdown();
    }

    public static void doRefactoringMiner(ThreadPoolExecutor executor) throws Exception {
        File[] list = new File(Constants.PREFIX_PATH).listFiles();
        int i = 0;
        for (File f : list) {
            String localName = f.getName();
            if (new File(Constants.PREFIX_RM_DATA_PATH + localName + ".json").exists()
                    || !f.isDirectory()) {
                continue;
            }
            i++;
//            System.out.println(localName);
            executor.execute(new RefactoringMinerThread(localName));
        }
        executor.shutdown();
        System.out.println(i);
    }
}

