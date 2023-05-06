package sample;


import io.excel.ExcelData;
import io.excel.ExcelReader;
import lombok.extern.slf4j.Slf4j;
import miner.NegativeMinerThread;
import miner.PositiveMinerThread;
import refactoringminer.handler.RefactoringMinerThread;
import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static io.json.JsonFileSplitter.OBJECTS_PER_FILE;

@Slf4j
public class Main {

    static final int CORE_POOL_SIZE = Runtime.getRuntime().availableProcessors() * 4 / 5;
    static final int MAX_POOL_SIZE = CORE_POOL_SIZE + 1;

    public static void main(String[] args) throws Exception {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(CORE_POOL_SIZE, MAX_POOL_SIZE,
                5L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
        log.info("core pool size: {}, max pool size: {}", CORE_POOL_SIZE, MAX_POOL_SIZE);
//        minePositive(executor);
//        mineNegative(executor);
        doRefactoringMiner(executor);
    }

    public static void minePositive(ThreadPoolExecutor executor){
        ExcelReader excelReader = new ExcelReader(Constants.EXCEL_PATH);
        excelReader.read();
        excelReader.getExcelDataList().sort(Comparator.comparing(ExcelData::getAccount));
        for (int i = excelReader.getExcelDataList().size() - 1; i >= 0; --i) {
            ExcelData v = excelReader.getExcelDataList().get(i);
            String localName = v.getAccount() + "@" + v.getRepository();
            int sampleNumber = v.getNumber();
            int lastFileIndex = OBJECTS_PER_FILE * (sampleNumber / OBJECTS_PER_FILE) + 1;
            if (!localName.equals("Activiti@Activiti")) {
                continue;
            }
//            ArrayList<File> tmp = new ArrayList<>();
//            Utils.getFileList(tmp, Constants.POSITIVE_OUTPUT_PATH + localName+Constants.FILE_SEPARATOR_PROPERTY
//                    , "json");
//            if(tmp.size()!=0){
//                new File(Constants.POSITIVE_OUTPUT_PATH + localName).delete();
//                System.out.println("delete "+localName);
//                continue;
//            }
//            if(new File( Constants.POSITIVE_OUTPUT_PATH + localName+File.separator +localName + "_" +  lastFileIndex   + ".json").exists()){
//                continue;
//            }
            executor.execute(new PositiveMinerThread(localName, sampleNumber));

        }
        executor.shutdown();
    }

    public static void mineNegative(ThreadPoolExecutor executor) throws IOException {

        ExcelReader excelReader = new ExcelReader(Constants.EXCEL_PATH);
        excelReader.read();
        excelReader.getExcelDataList().sort(Comparator.comparing(ExcelData::getNumber));
        for (int i = 0; i < excelReader.getExcelDataList().size(); ++i) {
            ExcelData v = excelReader.getExcelDataList().get(i);
            String localName = v.getAccount() + "@" + v.getRepository();
            int sampleNumber = v.getNumber();
            int lastFileIndex = OBJECTS_PER_FILE * (sampleNumber / OBJECTS_PER_FILE) + 1;
            if (new File(Constants.NEGATIVE_OUTPUT_PATH + localName + File.separator + localName + "_" + lastFileIndex + ".json").exists()) {
                continue;
            }
//            log.info("project {} has not been mined.",localName);

            executor.execute(new NegativeMinerThread(localName, sampleNumber));
        }
        executor.shutdown();
    }

    public static void doRefactoringMiner(ThreadPoolExecutor executor) throws Exception {
        ExcelReader excelReader = new ExcelReader(Constants.EXCEL_PATH);
        excelReader.read();
        excelReader.getExcelDataList().sort(Comparator.comparing(ExcelData::getAccount));
        for (int i = 0; i < excelReader.getExcelDataList().size(); ++i) {
            ExcelData v = excelReader.getExcelDataList().get(i);
            String localName = v.getAccount() + "@" + v.getRepository();
            if (new File(Constants.PREFIX_RM_DATA_PATH + localName + ".json").exists()) {
                continue;
            }
            executor.execute(new RefactoringMinerThread(localName));
        }
        executor.shutdown();

    }
}

