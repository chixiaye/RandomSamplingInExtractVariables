package sample;


import io.excel.ExcelData;
import io.excel.ExcelReader;
import lombok.extern.slf4j.Slf4j;
import miner.NegativeMinerThread;
import miner.PositiveMinerThread;

import java.io.File;
import java.io.IOException;
import java.util.Comparator;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static io.json.JsonFileSplitter.OBJECTS_PER_FILE;

@Slf4j
public class Main {

    static  final  int CORE_POOL_SIZE = Runtime.getRuntime().availableProcessors()  ;
    static  final  int MAX_POOL_SIZE = CORE_POOL_SIZE +1  ;

    public static void main(String[] args) throws IOException, InterruptedException {
        ThreadPoolExecutor executor =  new ThreadPoolExecutor(CORE_POOL_SIZE, MAX_POOL_SIZE,
                5L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
        log.info("core pool size: {}, max pool size: {}",CORE_POOL_SIZE,MAX_POOL_SIZE);
        minePositive(executor);
    }

    public static void minePositive(ThreadPoolExecutor executor){
        ExcelReader excelReader = new ExcelReader(Constants.EXCEL_PATH);
        excelReader.read();
        excelReader.getExcelDataList().sort(Comparator.comparing(ExcelData::getAccount));
        for (int i=excelReader.getExcelDataList().size()-1; i>=0  ;--i) {
            ExcelData v = excelReader.getExcelDataList().get(i);
            String localName=v.getAccount()+"@"+v.getRepository();
            int sampleNumber= v.getNumber();
            int lastFileIndex = OBJECTS_PER_FILE* ( sampleNumber/ OBJECTS_PER_FILE) +1;
            if(new File( Constants.POSITIVE_OUTPUT_PATH + localName+File.separator +localName + "_" +  lastFileIndex   + ".json").exists()){
                log.info("project {} has been mined.",localName);
                continue;
            }
            log.info("project {} is being mining.",localName);
            executor.execute(new PositiveMinerThread(localName,sampleNumber));
            if(sampleNumber>0)
                break;
        }
        executor.shutdown();
    }

    public static void mineNegative(ThreadPoolExecutor executor) throws IOException {

        ExcelReader excelReader = new ExcelReader(Constants.EXCEL_PATH);
        excelReader.read();
        excelReader.getExcelDataList().sort(Comparator.comparing(ExcelData::getAccount));
        for (int i=excelReader.getExcelDataList().size()-1; i>=0  ;--i) {
            ExcelData v = excelReader.getExcelDataList().get(i);
            String localName=v.getAccount()+"@"+v.getRepository();
//            if(!localName.contains("ankidroid")){
//                continue;
//            }
            int sampleNumber= v.getNumber();
            int lastFileIndex = OBJECTS_PER_FILE* ( sampleNumber/ OBJECTS_PER_FILE) +1;
            if(new File( Constants.NEGATIVE_OUTPUT_PATH + localName+File.separator +localName + "_" +  lastFileIndex   + ".json").exists()){
//                log.info("project {} has been mined.",localName);
                continue;
            }
//            if(sampleNumber>10)
//                continue;
//            if(i==3){
//                break;
//            }
//            i++;
//            log.info("project {} is unMined.",localName);
            executor.execute(new NegativeMinerThread(localName,sampleNumber));
//            executor.execute(new MinerThread(localName,sampleNumber));
//            Thread.sleep();
        }
//        for(String s:projectNames){
//            executor.execute(new MinerThread(s,10));
//        }
        executor.shutdown();
    }
}

