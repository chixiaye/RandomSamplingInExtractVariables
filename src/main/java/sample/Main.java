package sample;


import miner.MinerThread;

import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Main {

    static  final  int CORE_POOL_SIZE = Runtime.getRuntime().availableProcessors();
    static  final  int MAX_POOL_SIZE = CORE_POOL_SIZE * 2 ;

    public static void main(String[] args) throws IOException {
        ThreadPoolExecutor executor =  new ThreadPoolExecutor(CORE_POOL_SIZE, MAX_POOL_SIZE,
                5L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy());
        String[] projectNames= new String[]{"flink","jackson-databind"};//
        for(String s:projectNames){
            executor.execute(new MinerThread(s,10));
        }
        executor.shutdown();

    }
}

