package io.json;

import junit.framework.TestCase;
import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class EVRecordReaderTest extends TestCase {

    public void testDeserializeAsEVRecord() {
        try {
            ArrayList<File> arrayList = new ArrayList<>();
            Utils.getFileList(arrayList,"C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\","json");
            int res=0;
            for(File f:arrayList){
//                System.out.println(f.getName());
                res+=EVRecordReader.deserializeAsEVRecordList( f.getAbsolutePath());
            }
            System.out.println(res);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}