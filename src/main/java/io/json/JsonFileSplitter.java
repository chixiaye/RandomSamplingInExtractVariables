package io.json;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import json.EVRecord;
import lombok.Getter;
import lombok.Setter;
import org.junit.Assert;
import refactoringminer.json.RefactoringMinedData;
import sample.Constants;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public class JsonFileSplitter {
    public static final int OBJECTS_PER_FILE = 100;
    @Setter
    @Getter
    private Integer total;
    private final ObjectMapper mapper = new ObjectMapper();

    public JsonFileSplitter() {
        total = 0;
        mapper.enable(JsonGenerator.Feature.AUTO_CLOSE_TARGET);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void writeJsonArray(List<EVRecord> evRecords, String projectName, boolean flag) throws IOException {
        if(evRecords.isEmpty()){
            return;
        }
        int startIndex = total ;
        int fileIndex = OBJECTS_PER_FILE* (startIndex/ OBJECTS_PER_FILE) +1;
        String preFix=flag? Constants.POSITIVE_OUTPUT_PATH:Constants.NEGATIVE_OUTPUT_PATH;
        String fileName = preFix + projectName + File.separator +projectName + "_" +  fileIndex   + ".json";
        File file = new File(fileName);
        if(!file.getParentFile().exists()){
            file.getParentFile().mkdirs();
        }
        try (FileWriter fileWriter = new FileWriter(fileName, false);
             JsonGenerator generator = mapper.getFactory().createGenerator(fileWriter).useDefaultPrettyPrinter()) {
            generator.writeStartArray();
            int len= 0;
            for (EVRecord r : evRecords) {
                generator.writePOJO(r);
                if(r!=evRecords.get(evRecords.size()-1))
                    generator.writeRaw("\n");
                len++;
                if( 0== (len + startIndex)  % OBJECTS_PER_FILE)
                    break;
            }
            total = startIndex + len;
            generator.writeEndArray();
            if (len != evRecords.size()) {
                writeJsonArray(evRecords.subList(len, evRecords.size()), projectName, flag);
            }
        }
    }

    synchronized public void writeJsonArray(List<RefactoringMinedData> refactoringMinedDatas, String projectName) {
        if (refactoringMinedDatas.isEmpty()) {
            return;
        }
        String preFix = Constants.PREFIX_RM_DATA_PATH;
        String fileName = preFix + projectName + ".json";
        try (FileWriter fileWriter = new FileWriter(fileName, false);
             JsonGenerator generator = mapper.getFactory().createGenerator(fileWriter).useDefaultPrettyPrinter()) {
            generator.writeStartArray();
            Iterator<RefactoringMinedData> iterator = refactoringMinedDatas.iterator();
            while (iterator.hasNext()) {
                RefactoringMinedData r = iterator.next();
                generator.writePOJO(r);
                if (iterator.hasNext())
                    generator.writeRaw("\n");
            }
            generator.writeEndArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


}
