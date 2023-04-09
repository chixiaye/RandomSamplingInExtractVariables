package io;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import json.EVRecord;
import sample.Constants;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class JsonFileSplitter {
    public static final int OBJECTS_PER_FILE = 1000;
    private Integer total;
    private final ObjectMapper mapper = new ObjectMapper();

    public JsonFileSplitter(){
        total=0;
        mapper.enable(JsonGenerator.Feature.AUTO_CLOSE_TARGET);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void writeJsonArray(List<EVRecord> evRecords, String projectName) throws IOException {
        if(evRecords.isEmpty()){
            return;
        }
        int startIndex = total ;
        int fileIndex = OBJECTS_PER_FILE* (startIndex/ OBJECTS_PER_FILE) +1;
        String fileName = Constants.OUTPUT_PATH + projectName + File.separator +projectName + "_" +  fileIndex   + ".json";
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
            total =  startIndex + len;
            generator.writeEndArray();
            if(len !=evRecords.size() ){
                writeJsonArray(evRecords.subList(len,evRecords.size() ), projectName);
            }
        }
    }

}