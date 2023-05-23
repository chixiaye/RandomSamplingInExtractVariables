package io.json;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import json.EVRecord;
import json.LabelData;

import java.io.File;
import java.io.IOException;
import java.util.List;

// deserialize json as EVRecord
public class JSONReader {
    public static List<EVRecord> deserializeAsEVRecordList(String json) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        List<EVRecord> evRecordList = objectMapper.readValue(new File(json), new TypeReference<>() {
        });

        return evRecordList;
    }

    public static LabelData deserializeAsLabelData(String json) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();

        return objectMapper.readValue(new File(json), new TypeReference<>() {
        });
    }


}
