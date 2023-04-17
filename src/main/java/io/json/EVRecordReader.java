package io.json;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import json.EVRecord;

import java.io.File;
import java.io.IOException;
import java.util.List;

// deserialize json as EVRecord
public class EVRecordReader {
    public static int deserializeAsEVRecordList(String json) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        List<EVRecord> evRecordList = objectMapper.readValue(new File(json), new TypeReference<List<EVRecord>>(){});
        int size= evRecordList.size();
        return size;
    }
}
