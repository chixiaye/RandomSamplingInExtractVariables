package io.json;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import json.EVRecord;
import refactoringminer.json.RefactoringMinedData;
import sample.Constants;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import static sample.Constants.FILE_SEPARATOR_PROPERTY;

public class JsonFileSplitter {
    public static final int OBJECTS_PER_FILE = 100;
    private final ObjectMapper mapper = new ObjectMapper();

    public JsonFileSplitter() {
        mapper.enable(JsonGenerator.Feature.AUTO_CLOSE_TARGET);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void writeJsonArrayInCaseStudy(EVRecord r, boolean flag) throws IOException {
        String preFix = flag ? Constants.CASE_STUDY_POSITIVE_OUTPUT_PATH : Constants.CASE_STUDY_NEGATIVE_OUTPUT_PATH;
        String fileName = preFix   + r.getProjectName() + "_" + r.getId() + ".json";
        File file = new File(fileName);
        if (!file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        try (FileWriter fileWriter = new FileWriter(fileName, false);
             JsonGenerator generator = mapper.getFactory().createGenerator(fileWriter).useDefaultPrettyPrinter()) {
            generator.writePOJO(r);
        }
    }

    public void writeJsonArrayInSampled(EVRecord r, boolean flag) throws IOException {
        String preFix = flag ? Constants.POSITIVE_OUTPUT_PATH : Constants.NEGATIVE_OUTPUT_PATH;
        String fileName = preFix + r.getProjectName() + "_" + r.getId() + ".json";
        File file = new File(fileName);
        if (!file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        try (FileWriter fileWriter = new FileWriter(fileName, false);
             JsonGenerator generator = mapper.getFactory().createGenerator(fileWriter).useDefaultPrettyPrinter()) {
            generator.writePOJO(r);
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
