package io;

import json.EVRecord;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import sample.Constants;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.apache.commons.io.FileUtils.readLines;

@Slf4j
public class RMDataReader extends AbstractIOer{

    public RMDataReader(String filePath) {
        this.filePath = filePath;
    }

    public static String convertPath(String path) {
        Path pathObj = Paths.get(path);
        int count = pathObj.getNameCount();
        if (count < 2) {
            return path;
        } else {
            Path subPath = pathObj.subpath(1, count);
            return subPath.toString().replace(Constants.FILE_SEPARATOR_PROPERTY, "/");
        }
    }

    public ArrayList<EVRecord> doReadAction() throws IOException {
        File file=new File(filePath);
//        log.info("file path:{}",file.getAbsolutePath());
        ArrayList<EVRecord> recordDataArrayList = new ArrayList<>();
        if (file.exists()){
            List<String> lines = readLines(file);
            for (String line:lines){
                EVRecord record = resolveValueBag(line);
                if (record !=null){
                    recordDataArrayList.add(record);
                }
            }
        }
        return recordDataArrayList;
    }


    public EVRecord resolveValueBag(String variableLine) {
// f0e1b75fb9632a50cf37307630493b9780a26bb0###/addthis_stream-lib/src/test/java/com/clearspring/analytics/stream/cardinality/TestCountThenEstimate.java###/TestCountThenEstimate.java###com.clearspring.analytics.stream.cardinality.TestCountThenEstimate###assertCountThenEstimateEquals:CountThenEstimate CountThenEstimate ###assertArrayEquals(expected.estimator.getBytes(),actual.estimator.getBytes());###expBytes###expected.estimator.getBytes()###256:13:256:91
// commitID+"###"+javaFilePath +"###" javaFileName +"###" + classPath + "###" + methodInfo +"###" + involvedExpression (parent node) +"###" + variableName+"###" + initializer +"###" + startLine:startColumn:EndLine:EndColumn;
        String[] splitArray = variableLine.replaceAll("\n", "").split("###");
        VariableData variableData = new VariableData();
        if (splitArray.length != 9) {
            log.warn("not valid record <8");
            return null;
        } else {
            String commitID = splitArray[0];
            variableData.setCommitID( commitID);
            String javaFileName = splitArray[2];
            variableData.setJavaFileName(javaFileName);
            String javaFilePath = splitArray[1];
            variableData.setJavaFilePath( javaFilePath);
            String classPath = splitArray[3];
            variableData.setClassPath( classPath);
            String methodInfo = splitArray[4];
            variableData.setMethodInfo( methodInfo);
            String involvedExpression = splitArray[5];
            variableData.setInvolvedExpression( involvedExpression);
            String variableName = splitArray[6];
            variableData.setVariableName( variableName);
            String initializer = splitArray[7];
            variableData.setInitializer( initializer);
            String lineAndColumn = splitArray[8];
            variableData.setLineAndColumn( lineAndColumn);
        }

        EVRecord evRecord = new EVRecord();
        evRecord.setCommitID(variableData.getCommitID());
        evRecord.setExpression(variableData.getInitializer());
        evRecord.setFilePath(convertPath(variableData.getJavaFilePath()));
        evRecord.setName(variableData.getVariableName());
        String[] strings = variableData.getLineAndColumn().split(":");
        evRecord.setNodePosition(new NodePosition(Integer.parseInt(strings[0]),
                Integer.parseInt(strings[1]),
                Integer.parseInt(strings[2]),
                Integer.parseInt(strings[3]) ,
                variableData.getInitializer().length()));

        return evRecord;
    }


    class VariableData{
        @Getter
        @Setter
        String commitID;
        @Getter
        @Setter
        String javaFilePath;
        @Getter
        @Setter
        String javaFileName;
        @Getter
        @Setter
        String classPath;
        @Getter
        @Setter
        String methodInfo;
        @Getter
        @Setter
        String involvedExpression;
        @Getter
        @Setter
        String variableName;
        @Getter
        @Setter
        String initializer;
        @Getter
        @Setter
        String lineAndColumn;
    }
}
