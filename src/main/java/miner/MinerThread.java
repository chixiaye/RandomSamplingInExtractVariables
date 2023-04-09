package miner;

import ast.ProjectsParser;
import json.EVRecord;
import io.JsonFileSplitter;
import json.MetaData;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.CompilationUnit;
import sample.Constants;
import utils.RandomSelection;
import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


@Slf4j
public class MinerThread extends Thread {
    private ProjectsParser fProjectsParser;
    private JsonFileSplitter fJsonFileSplitter;

    private final int fTotalRecords;

    public MinerThread(String projectName, int totalRecords) {
        init(projectName);
        this.setName(projectName);
        fTotalRecords = totalRecords;
        fRandomSelection = new RandomSelection(totalRecords);
    }

    @Override
    public void run() {
        try {
            analyzeProject();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    String fProjectName;
    String fCommitID;
    ArrayList<File> fFileList;

    RandomSelection fRandomSelection;

    public void init(String projectName) {
        fJsonFileSplitter = new JsonFileSplitter();
        fProjectName = projectName;
        String gitPath = Constants.PREFIX_PATH + fProjectName;
        fCommitID = Utils.getCurrentSHA(gitPath);
        fFileList = new ArrayList<>();
        try {
            Path projectPath = Paths.get(gitPath);
            fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void analyzeProject() throws IOException {
        ArrayList<EVRecord> evRecords = new ArrayList<>();
        int id = 1;
        HashSet<String> targetJavaFiles = fProjectsParser.getTargetJavaFiles();
        int size = targetJavaFiles.size();

        log.info("start analyzing {} ... total {} files", fProjectName, size);
        int currentProcessed = -1;
        while (fRandomSelection.getCurrentRecords() < this.fTotalRecords){
            String path = fRandomSelection.generateRandomObjectFromSet(targetJavaFiles);
            CompilationUnit cu = fProjectsParser.parse(path);
            if (cu == null) {
                log.error(path + " error in parsing.");
                targetJavaFiles.remove(path);
                continue;
            }
            ExpressionVisitor visitor = new ExpressionVisitor(cu);
            cu.accept(visitor);
            String str = path.replace(Constants.PREFIX_PATH + fProjectName, "").replace("\\", "/");
            Set<Map.Entry<String, ArrayList<MetaData>>> entrySet = visitor.recordMap.entrySet();
            Map.Entry<String, ArrayList<MetaData>> entry = fRandomSelection.generateRandomObjectFromSet(entrySet);
            if(entry==null){
                targetJavaFiles.remove(path);
                continue;
            }
            ArrayList<MetaData> metaDataList = entry.getValue();
            for (MetaData m : metaDataList) {
                visitor.loadMetaData(m);
            }
            EVRecord r = new EVRecord();
            r.setProjectName(fProjectName);
            r.setId(id++);
            r.setExpression(metaDataList.get(0).getNodeContext());
            r.setCommitID(fCommitID);
            r.setFilePath(str);
            r.setOccurrences(metaDataList.size());
            r.setExpressionList(metaDataList);
            r.setPositionList(metaDataList);
            evRecords.add(r);
            if (evRecords.size() >= JsonFileSplitter.OBJECTS_PER_FILE) {
                fJsonFileSplitter.writeJsonArray(evRecords, fProjectName);
                evRecords.clear();
            }

            fRandomSelection.incCurrentRecords();
//            entrySet.remove(entry);
            targetJavaFiles.remove(path);
            int currentRecords = fRandomSelection.getCurrentRecords();
            int process = (100 * (currentRecords)) / fTotalRecords;
            if (process % 5 == 0 && currentProcessed != (100 * (currentRecords)) / fTotalRecords) {
                currentProcessed = (100 * (currentRecords)) / this.fTotalRecords;
                log.info("analyzing {} ... {}%", fProjectName, currentProcessed);
            }

        }
        if (!evRecords.isEmpty()) {
            fJsonFileSplitter.writeJsonArray(evRecords, fProjectName);
            evRecords.clear();
        }

    }

}
