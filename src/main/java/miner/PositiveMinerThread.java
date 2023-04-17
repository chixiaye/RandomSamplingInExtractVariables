package miner;

import ast.ProjectsParser;
import git.GitUtils;
import io.RMDataReader;
import io.json.JsonFileSplitter;
import json.EVRecord;
import json.MetaData;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.VariableDeclarationFragment;
import sample.Constants;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Slf4j
public class PositiveMinerThread extends  AbstractMinerThread{

    public PositiveMinerThread(String projectName, int totalRecords) {
        super(projectName, totalRecords);
    }

    @Override
    public void analyzeProject() throws IOException {

        RMDataReader rmDataReader = new RMDataReader(Constants.PREFIX_RM_PATH + fProjectName + ".txt");
        ArrayList<EVRecord> evRecordArrayList  =rmDataReader.doReadAction();
        String gitPath = Constants.PREFIX_PATH + fProjectName;
        Path projectPath = Paths.get(gitPath);
        log.info("start analyzing {} ...  will sample {} refactorings", fProjectName,  fTotalRecords);
        int id = 1;
        int currentProcessed = -1;
        List<EVRecord> evRecords=new ArrayList<>();
        for(EVRecord record:evRecordArrayList){
            try {
                GitUtils.rollbackToCommit(gitPath,record.getCommitID());
                fCommitID=record.getCommitID();
                fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            CompilationUnit cu = fProjectsParser.parse(gitPath + System.getProperty("file.separator") +record.getFilePath());
            PositiveExpressionVisitor visitor = new PositiveExpressionVisitor(cu,record.getName(),record.getNodePosition());
            cu.accept(visitor);
            ArrayList<MetaData> metaDataList = visitor.getMetaDataList();
            for (MetaData m : metaDataList) {
                visitor.reLoadMetaData(m);
            }
            if(metaDataList.size()<=1){
                continue;
            }
            if (evRecords.size() >= JsonFileSplitter.OBJECTS_PER_FILE) {
                fJsonFileSplitter.writeJsonArray(evRecords, fProjectName,true);
                evRecords.clear();
            }
            EVRecord r = new EVRecord();
            r.setProjectName(fProjectName);
            r.setId(id++);

            r.setCommitID(fCommitID);
            if( visitor.metaDataASTNodeHashMap.get(metaDataList.get(0)).getParent()  instanceof VariableDeclarationFragment vdf){
                visitor.reLoadMetaData( metaDataList.get(0) ,vdf.getInitializer());
                r.setExpression( vdf.getInitializer().toString());
            }
            r.setFilePath(record.getFilePath());
            r.setOccurrences(metaDataList.size());
            r.setExpressionList(metaDataList);
            r.generatePositionList(metaDataList);
            r.setLayoutRelationDataList();
            evRecords.add(r);
            fRandomSelection.incCurrentRecords();
//            entrySet.remove(entry);
            int currentRecords = fRandomSelection.getCurrentRecords();
            int process = (100 * (currentRecords)) / fTotalRecords;
            if (process % 5 == 0 && currentProcessed != (100 * (currentRecords)) / fTotalRecords) {
                currentProcessed = (100 * (currentRecords)) / this.fTotalRecords;
                log.info("analyzing {} ... {}%", fProjectName, currentProcessed);
            }
        }
        if (!evRecords.isEmpty()) {
            fJsonFileSplitter.writeJsonArray(evRecords, fProjectName,true);
            evRecords.clear();
        }
    }

    @Override
    public void init(String projectName) {
        fJsonFileSplitter = new JsonFileSplitter();
        fProjectName = projectName;
        fCommitID = null;
        fFileList = new ArrayList<>();
    }

    @Override
    public void run() {
        try {
            init(fProjectName);
            analyzeProject();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
