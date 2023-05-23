package miner;

import ast.ProjectsParser;
import git.GitUtils;
import io.json.JsonFileSplitter;
import json.EVRecord;
import json.MetaData;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.CompilationUnit;
import sample.Constants;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


@Slf4j
public class NegativeMinerThread extends AbstractMinerThread {

    public NegativeMinerThread(String projectName, int totalRecords) {
        super(projectName,totalRecords);
    }

    @Override
    public void init(String projectName) {
        fJsonFileSplitter = new JsonFileSplitter();
        fProjectName = projectName;
        String gitPath = Constants.PREFIX_PATH + fProjectName + System.getProperty("file.separator");
        GitUtils.removeGitLock(gitPath);
        fCommitID = null;
        HashSet<String> set = new HashSet<>();
        fFileList = new ArrayList<>();
        Path projectPath = Paths.get(gitPath);
        while (true) {
            if (fCommitID != null)
                set.add(fCommitID);
            try {
                fCommitID = GitUtils.getLatestCommitSHA(gitPath, set);
                if(fCommitID==null){
                    break;
                }
                GitUtils.rollbackToCommit(gitPath, fCommitID);
                fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
                if (fProjectsParser.getTargetJavaFiles().size() >= fTotalRecords) {
                    break;
                }
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    }

    @Override
    public void run() {
        try {
            init(fProjectName);
            analyzeProject();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void analyzeProject() throws IOException {
        ArrayList<EVRecord> evRecords = new ArrayList<>();

        if (this.fCommitID == null) {
            return;
        }

        int id = 1;
        HashSet<String> targetJavaFiles = fProjectsParser.getTargetJavaFiles();
        int size = targetJavaFiles.size();

        log.info("start analyzing {} ... total {} files, will sample {} refactorings", fProjectName, size, fTotalRecords);
        int currentProcessed = -1;
        while (fRandomSelection.getCurrentRecords() < fTotalRecords) {
            String path = fRandomSelection.generateRandomObjectFromSet(targetJavaFiles);
            if (path == null) {
                log.error("no more files to process in project {}", fProjectName);
                break;
            }
            CompilationUnit cu = fProjectsParser.parse(path);
            if (cu == null) {
                targetJavaFiles.remove(path);
                continue;
            }
            NegativeExpressionVisitor visitor = new NegativeExpressionVisitor(cu);
            cu.accept(visitor);
            String str = path.replace(Constants.PREFIX_PATH + fProjectName, "").replace("\\", "/");
            Set<Map.Entry<String, ArrayList<MetaData>>> entrySet = visitor.recordMap.entrySet();
            Map.Entry<String, ArrayList<MetaData>> entry = fRandomSelection.generateRandomObjectFromSet(entrySet);
            if (entry == null || entry.getValue().isEmpty()) {
                targetJavaFiles.remove(path);
                continue;
            }
            ArrayList<MetaData> metaDataList = entry.getValue();
            for (MetaData m : metaDataList) {
                visitor.loadMetaData(m);
            }
            Iterator<MetaData> iterator = metaDataList.iterator();
            String nodeContext = metaDataList.get(0).getParentDataList().get(metaDataList.get(0).getParentDataList().size() - 1).getNodeContext();
            //删除不同parent
            while (iterator.hasNext()) {
                MetaData m = iterator.next();
                if (!m.getParentDataList().get(m.getParentDataList().size() - 1).getNodeContext().equals(
                        nodeContext
                )) {
                    iterator.remove();
                }
            }
            EVRecord r = new EVRecord();
            r.setProjectName(fProjectName);
            r.setId(id++);
            r.setExpression(metaDataList.get(0).getNodeContext());
            r.setCommitID(fCommitID);
            r.setFilePath(str);
            r.setOccurrences(metaDataList.size());
            r.setExpressionList(metaDataList);
            r.generatePositionList(metaDataList);
            r.setLayoutRelationDataList();
            fJsonFileSplitter.writeJsonArrayInSampled(r, false);

            fRandomSelection.incCurrentRecords();
            targetJavaFiles.remove(path);
            int currentRecords = fRandomSelection.getCurrentRecords();
            int process = (100 * (currentRecords)) / fTotalRecords;
            if (process % 5 == 0 && currentProcessed != (100 * (currentRecords)) / fTotalRecords) {
                currentProcessed = (100 * (currentRecords)) / this.fTotalRecords;
                log.info("analyzing {} ... {}%, total {} refactorings", fProjectName, currentProcessed, fTotalRecords);
            }
        }

    }

}
