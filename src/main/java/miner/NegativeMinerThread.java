package miner;

import ast.ProjectsParser;
import git.GitUtils;
import io.json.JsonFileSplitter;
import json.EVRecord;
import json.LabelData;
import json.MetaData;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.*;
import sample.Constants;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


@Slf4j
public class NegativeMinerThread extends AbstractMinerThread {

    private final LabelData labelData;
    @Getter
    int fIndex ;
    //每个文件最多采集多少个
    private int fMaxRecordsPerFile = 1;

    public NegativeMinerThread(String projectName, LabelData labelData,int  index) {
        super(projectName,-1);
        fIndex = index;
        this.labelData = labelData;
    }

    @Override
    public void init(String projectName) {
        fJsonFileSplitter = new JsonFileSplitter();
        fProjectName = projectName;
        String gitPath = Constants.PREFIX_PATH + fProjectName + System.getProperty("file.separator");
        GitUtils.removeGitLock(gitPath);
        fCommitID = labelData.getRefactoredCommitID();
        fFileList = new ArrayList<>();
        Path projectPath = Paths.get(gitPath);
        try {
            GitUtils.rollbackToCommit(gitPath, fCommitID);
            fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
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
        if (this.fCommitID == null) {
            return;
        }
        String gitPath = Constants.PREFIX_PATH + fProjectName + System.getProperty("file.separator");
        String path = labelData.getRefactoredFilePath();
        String name = labelData.getRefactoredName();
        NodePosition nodePosition = labelData.getRefactoredPositionList()[0];
        CompilationUnit cu ;
        try{
            cu = fProjectsParser.parse(gitPath + Constants.FILE_SEPARATOR_PROPERTY + path);
        }catch (Exception e){
            log.error("parse error {}",path);
            return;
        }
        // 根据nodePosition和name找到所在的方法体
        ASTNode[] enclosingBody = new ASTNode[1];
        NegativeExpressionVisitor visitor = new NegativeExpressionVisitor(cu);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean preVisit2(ASTNode node) {
                ASTNode parent = node.getParent();
                // 首先找到声明的位置
                if (enclosingBody[0] == null && node instanceof SimpleName
                        && parent instanceof VariableDeclarationFragment vdf &&
                        node.equals(vdf.getName()) && name.equals(node.toString())) {
                    int offset = node.getStartPosition();
                    int length = node.getLength();
                    NodePosition pos = new NodePosition(cu.getLineNumber(offset), cu.getColumnNumber(offset)
                            , cu.getLineNumber(offset + length), cu.getColumnNumber(offset + length), length);
                    if (pos.getStartLineNumber() >= nodePosition.getStartLineNumber()
                            && pos.getEndLineNumber() <= nodePosition.getEndLineNumber()) {
                        // 对覆盖范围做初始化
                        while (parent != null) {
                            if (parent instanceof MethodDeclaration || parent instanceof Initializer || parent instanceof LambdaExpression) {
                                break;
                            }
                            parent = parent.getParent();
                        }
                        enclosingBody[0] = parent;
                    }
                }
                return super.preVisit2(node);
            }
        });
        if(enclosingBody[0]==null){
            return ;
        }
        enclosingBody[0].accept(visitor);

        String str = path.replace(Constants.PREFIX_PATH + fProjectName, "").replace("\\", "/");
        Set<Map.Entry<String, ArrayList<MetaData>>> entrySet = visitor.recordMap.entrySet();
        int size = entrySet.size();
        log.info("start analyzing {} ... total {} samples", fProjectName+"-"+labelData.getId(), size);
        for (Map.Entry<String, ArrayList<MetaData>> entry : entrySet) {
            String key = entry.getKey();
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
            r.setId(fIndex++);
            r.setExpression(metaDataList.get(0).getNodeContext());
            r.setCommitID(fCommitID);
            r.setFilePath(str);
            r.setOccurrences(metaDataList.size());
            r.setExpressionList(metaDataList);
            r.generatePositionList(metaDataList);
            r.setLayoutRelationDataList();
            r.setIsGetTypeMethod(visitor.getTypeMethodState());
            r.setIsArithmeticExpression(visitor.getArithmeticExpressionState());
            fJsonFileSplitter.writeJsonArrayInSampled(r, false);
            fRandomSelection.incCurrentRecords();
//            visitor.recordMap.put(key, value);
//            int currentRecords = fRandomSelection.getCurrentRecords();
//            int process = (100 * (currentRecords)) / size;
//            if (process % 5 == 0 && currentProcessed != (100 * (currentRecords)) / size) {
//                currentProcessed = (100 * (currentRecords)) /size;
//                log.info("analyzing {} ... {}%, total {} refactorings", fProjectName, currentProcessed, size);
//            }
        }

    }

}
