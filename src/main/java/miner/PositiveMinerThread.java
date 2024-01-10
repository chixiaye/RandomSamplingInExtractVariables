package miner;

import ast.ProjectsParser;
import git.GitUtils;
import io.json.JsonFileSplitter;
import json.EVRecord;
import json.LabelData;
import json.MetaData;
import json.ParentData;
import json.utils.NodePosition;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.CompilationUnit;
import sample.Constants;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Slf4j
public class PositiveMinerThread extends AbstractMinerThread {

    private final LabelData fLabelData;

    public PositiveMinerThread(String projectName, int totalRecords, LabelData labelData) {
        super(projectName, totalRecords);
        this.fLabelData = labelData;
    }

    @Override
    public void analyzeProject() throws IOException {
        String gitPath = Constants.PREFIX_PATH + fProjectName + System.getProperty("file.separator");
        GitUtils.removeGitLock(gitPath);
        Path projectPath = Paths.get(gitPath);
        log.info("extracting features from  {}-{}", fProjectName, fLabelData.getId());
        String commitID = fLabelData.getRefactoredCommitID();
        String filePath = fLabelData.getRefactoredFilePath();
        String name = fLabelData.getRefactoredName();
        NodePosition nodePosition = fLabelData.getRefactoredPositionList()[0];
//        for (NodePosition nodePosition1:fLabelData.getRefactoredPositionList()){
//            System.out.println(nodePosition1);
//        }
        try {
            GitUtils.rollbackToCommit(gitPath, commitID);
            fCommitID = commitID;
            fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
        CompilationUnit cu = fProjectsParser.parse(gitPath + Constants.FILE_SEPARATOR_PROPERTY + filePath);

        PositiveExpressionVisitor visitor = new PositiveExpressionVisitor(cu, name, nodePosition);
        cu.accept(visitor);
        ArrayList<MetaData> metaDataList = new ArrayList<>();
        MetaData base = visitor.getMetaDataList().get(0);
        for (int i = 1; i < visitor.getMetaDataList().size(); i++) {
            MetaData metaData = visitor.getMetaDataList().get(i);
            String variableName = metaData.getNodeContext();
            metaData.setNodeContext(base.getNodeContext());
            NodePosition position = metaData.getNodePosition();
            position.setCharLength(base.getNodeContext().length());
            position.setEndColumnNumber(position.getStartColumnNumber() + position.getCharLength());
            metaData.setNodePosition(position);
            metaData.setTokenLength();

            for (int j = 0; j < metaData.getParentDataList().size(); j++) {
                ParentData parentData = metaData.getParentDataList().get(j);
                parentData.setNodeContext(parentData.getNodeContext().replace(variableName, base.getNodeContext()));
                NodePosition pos = parentData.getNodePosition();
                if(pos.getStartLineNumber()==pos.getEndColumnNumber()){
                    pos.setEndColumnNumber(pos.getStartColumnNumber() + pos.getCharLength());
                }
                pos.setCharLength(parentData.getNodeContext().length());
                parentData.setAstNodeNumber(base.getAstNodeNumber()+ parentData.getAstNodeNumber() -1);
                parentData.setAstHeight(Math.max(base.getAstHeight()+j ,parentData.getAstHeight()));
            }

        }



        EVRecord r = new EVRecord();
        r.setProjectName(fProjectName);
        r.setExpression(fLabelData.getOriginalName());
        r.setId(fLabelData.getId());
        r.setCommitID(fCommitID);
        r.setFilePath(filePath);
        r.setOccurrences(metaDataList.size() );
        r.setExpressionList(metaDataList);
        r.generatePositionList(metaDataList);
        r.setIsGetTypeMethod(visitor.getTypeMethodState());
        r.setIsArithmeticExpression(visitor.getArithmeticExpressionState());

//        r.se

        r.initLayoutRelationDataListInit();
        if (visitor.getFCoveredNodePosition() != null && r.getOccurrences() > 1){
            fJsonFileSplitter.writeJsonArrayInSampled(r, true);
        }

    }

    @Override
    public void init(String projectName) {
        fJsonFileSplitter = new JsonFileSplitter();
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
