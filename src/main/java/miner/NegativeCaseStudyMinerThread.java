package miner;

import ast.ProjectsParser;
import git.GitUtils;
import io.json.JsonFileSplitter;
import json.*;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.*;
import sample.Constants;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Slf4j
public class NegativeCaseStudyMinerThread extends AbstractMinerThread {
//    List<CaseStudy> fCaseStudyList;

    @Getter
    int fIndex;

    private final CaseStudy caseStudy;

    public NegativeCaseStudyMinerThread(String projectName, CaseStudy caseStudy, int index) {
        super(projectName, -1);
        this.caseStudy = caseStudy;
        fIndex = index;
    }

    @Override
    public void init(String projectName) {
        fJsonFileSplitter = new JsonFileSplitter();
        fProjectName = projectName;
        fFileList = new ArrayList<>();
    }

    @Override
    public void analyzeProject() throws Exception {
        String gitPath = Constants.CASE_STUDY_PREFIX_PATH + fProjectName + System.getProperty("file.separator");
        Path projectPath = Paths.get(gitPath);
        String commitID = caseStudy.getOriginalCommitId();
        String filePath = caseStudy.getPath();
        String name = caseStudy.getOriginalName();
        log.info("extracting features from  {}-{}", fProjectName,  caseStudy.getNo());
        fCommitID = commitID;
        OriginalExpression originalExpression = caseStudy.getOriginalExpressionList()[0];
        NodePosition nodePosition = new NodePosition(originalExpression.getLine(),
                originalExpression.getColumn(), originalExpression.getLine(),
                originalExpression.getLine() + name.length(), name.length());
        try {
            GitUtils.removeGitLock(gitPath);
            GitUtils.rollbackToCommit(gitPath, fCommitID);
            fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
        CompilationUnit cu = fProjectsParser.parse(gitPath + Constants.FILE_SEPARATOR_PROPERTY + filePath);
        // 根据nodePosition和name找到所在的方法体
        ASTNode[] enclosingBody = new ASTNode[1];
        NegativeExpressionVisitor visitor = new NegativeExpressionVisitor(cu);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean preVisit2(ASTNode node) {
                ASTNode parent = node.getParent();
                // 首先找到覆盖位置
                if (enclosingBody[0] == null && name.equals(node.toString())) {
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
                        if (parent instanceof MethodDeclaration || parent instanceof Initializer || parent instanceof LambdaExpression) {
                            enclosingBody[0] = parent;
                        }
                    }
                }
                return super.preVisit2(node);
            }
        });
        if (enclosingBody[0] == null) {
            return;
        }
        enclosingBody[0].accept(visitor);

        String str = caseStudy.getPath().replace(Constants.PREFIX_PATH + fProjectName, "").replace("\\", "/");
        Set<Map.Entry<String, ArrayList<MetaData>>> entrySet = visitor.recordMap.entrySet();
        int size = entrySet.size();
        log.info("start analyzing {} ... total {} samples", fProjectName + "-" + caseStudy.getNo(), size);
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
            // 不重复
            if (caseStudy.getOriginalName().equals(metaDataList.get(0).getNodeContext())){
                continue;
            }
            EVRecord r = new EVRecord();
            r.setProjectName(fProjectName);
            r.setId( fIndex++ );
            r.setExpression(metaDataList.get(0).getNodeContext());
            r.setCommitID(fCommitID);
            r.setFilePath(str);
            r.setOccurrences(metaDataList.size());
            r.setExpressionList(metaDataList);
            r.generatePositionList(metaDataList);
            r.setLayoutRelationDataList();
            r.setIsGetTypeMethod(visitor.getTypeMethodState());
            r.setIsArithmeticExpression(visitor.getArithmeticExpressionState());
            fJsonFileSplitter.writeJsonArrayInCaseStudy(r, false);
            fRandomSelection.incCurrentRecords();
        }
    }

    @Override
    public void run() {
        try {
            init(fProjectName);
            analyzeProject();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
