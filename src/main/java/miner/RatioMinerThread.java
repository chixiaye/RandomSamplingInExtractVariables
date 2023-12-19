package miner;

import ast.ProjectsParser;
import git.GitUtils;
import json.LabelData;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.*;
import sample.Constants;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

@Slf4j
public class RatioMinerThread extends AbstractMinerThread {

    LabelData fLabelData;
    @Getter
    private RatioVisitor visitor;

    public RatioMinerThread(String projectName, int totalRecords, LabelData labelData) {
        super(projectName, totalRecords);
        fLabelData = labelData;
        visitor = new RatioVisitor();
    }

    @Override
    public void analyzeProject() throws IOException {
        String gitPath = Constants.PREFIX_PATH + fProjectName + System.getProperty("file.separator");
        GitUtils.removeGitLock(gitPath);
        Path projectPath = Paths.get(gitPath);
        String commitID = fLabelData.getRefactoredCommitID();
        String filePath = fLabelData.getRefactoredFilePath();
        String name = fLabelData.getRefactoredName();
        NodePosition nodePosition = fLabelData.getRefactoredPositionList()[0];
        try {
            GitUtils.rollbackToCommit(gitPath, commitID);
            fCommitID = commitID;
            fProjectsParser = new ProjectsParser(new Path[]{projectPath}, projectPath, projectPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
        CompilationUnit cu = fProjectsParser.parse(gitPath + Constants.FILE_SEPARATOR_PROPERTY + filePath);
        // 根据nodePosition和name找到所在的方法体
        ASTNode[] enclosingBody = new ASTNode[1];
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
        enclosingBody[0].accept(visitor);
//        System.out.println(enclosingBody[0]);
//        System.out.println(visitor.getCount());

        log.info("searching ... {}-{}， got count {}", fProjectName, fLabelData.getId(), visitor.getCount() );

        //enclosingBody
//        cu.getCommentList().forEach(node -> System.out.println(node));
    }

    @Override
    public void init(String projectName) {

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
