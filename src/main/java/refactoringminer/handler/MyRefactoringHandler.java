package refactoringminer.handler;

import gr.uom.java.xmi.diff.CodeRange;
import gr.uom.java.xmi.diff.ExtractVariableRefactoring;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.revwalk.RevWalk;
import org.refactoringminer.api.Refactoring;
import org.refactoringminer.api.RefactoringHandler;
import refactoringminer.json.RefactoringMinedData;

import java.io.IOException;
import java.util.*;

@Slf4j
public class MyRefactoringHandler extends RefactoringHandler {
    Repository fRepo;
    int currentId;
    String fProjectName;

    @Getter
    ArrayList<RefactoringMinedData> RefactoringMinedDataList;

    MyRefactoringHandler(Repository repo, String projectName, int startIndex) {
        fRepo = repo;
        currentId = startIndex;
        fProjectName = projectName;
        RefactoringMinedDataList = new ArrayList<>();
    }

    @Override
    public void handle(String commitId, List<Refactoring> refactorings) {
        super.handle(commitId, refactorings);
        RevWalk walk = new RevWalk(fRepo);
        for (Refactoring refactoring : refactorings) {
            if (refactoring instanceof ExtractVariableRefactoring extractVariableRefactoring) {
                // 符合提取变量的定义
                if (extractVariableRefactoring.getVariableDeclaration() == null || extractVariableRefactoring.getVariableDeclaration().getInitializer() == null) {
                    continue;
                }
                //commit 信息

                RevCommit commit = null;
                try {
                    commit = walk.parseCommit(fRepo.resolve(commitId));
                } catch (IOException e) {
                    e.printStackTrace();
                    continue;
                }
                String beforeCommit = commit.getParent(0).getName();
                String afterCommit = commitId;

                String oldName = extractVariableRefactoring.getVariableDeclaration().getInitializer().toString();
                String newName = extractVariableRefactoring.getVariableDeclaration().getVariableName().toString();
                String type = extractVariableRefactoring.getVariableDeclaration().getType().toString();
                String originalFilePath = extractVariableRefactoring.leftSide().get(0).getFilePath();
                String refactoredFilePath = extractVariableRefactoring.rightSide().get(0).getFilePath();
                ArrayList<NodePosition> beforeArrayList = new ArrayList<>();
                List<CodeRange> leftSides = extractVariableRefactoring.leftSide();
                for (CodeRange codeRange : leftSides) {
                    if (!isOriginalRecord(codeRange.getDescription())) {
                        continue;
                    }
                    int startLine = codeRange.getStartLine();
                    int endLine = codeRange.getEndLine();
                    int startColumn = codeRange.getStartColumn();
                    int endColumn = codeRange.getEndColumn();
                    beforeArrayList.add(new NodePosition(startLine, startColumn, endLine, endColumn, -1));
                }

                ArrayList<NodePosition> afterArrayList = new ArrayList<>();
                List<CodeRange> rightSides = extractVariableRefactoring.rightSide();
                for (CodeRange codeRange : rightSides) {
                    String description = codeRange.getDescription();
                    if (!isOriginalRecord(description) && !isExtractVariableDeclaration(description)) {
                        continue;
                    }
                    int startLine = codeRange.getStartLine();
                    int endLine = codeRange.getEndLine();
                    int startColumn = codeRange.getStartColumn();
                    int endColumn = codeRange.getEndColumn();
                    NodePosition elemInfo = new NodePosition(startLine, startColumn, endLine, endColumn, -1);
                    afterArrayList.add(elemInfo);
                }

                Comparator<NodePosition> cmp = (o1, o2) -> {
                    if (o1.getStartLineNumber() != o2.getStartLineNumber()) {
                        return o1.getStartLineNumber() - o2.getStartLineNumber();
                    } else if (o1.getEndLineNumber() != o2.getEndLineNumber()) {
                        return o1.getEndLineNumber() - o2.getEndLineNumber();
                    } else {
                        return o1.getStartColumnNumber() - o2.getStartColumnNumber();
                    }
                };

                if (beforeArrayList.size() == 0 || afterArrayList.size() == 0) {
                    continue;
                }

                beforeArrayList.sort(cmp);
                afterArrayList.sort(cmp);

                RefactoringMinedData refactoringMinerData = new RefactoringMinedData(currentId++, fProjectName,
                        beforeCommit, afterCommit, originalFilePath, refactoredFilePath,
                        oldName, newName,
                        type,
                        beforeArrayList, afterArrayList);

                RefactoringMinedDataList.add(refactoringMinerData);
            }
        }
    }

    private static boolean isOriginalRecord(String s) {
        final String originalDecal1 = "statement with the initializer of the extracted variable";
        final String originalDecal2 = "statement with the name of the extracted variable";
        return originalDecal1.equals(s) || originalDecal2.equals(s);
    }

    private static boolean isExtractVariableDeclaration(String s) {
        final String originalDecal = "extracted variable declaration";
        return originalDecal.equals(s);
    }
}
