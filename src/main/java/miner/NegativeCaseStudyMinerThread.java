package miner;

import ast.ProjectsParser;
import git.GitUtils;
import io.json.JsonFileSplitter;
import json.*;
import json.utils.NodePosition;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jgit.api.errors.GitAPIException;
import sample.Constants;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Slf4j
public class PositiveCaseStudyMinerThread extends AbstractMinerThread {
    List<CaseStudy> fCaseStudyList;

    public PositiveCaseStudyMinerThread(String projectName, List<CaseStudy> caseStudyList) {
        super(projectName, caseStudyList.size());
        fCaseStudyList = caseStudyList;
    }

    @Override
    public void init(String projectName) {
        fJsonFileSplitter = new JsonFileSplitter();
        fProjectName = projectName;
        fFileList = new ArrayList<>();

    }

    @Override
    public void analyzeProject() throws Exception {
        for (int i = 0; i < fCaseStudyList.size(); i++) {
            CaseStudy caseStudy = fCaseStudyList.get(i);
            String gitPath = Constants.CASE_STUDY_PREFIX_PATH + fProjectName + System.getProperty("file.separator");
            Path projectPath = Paths.get(gitPath);
            String commitID = caseStudy.getOriginalCommitId();
            String filePath = caseStudy.getPath();
            String name = caseStudy.getOriginalName();
            log.info("extracting features from  {}-{}", fProjectName, (i+1));
            fCommitID = commitID;
            OriginalExpression originalExpression = caseStudy.getOriginalExpressionList()[0];
            NodePosition nodePosition = new NodePosition(originalExpression.getLine(),
                    originalExpression.getColumn(),originalExpression.getLine(),
                    originalExpression.getLine()+ name.length(),name.length());
            System.out.println(nodePosition);
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

            PositiveCaseStudyExpressionVisitor visitor = new PositiveCaseStudyExpressionVisitor(cu, name, caseStudy.getOriginalExpressionList());
            cu.accept(visitor);
            ArrayList<MetaData> metaDataList = visitor.getMetaDataList();

            EVRecord r = new EVRecord();
            r.setProjectName(fProjectName);
            r.setExpression(name);
            r.setId(i+1);
            r.setCommitID(fCommitID);
            r.setFilePath(filePath);
            assert metaDataList.size() == caseStudy.getOriginalExpressionList().length;
            r.setOccurrences(metaDataList.size());
            r.setExpressionList(metaDataList);
            r.generatePositionList(metaDataList);
            r.setIsGetTypeMethod(visitor.getTypeMethodState());
            r.setIsArithmeticExpression(visitor.getArithmeticExpressionState());

            r.initLayoutRelationDataListInit();
            fJsonFileSplitter.writeJsonArrayInCaseStudy(r, true);
//            System.out.println(r);
//            break;
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
