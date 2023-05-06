package refactoringminer.handler;

import git.GitUtils;
import io.json.JsonFileSplitter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jgit.lib.Repository;
import org.refactoringminer.api.GitService;
import org.refactoringminer.rm1.GitHistoryRefactoringMinerImpl;
import org.refactoringminer.util.GitServiceImpl;
import refactoringminer.json.RefactoringMinedData;
import sample.Constants;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;

@Slf4j
public class RefactoringMinerThread extends Thread {
    private String fProjectName;
    private int fLimitTime;
    private String fOutputDict;
    protected JsonFileSplitter fJsonFileSplitter;


    public RefactoringMinerThread(String name) {
        fProjectName = name;
        fJsonFileSplitter = new JsonFileSplitter();
        String gitPath = Constants.PREFIX_PATH + fProjectName + System.getProperty("file.separator");
        GitUtils.removeGitLock(gitPath);

    }

    @Override
    public void run() {
        File file = new File(Constants.PREFIX_PATH + Constants.FILE_SEPARATOR_PROPERTY + fProjectName);
        if (!file.exists() || !file.isDirectory()) {
            return;
        }
        log.info("start analyzing {} ... ", fProjectName);

        ArrayList<String> commits = GitUtils.getGitVersion(file);
        GitService gitService = new GitServiceImpl();
        GitHistoryRefactoringMinerImpl gitHistoryRefactoringMiner = new GitHistoryRefactoringMinerImpl();
        int currentID = 1;
        ArrayList<RefactoringMinedData> result = new ArrayList<>();
        try (Repository repo = gitService.openRepository(file.getAbsolutePath())) {
            for (String commit : commits) {
                MyRefactoringHandler myRefactoringHandler = new MyRefactoringHandler(repo, fProjectName, currentID);
                // 不要用有超时选项的代码，他的并发有问题
                myDetectAtCommit(gitHistoryRefactoringMiner, commit, repo, myRefactoringHandler, Constants.LIMIT_MILLISECOND);

                ArrayList<RefactoringMinedData> list = myRefactoringHandler.getRefactoringMinedDataList();

                result.addAll(list);
                currentID += list.size();
            }
        } catch (Exception e) {
        }

        fJsonFileSplitter.writeJsonArray(result, fProjectName);
    }

    private void myDetectAtCommit(GitHistoryRefactoringMinerImpl gitHistoryRefactoringMiner, String commit, Repository repo, MyRefactoringHandler myRefactoringHandler,
                                  int timeout) {
        ExecutorService service = Executors.newSingleThreadExecutor();
        Future<?> f = null;
        try {
            Runnable r = () -> gitHistoryRefactoringMiner.detectAtCommit(repo, commit, myRefactoringHandler);
            f = service.submit(r);
            f.get(timeout, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            f.cancel(true);
        } catch (ExecutionException e) {
//            e.printStackTrace();
        } catch (InterruptedException e) {
//            e.printStackTrace();
        } finally {
            service.shutdown();
        }
    }


//    private List<ExtractVariableRecord>  myDetectAtCommit(String path,String commitId,int limitTime)  {
//        GitService gitService = new GitServiceImpl();
//
//        List<ExtractVariableRecord> extractVariableRecordList=new ArrayList<>();
//        try {
//            Repository repo = gitService.openRepository(path);
//            new GitHistoryRefactoringMinerImpl().detectAtCommit(repo, commitId, new RefactoringHandler() {
//                @Override
//                public void handle(String commitId, List<Refactoring> refactorings) {
//                    for (Refactoring refactoring : refactorings) {
//                        if (refactoring instanceof ExtractVariableRefactoring) {
//                            ExtractVariableRefactoring extractVariableRefactoring = (ExtractVariableRefactoring) refactoring;
//                            PlaceRecord variablePlace = null;
//                            //commit 信息
//                            RevWalk walk = new RevWalk(repo);
//                            RevCommit commit = null;
//                            try {
//                                commit = walk.parseCommit(repo.resolve(commitId));
//                            } catch (IOException e) {
//                                e.printStackTrace();
//                                continue;
//                            }
//                            String beforeCommit = commit.getParent(0).getName();
//                            String afterCommit = commitId;
//                            //初始化为空的时候会抛出异常，这种我们本来就要过滤
//                            if (extractVariableRefactoring.getVariableDeclaration() == null || extractVariableRefactoring.getVariableDeclaration().getInitializer() == null) {
//                                continue;
//                            }
//                            String oldName = extractVariableRefactoring.getVariableDeclaration().getInitializer().toString();
//                            String newName = extractVariableRefactoring.getVariableDeclaration().getVariableName().toString();
//                            String type = extractVariableRefactoring.getVariableDeclaration().getType().toString();
//                            String leftFilePath = extractVariableRefactoring.leftSide().get(0).getFilePath();
//                            String rightFilePath = extractVariableRefactoring.rightSide().get(0).getFilePath();
//                            String filePath = leftFilePath.equals(rightFilePath) ? leftFilePath : null;
//                            //如果出现跨文件重构，则不处理
//                            if (filePath == null) {
//                                continue;
//                            }
//                            Set<PlaceRecord> beforeSet = new HashSet<>();
//                            List<CodeRange> leftSides = extractVariableRefactoring.leftSide();
//                            for (CodeRange codeRange : leftSides) {
//                                if (!isOriginalRecord(codeRange.getDescription())) {
//                                    continue;
//                                }
//                                int startLine = codeRange.getStartLine();
//                                int endLine = codeRange.getEndLine();
//                                int startColumn = codeRange.getStartColumn();
//                                int endColumn = codeRange.getEndColumn();
//                                beforeSet.add(new PlaceRecord(startLine, endLine, startColumn, endColumn));
//                            }
//                            Set<PlaceRecord> afterSet = new HashSet<>();
//                            List<CodeRange> rightSides = extractVariableRefactoring.rightSide();
//                            for (CodeRange codeRange : rightSides) {
//                                String description = codeRange.getDescription();
//                                if (!isOriginalRecord(description) && !isExtractVariableDeclaration(description)) {
//                                    continue;
//                                }
//                                int startLine = codeRange.getStartLine();
//                                int endLine = codeRange.getEndLine();
//                                int startColumn = codeRange.getStartColumn();
//                                int endColumn = codeRange.getEndColumn();
//                                PlaceRecord elemInfo = new PlaceRecord(startLine, endLine, startColumn, endColumn);
//                                if (isExtractVariableDeclaration(description)) {
//                                    variablePlace = elemInfo;
//                                } else {
//                                    afterSet.add(elemInfo);
//                                }
//                            }
//                            //过滤一下，不要对针对test部分重构以及只对一处进行提取变量重构的commit
//                            //不要对null做提取的、不要没有.或者[]
//                            //过滤lambda表达式,即->这种情况
////                            if( oldName.equals("null") ){
////                                continue;
////                            }else
//                            {
////                                System.out.println("you");
//                                ExtractVariableRecord extractVariableRecord = new ExtractVariableRecord(
//                                        beforeCommit, afterCommit, oldName, newName, type, filePath, variablePlace, beforeSet, afterSet);
//                                extractVariableRecordList.add(extractVariableRecord);
//                            }
//
//                        }
//                    }
//                }
//            },limitTime);
//            repo.close();
//        } catch (Exception e) {
//        }
//
//        return extractVariableRecordList;
//    }
}