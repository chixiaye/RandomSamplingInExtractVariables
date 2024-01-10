package git;

import io.json.JSONReader;
import json.LabelData;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.jgit.api.CleanCommand;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ResetCommand;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.PersonIdent;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.revwalk.RevWalk;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import sample.Constants;
import utils.Utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

@Slf4j
public class GitUtils {

    public static void removeGitLock(String gitDirectoryPath) {
//        String lockFilePath = gitDirectoryPath  + ".git" + Constants.FILE_SEPARATOR_PROPERTY + "index";
//        File lockFile = new File(lockFilePath);
//        if(lockFile.exists()){
//            lockFile.delete();
//        }
//        lockFilePath = gitDirectoryPath +   ".git" + Constants.FILE_SEPARATOR_PROPERTY +"index.lock";
//        lockFile = new File(lockFilePath );
//        if(lockFile.exists()){
//            lockFile.delete();
//        }
//        lockFilePath = gitDirectoryPath +   ".git" + Constants.FILE_SEPARATOR_PROPERTY +"HEAD.lock";
//        lockFile = new File(lockFilePath );
//        if(lockFile.exists()){
//            lockFile.delete();
//        }
        deleteLockFiles(new File(gitDirectoryPath));
    }

    private static void deleteLockFiles(File file) {
        if (file.isDirectory()) {
            // 遍历目录中的所有文件和子目录
            for (File subFile : file.listFiles()) {
                deleteLockFiles(subFile);
            }
        } else {
            // 如果文件是以 .lock 结尾，则删除它
            if (file.getName().endsWith(".lock")) {
                file.delete();
            }
        }
    }

    public static String getCurrentSHA(String gitPath) { //获取当前commit的sha
        String sha = null;
        try (Git git = Git.open(new File(gitPath))) {
            Repository repository = git.getRepository();
            sha = repository.resolve("HEAD").getName();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sha;
    }

    public static String getLatestCommitSHA(String pathname,HashSet<String> invalidCommitIDHashSet) throws IOException, GitAPIException {

        ArrayList<File> list = new ArrayList<>();
        Utils.getFileList(list, Constants.LABELED_DATA_PATH, "json");
        HashSet<String> labeledCommitIDHashSet = new HashSet<>();
        for (int i = list.size() - 1; i >= 0; --i) {
            File file = list.get(i);
            String fileName = file.getName();
            String localName = fileName.substring(0, fileName.lastIndexOf("_"));
            if (pathname.equals(Constants.PREFIX_PATH + localName + System.getProperty("file.separator"))) {
                LabelData labelData = JSONReader.deserializeAsLabelData(file.getAbsolutePath());
                if (!invalidCommitIDHashSet.contains(labelData.getRefactoredCommitID())){
                    labeledCommitIDHashSet.add(labelData.getRefactoredCommitID());
                }
            }
        }

        // 打开本地Git仓库
        try (Repository repository = new FileRepositoryBuilder().setGitDir(new File(pathname + ".git")).build()) {
            // 初始化最新提交的SHA值和时间
            String latestCommitSha = null;
            long latestCommitTime = -1;
            RevWalk revWalk = new RevWalk(repository);
            for (String s : labeledCommitIDHashSet) {
                // 检查提交时间是否比当前最新时间更新
                // 获取指定 commit 的 RevCommit 对象
                RevCommit commit = revWalk.parseCommit(repository.resolve(s));

                // 获取提交时间
                PersonIdent committerIdent = commit.getCommitterIdent();
//                String commitTime = committerIdent.getWhen().toString();
                if (commit.getCommitTime() > latestCommitTime) {
                    latestCommitTime = commit.getCommitTime();
                    latestCommitSha = commit.getName();
                }
            }
//            System.out.println(invalidCommitIDHashSet.size()+ " " +  latestCommitTime + " " + latestCommitSha);
            if (latestCommitSha == null) {
                log.error(pathname+" latestCommitSha is null");
            }
            return latestCommitSha;
        }
    }

    public static void rollbackToCommit(String repoPath, String commitSHA) throws IOException, GitAPIException {
        // 打开本地git仓库

        try (Repository repo = Git.open(new File(repoPath)).getRepository()) {
            Git git = new Git(repo);
            ObjectId commitId = repo.resolve(commitSHA);
            if (commitId == null) {
                throw new IllegalArgumentException("Invalid commitSHA: " + commitSHA);
            }

            // for untracked
            CleanCommand cleanCommand = git.clean();
            cleanCommand.setForce(true);
            cleanCommand.setCleanDirectories(true);
            cleanCommand.call();

            // for tracked
            ResetCommand resetCommand = git.reset();
            resetCommand.setMode(ResetCommand.ResetType.HARD);
            resetCommand.setRef(commitId.getName());
            resetCommand.call();

//            System.out.println("Rollback to commit " + commitSHA + " is successful.");
        }
    }

    /**
     * @方法简介: 获取所有的版本号与版本号对应的时间戳
     */
    public static ArrayList<String> getGitVersion(File file) {
        Git git;
        Iterable<RevCommit> logIterable;
        try {
            git = Git.open(file);
            logIterable = git.log().call();
        } catch (Exception e) {
            return new ArrayList<>();
        }

        Iterator<RevCommit> logIterator = logIterable.iterator();//获取所有版本号的迭代器

        ArrayList<String> list = new ArrayList<>();
        while (logIterator.hasNext()) {

            RevCommit commit = logIterator.next();
            if (commit.getParentCount() > 1) { //如果是merge的提交，就跳过
                continue;
            }
            // 过滤掉 pull request
            if (commit.getShortMessage().startsWith("Merge pull request #")) {
                continue;
            }
//            Date commitDate = commit.getAuthorIdent().getWhen();     //提交时间
//            String commitPerson = commit.getAuthorIdent().getName() ;    //提交人
            String commitID = commit.getName();    //提交的版本号（之后根据这个版本号去获取对应的详情记录）

            list.add(commitID); //将版本号添加到list中

        }
        return list;
    }
}
