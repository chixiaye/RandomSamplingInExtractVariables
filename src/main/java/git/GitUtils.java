package git;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ResetCommand;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
@Slf4j
public class GitUtils {
    public static String getCurrentSHA(String gitPath){ //获取当前commit的sha
        String sha = null;
        try (Git git = Git.open(new File(gitPath))){
            Repository repository = git.getRepository();
            sha = repository.resolve("HEAD").getName();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sha;
    }

    public static String getLatestCommitSHA(String pathname,HashSet<String> invalidCommitIDHashSet) throws IOException, GitAPIException {
        // 打开本地Git仓库
        try (Repository repository = new FileRepositoryBuilder().setGitDir(new File(pathname+".git")).build()) {
            Git git = new Git(repository);
            // 获取所有分支的引用
            Map<String, Ref> refs = repository.getAllRefs();
            // 初始化最新提交的SHA值和时间
            String latestCommitSha = null;
            long latestCommitTime = -1;

            // 遍历所有分支的引用
            for (Map.Entry<String, Ref> entry : refs.entrySet()) {
                Ref ref = entry.getValue();
                ObjectId objectId = ref.getObjectId();
                // 获取提交对象
                RevCommit commit = git.log().add(objectId).setMaxCount(1).call().iterator().next();

                // 检查提交时间是否比当前最新时间更新
                if (!invalidCommitIDHashSet.contains(commit.getName()) && commit.getCommitTime() > latestCommitTime) {
                    latestCommitTime = commit.getCommitTime();
                    latestCommitSha = commit.getName();
                }
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

            ResetCommand resetCommand = git.reset();
            resetCommand.setMode(ResetCommand.ResetType.HARD);
            resetCommand.setRef(commitId.getName());
            resetCommand.call();
//            System.out.println("Rollback to commit " + commitSHA + " is successful.");
        }
    }
}
