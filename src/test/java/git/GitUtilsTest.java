package git;

import junit.framework.TestCase;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.junit.Test;

import java.io.IOException;
import java.util.HashSet;

public class GitUtilsTest extends TestCase {

    @Test
    public void testGetLatestSHA() throws IOException, GitAPIException {
        String gitPath = "D:\\Top1K\\dataset\\unofficial-openjdk@openjdk\\";
        String sha = GitUtils.getLatestCommitSHA(gitPath,new HashSet<>());
        System.out.println(sha);
    }
}