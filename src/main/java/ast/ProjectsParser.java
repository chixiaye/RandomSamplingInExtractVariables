package ast;

import lombok.Getter;
import lombok.Setter;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;

import utils.Utils;

public class ProjectsParser {
    @Getter
    @Setter
    protected HashSet<String> targetJavaFiles;
    protected ArrayList<String> allJavaFiles;
    protected String[] sourcetreeEntries;
    protected String[] encodings;
    protected String[] classpathEntries;

    private Path[] targetPaths;
    private ArrayList<String> classpathEntriesList;
    private Path repoPath;

    public ProjectsParser(Path[] targetPaths, Path projectPath, Path repoPath) throws IOException {
        this.targetPaths = targetPaths;
        this.repoPath = repoPath;
        targetJavaFiles = new HashSet<>();
        allJavaFiles = new ArrayList<>();
        classpathEntriesList = new ArrayList<>();
        traverseFile(projectPath.toFile());
        classpathEntries = classpathEntriesList.toArray(new String[classpathEntriesList.size()]);
        parseSourceEntries();
    }

    private void traverseFile(File root) {
        if (root.isFile()) {
            if (root.getName().endsWith(".java")) {
                allJavaFiles.add(root.getAbsolutePath());
                for (Path targetPath : targetPaths)
                    if (root.getAbsolutePath().startsWith(targetPath.toString())) {
                        targetJavaFiles.add(root.getAbsolutePath());
                        break;
                    }
            }
//            if (root.getName().equals("pom.xml")) {
//                classpathEntriesList.addAll(parsePOM(root.getAbsolutePath()));
//            }
            return;
        } else if (root.isDirectory()) {
            for (File f : root.listFiles())
                traverseFile(f);
        }
    }


    private void parseSourceEntries() throws IOException {
        HashSet<String> sourceRootSet = new HashSet<String>();
        for (String javaFile : allJavaFiles) {
            ASTParser astParser = Utils.getNewASTParser();
            String code = Utils.getCodeFromFile(new File(javaFile));
            astParser.setSource(code.toCharArray());
            CompilationUnit compilationUnit = (CompilationUnit) astParser.createAST(null);
            if (compilationUnit.getPackage() == null)
                continue;
            try {
                String rootPath = parseRootPath(javaFile, compilationUnit.getPackage().getName().toString());
                if (!rootPath.equals("")) sourceRootSet.add(rootPath);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        sourcetreeEntries = new String[sourceRootSet.size()];
        encodings = new String[sourceRootSet.size()];
        int index = 0;
        for (String sourceRoot : sourceRootSet) {
            sourcetreeEntries[index] = sourceRoot;
            encodings[index] = "utf-8";
            index++;
        }
    }

    private ArrayList<String> parsePOM(String pomPath) {
        ArrayList<String> tempClasspathEntriesList = new ArrayList<String>();
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(pomPath);
            NodeList dependencies = document.getElementsByTagName("dependency");
            for (int i = 0; i < dependencies.getLength(); i++) {
                Node dependency = dependencies.item(i);
                String groupId = "";
                String artifactId = "";
                String version = "";
                NodeList childNodes = dependency.getChildNodes();
                for (int j = 0; j < childNodes.getLength(); j++) {
                    Node childNode = childNodes.item(j);
                    if (childNode.getNodeName().equals("groupId"))
                        groupId = childNode.getTextContent();
                    else if (childNode.getNodeName().equals("artifactId"))
                        artifactId = childNode.getTextContent();
                    else if (childNode.getNodeName().equals("version"))
                        version = childNode.getTextContent();
                }
                if (groupId.equals("") || artifactId.equals("") || version.equals("")) continue;
                groupId = groupId.replaceAll("\\.", "\\\\");
                Path classEntry = repoPath.resolve(groupId);
                classEntry = classEntry.resolve(artifactId);
                classEntry = classEntry.resolve(version);
                //String classEntry = repoPath + groupId + "\\" + artifactId + "\\" + version;
                File classEntryFolder = classEntry.toFile();
                if (classEntryFolder.exists()) tempClasspathEntriesList.add(classEntry.toString());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return tempClasspathEntriesList;
    }

    private String parseRootPath(String filePath, String packageName) {
        String t = packageName.replaceAll("\\.", "\\\\");
        Path relativePath = Paths.get(t);
        Path absolutePath = Paths.get(filePath).resolveSibling("");
        int end = absolutePath.toString().lastIndexOf(relativePath.toString());
        if (end == -1) return "";
        return absolutePath.toString().substring(0, end);
    }

    public CompilationUnit parse(String path) throws IOException{
        ASTParser astParser = Utils.getNewASTParser(sourcetreeEntries, encodings);
        String code = Utils.getCodeFromFile(new File(path));
        astParser.setSource(code.toCharArray());
        CompilationUnit cu = (CompilationUnit) astParser.createAST(null);
        return cu;
    }

    public String[] getSourcetreeEntries() {
        return sourcetreeEntries;
    }

    public String[] getEncodings() {
        return encodings;
    }

    public String[] getClasspathEntries() {
        return classpathEntries;
    }
}
