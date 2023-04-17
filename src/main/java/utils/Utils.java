package utils;

import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;

import java.io.*;
import java.util.ArrayList;
import java.util.Map;

public class Utils {

    public static char[] readJavaFile(File file){
        if(!file.exists()||!file.isFile() )
            return null;
        StringBuffer sb=new StringBuffer();
        try {
            BufferedReader in = new BufferedReader(new FileReader(file));
            String str;
            while ((str = in.readLine()) != null) {
                sb.append(str+"\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sb.toString().toCharArray();
    }

    public static void getFileList(ArrayList<File> arrayList, String strPath,String extension) {
        File fileDir = new File(strPath);
        if(!extension.startsWith("."))
            extension="."+extension;
        if (null != fileDir && fileDir.isDirectory()) {
            File[] files = fileDir.listFiles();
            if (null != files) {
                for (int i = 0; i < files.length; i++) {
                    // 如果是文件夹 继续读取
                    if (files[i].isDirectory()) {
                        getFileList(arrayList, files[i].getPath(), extension);
                    } else {
                        String strFileName = files[i].getPath();
                        if (files[i].exists() && (strFileName.endsWith( extension) )) {
                            arrayList.add(files[i]);
                        }
                    }
                }
            } else {
                if (null != fileDir) {
                    String strFileName = fileDir.getPath();
                    if (fileDir.exists() && (strFileName.endsWith( extension))) {
                        arrayList.add(fileDir);
                    }
                }
            }
        }
    }


    public static ASTParser getNewASTParser() {
        ASTParser astParser;
        astParser = ASTParser.newParser(AST.getJLSLatest());
        astParser.setKind(ASTParser.K_COMPILATION_UNIT);
        return astParser;
    }

    public static ASTParser getNewASTParser(String[] sourcepathEntries, String[] encodings) {
        ASTParser astParser;
        astParser = ASTParser.newParser(AST.getJLSLatest());
        astParser.setKind(ASTParser.K_COMPILATION_UNIT);
//        astParser.setEnvironment(null, sourcepathEntries, null,true);
        astParser.setEnvironment(null, sourcepathEntries, encodings, true);
        astParser.setResolveBindings(true);
        astParser.setStatementsRecovery(true);
        astParser.setBindingsRecovery(true);
        astParser.setUnitName("");
        Map options = JavaCore.getOptions();
        astParser.setCompilerOptions(options);
        return astParser;
    }

    static public String getCodeFromFile(File javaFile) throws IOException {
        BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(javaFile));
        byte[] input = new byte[bufferedInputStream.available()];
        bufferedInputStream.read(input);
        bufferedInputStream.close();
        return new String(input);
    }

}
