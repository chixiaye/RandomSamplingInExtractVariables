package ast;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;
@Slf4j
public class LightASTParser {
    private CompilationUnit compilationUnit;

    public LightASTParser(char[] source) {
        ASTParser astParser = ASTParser.newParser(AST.getJLSLatest());
        astParser.setKind(ASTParser.K_COMPILATION_UNIT);
        astParser.setSource(source);
        astParser.setStatementsRecovery(true);
        astParser.setResolveBindings(true);
        astParser.setResolveBindings(true);

//        Map<String, String> compilerOptions = JavaCore.getOptions();
//        compilerOptions.put(JavaCore.COMPILER_COMPLIANCE, JavaCore.latestSupportedJavaVersion());
//        compilerOptions.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, JavaCore.latestSupportedJavaVersion());
//        compilerOptions.put(JavaCore.COMPILER_SOURCE, JavaCore.latestSupportedJavaVersion());
//        astParser.setCompilerOptions(compilerOptions);
        try{
            this.compilationUnit =  (CompilationUnit) (astParser.createAST(null));
        }catch (Exception e){
            log.error("Error in parsing file: " + e.getLocalizedMessage());
        }
    }

    public CompilationUnit getCompilationUnit() {
        return this.compilationUnit;
    }


}