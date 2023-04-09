package miner;

import json.CurrentLineData;
import json.MetaData;
import json.utils.NodePosition;
import json.ParentData;
import org.eclipse.jdt.core.dom.*;

import java.util.*;


public class ExpressionVisitor extends ASTVisitor {
    HashMap<String,ArrayList<MetaData>> recordMap;
    HashMap<MetaData,ASTNode  > metaDataASTNodeHashMap;
//    HashMap<String,ArrayList<MetaData>> nodeMap ;
    private CompilationUnit fCU;

    public ExpressionVisitor(CompilationUnit fCU) {
        this.fCU = fCU;
        this.recordMap = new HashMap<>();
        this.metaDataASTNodeHashMap = new HashMap<>();
    }

    public void loadMetaData(MetaData metaData){
        ASTNode node = this.metaDataASTNodeHashMap.get(metaData);
        metaData.setNodeType(ASTNode.nodeClassForType(node.getNodeType()).getSimpleName());
        metaData.countASTNodeComplexity(node);
        metaData.setTokenLength();

        ArrayList<ASTNode> parentNodes = getAllSuperNodes(node);
        ArrayList<ParentData> parentDataList = new ArrayList<>();
        for (ASTNode n : parentNodes) {
            int offset = n.getStartPosition();
            int length = n.getLength();
            NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                    , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);

            ParentData data = new ParentData(n.toString(), ASTNode.nodeClassForType(n.getNodeType()).getSimpleName(),
                    n.getLocationInParent().toString(),pos);
            data.countASTNodeComplexity(n);
            parentDataList.add(data);
        }
        metaData.setParentDataList(parentDataList);
        int currentLineContextIndex = Math.min(findCurrentLineContextIndex(node), parentDataList.size()-1);
        ParentData parentData = parentDataList.get(currentLineContextIndex);
        CurrentLineData currentLineData =  new CurrentLineData(parentData.getNodeContext(),parentData.getNodeType(),
                parentData.getLocationInParent(),parentData.getNodePosition());
        currentLineData.setTokenLength();
        currentLineData.countASTNodeComplexity(parentNodes.get(currentLineContextIndex));
        metaData.setCurrentLineData(currentLineData);
    }


    @Override
    public boolean preVisit2(ASTNode node) {
        if (node instanceof SimpleName || node instanceof NumberLiteral || node instanceof NullLiteral
                || node instanceof TypeLiteral || node instanceof BooleanLiteral || node instanceof StringLiteral
                || node instanceof CharacterLiteral || node instanceof  ArrayInitializer) {
            return false;
        }
        else if (node instanceof Expression || node instanceof Name) {
            if (canReplace(node) ) {
                MetaData metaData;
                int offset = node.getStartPosition();
                int length = node.getLength();
                NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                            , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
                metaData = new MetaData(pos, node.toString());//ASTNode.nodeClassForType(node.getNodeType()).getName()
                metaDataASTNodeHashMap.put(metaData,node);
                if(recordMap.get(node.toString())!=null){
                    recordMap.get(node.toString()).add(metaData);
                }
                else{
                    ArrayList<MetaData> list = new ArrayList<>();
                    list.add(metaData);
                    recordMap.put(node.toString(),list);
                }
            }
        }
        return super.preVisit2(node);
    }

    private int findCurrentLineContextIndex(ASTNode node){
        final int offset = node.getStartPosition();
        final int length = node.getLength();
        final NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
        int index=0;
        int res=0;
        ASTNode parent = node.getParent();
        while(parent!=null){
            int parentOffset = parent.getStartPosition();
            int parentLength = parent.getLength();
            NodePosition parentPos = new NodePosition(fCU.getLineNumber(parentOffset), fCU.getColumnNumber(parentOffset)
                    , fCU.getLineNumber(parentOffset + parentLength), fCU.getColumnNumber(parentOffset + parentLength), parentLength);
            if( parentPos.getStartLineNumber()==pos.getStartLineNumber()
                    && parentPos.getEndLineNumber()==pos.getEndLineNumber()){
                res=index;
            }
            index++;
            parent=parent.getParent();
        }
        return res;
    }

    private ArrayList<ASTNode> getAllSuperNodes(ASTNode node) {
        ArrayList<ASTNode> list = new ArrayList<>();

        while (node.getParent() != null) {
            node=node.getParent();
            list.add(node);
            if (node instanceof MethodDeclaration || node instanceof Initializer || node instanceof LambdaExpression) {
                break;
            }
        }
        return list;
    }

    private final class ForStatementChecker extends ASTVisitor {

        private final Collection<String> fForInitializerVariables;

        private boolean fReferringToForVariable = false;

        public ForStatementChecker(Collection<String> forInitializerVariables) {
            fForInitializerVariables = forInitializerVariables;
        }

        public boolean isReferringToForVariable() {
            return fReferringToForVariable;
        }

        @Override
        public boolean visit(SimpleName node) {
            if (fForInitializerVariables.contains(node.toString())) {
                fReferringToForVariable = true;
            }
            return false;
        }
    }

    private  boolean canReplace(ASTNode node) {
        if(node instanceof  MethodInvocation){
            MethodInvocation mi=(MethodInvocation)node;
            // if binding is null, it means that the method is not resolved
            if(mi.resolveMethodBinding()==null || mi.resolveMethodBinding().getReturnType().getName().equals("void") ){
                return  false;
            }
        }
        ASTNode parent = node.getParent();
        if (parent instanceof VariableDeclarationFragment) {
            VariableDeclarationFragment vdf = (VariableDeclarationFragment) parent;
            if (node.equals(vdf.getName()))
                return false;
        }
        if (parent instanceof Statement && node instanceof MethodInvocation)
            return false;
        if (getEnclosingBodyNode(node) == null)
            return false;
        if (isMethodParameter(node))
            return false;
        if (isThrowableInCatchBlock(node))
            return false;
        if (parent instanceof ExpressionStatement)
            return false;
        if (parent instanceof LambdaExpression)
            return false;
        if (isLeftValue(node))
            return false;
        if (isReferringToLocalVariableFromFor((Expression) node))
            return false;
        if (isUsedInForInitializerOrUpdater((Expression) node))
            return false;
        if (parent instanceof SwitchCase)
            return true;
        if (node instanceof SimpleName && node.getLocationInParent() != null) {
            return !"name".equals(node.getLocationInParent().getId()); //$NON-NLS-1$
        }
        return true;
    }

    private boolean isMethodParameter(ASTNode node) {
        return (node instanceof SimpleName) && (node.getParent() instanceof SingleVariableDeclaration) && (node.getParent().getParent() instanceof MethodDeclaration);
    }

    private boolean isReferringToLocalVariableFromFor(Expression expression) {
        ASTNode current = expression;
        ASTNode parent = current.getParent();

        while (parent != null && !(parent instanceof BodyDeclaration)) {
            if (parent instanceof ForStatement) {
                ForStatement forStmt = (ForStatement) parent;
                if (forStmt.initializers().contains(current) || forStmt.updaters().contains(current) || forStmt.getExpression() == current) {
                    List<Expression> initializers = forStmt.initializers();
                    if (initializers.size() == 1 && initializers.get(0) instanceof VariableDeclarationExpression) {
                        List<String> forInitializerVariables = getForInitializedVariables((VariableDeclarationExpression) initializers.get(0));
                        ForStatementChecker checker = new ForStatementChecker(forInitializerVariables);
                        expression.accept(checker);
                        if (checker.isReferringToForVariable())
                            return true;
                    }
                }
            }
            current = parent;
            parent = current.getParent();
        }
        return false;
    }

    private boolean isThrowableInCatchBlock(ASTNode node) {
        return (node instanceof SimpleName) && (node.getParent() instanceof SingleVariableDeclaration) && (node.getParent().getParent() instanceof CatchClause);
    }

    private boolean isUsedInForInitializerOrUpdater(Expression expression) {
        ASTNode parent = expression.getParent();
        if (parent instanceof ForStatement) {
            ForStatement forStmt = (ForStatement) parent;
            return forStmt.initializers().contains(expression) || forStmt.updaters().contains(expression);
        }
        return false;
    }

    private boolean isLeftValue(ASTNode node) {
        ASTNode parent = node.getParent();
        if (parent instanceof Assignment) {
            Assignment assignment = (Assignment) parent;
            if (assignment.getLeftHandSide() == node)
                return true;
        }
        if (parent instanceof PostfixExpression)
            return true;
        if (parent instanceof PrefixExpression) {
            PrefixExpression.Operator op = ((PrefixExpression) parent).getOperator();
            if (op.equals(PrefixExpression.Operator.DECREMENT))
                return true;
            if (op.equals(PrefixExpression.Operator.INCREMENT))
                return true;
            return false;
        }
        return false;
    }

    // return List<IVariableBinding>
    private List<String> getForInitializedVariables(VariableDeclarationExpression variableDeclarations) {
        List<String> forInitializerVariables = new ArrayList<>(1);
        for (Iterator<VariableDeclarationFragment> iter = variableDeclarations.fragments().iterator(); iter.hasNext(); ) {
            VariableDeclarationFragment fragment = iter.next();
            forInitializerVariables.add(fragment.toString());
        }
        return forInitializerVariables;
    }

    private ASTNode getEnclosingBodyNode(ASTNode node) {

        // expression must be in a method, lambda or initializer body
        // make sure it is not in method or parameter annotation
        StructuralPropertyDescriptor location = null;
        while (node != null && !(node instanceof BodyDeclaration)) {
            location = node.getLocationInParent();
            node = node.getParent();
            if (node instanceof LambdaExpression) {
                break;
            }
        }
        if (location == MethodDeclaration.BODY_PROPERTY || location == Initializer.BODY_PROPERTY
                || (location == LambdaExpression.BODY_PROPERTY && ((LambdaExpression) node).resolveMethodBinding() != null)) {
            return (ASTNode) node.getStructuralProperty(location);
        }
        return null;
    }
}
