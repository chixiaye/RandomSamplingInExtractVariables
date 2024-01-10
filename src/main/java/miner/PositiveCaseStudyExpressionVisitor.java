package miner;

import json.CurrentLineData;
import json.MetaData;
import json.OriginalExpression;
import json.ParentData;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;


public class PositiveCaseStudyExpressionVisitor extends AbstractExpressionVisitor {
    @Getter
    @Setter
    private ArrayList<MetaData> metaDataList;

    private Set<OriginalExpression> originalExpressions;
    private String fSearchedNodeContext;


    public PositiveCaseStudyExpressionVisitor(CompilationUnit cu, String node, OriginalExpression[] originalExpressions) {
        super(cu);
        this.metaDataList = new ArrayList<>();
        this.originalExpressions = new HashSet<>(Arrays.asList(originalExpressions));  ;
        this.fSearchedNodeContext = node;
    }

    public void reLoadMetaData(MetaData metaData, ASTNode node) {
        metaData.setNodeType(ASTNode.nodeClassForType(node.getNodeType()).getSimpleName());
        metaData.countASTNodeComplexity(node);
        metaData.setTokenLength();

        ArrayList<ASTNode> parentNodes = getAllSuperNodes(node);
        ArrayList<ParentData> parentDataList = new ArrayList<>();
        for (int i = 0; i < parentNodes.size(); i++) {
            ASTNode n = parentNodes.get(i);
            int offset = n.getStartPosition();
            int length = n.getLength();
            NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                    , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);

            ParentData data = new ParentData(n.toString(), ASTNode.nodeClassForType(n.getNodeType()).getSimpleName(),
                    n.getLocationInParent().toString(), pos);
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
        if (isSearchNode(node)) {
            int offset = node.getStartPosition();
            int length = node.getLength();
            NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                    , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
            MetaData metaData = new MetaData(pos, node.toString(), getExpressionType(node));//ASTNode.nodeClassForType(node.getNodeType()).getName()
            reLoadMetaData(metaData, node);
            metaDataList.add(metaData);
        }
        return super.preVisit2(node);
    }

    private boolean isSearchNode(ASTNode node) {
        if(!node.toString().equals(fSearchedNodeContext)){
            return false;
        }
        ASTNode parent = node.getParent();
        // 节点所在的方法体是否和重构表达式在同一个方法体内
        while (parent != null) {
            if (parent instanceof MethodDeclaration  ) {
                break;
            }
            int startLineNumber = fCU.getLineNumber(parent.getStartPosition());
            int endLineNumber = fCU.getLineNumber(parent.getStartPosition() + parent.getLength());
            for (OriginalExpression originalExpression : originalExpressions) {
                if ( startLineNumber <= originalExpression.getLine() && endLineNumber >= originalExpression.getLine()) {
                    return true;
                }
            }
            parent = parent.getParent();
        }
        return false;
    }

}
