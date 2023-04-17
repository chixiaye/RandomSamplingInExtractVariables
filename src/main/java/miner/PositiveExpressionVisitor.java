package miner;

import json.CurrentLineData;
import json.MetaData;
import json.ParentData;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;
import java.util.HashMap;


public class PositiveExpressionVisitor extends AbstractExpressionVisitor  {
    @Getter
    @Setter
    private ArrayList<MetaData>  metaDataList;
    HashMap<MetaData,ASTNode  > metaDataASTNodeHashMap;

    private NodePosition fNodePosition;
    private String fSearchedNode;

    public PositiveExpressionVisitor(CompilationUnit cu, String node, NodePosition nodePosition) {
        super(cu);
        this.metaDataList =  new ArrayList<>();
        this.metaDataASTNodeHashMap = new HashMap<>();
        this.fNodePosition = nodePosition;
        this.fSearchedNode=node;
    }

    public void reLoadMetaData(MetaData metaData ){
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
        if ( node instanceof SimpleName && isSearchNode(node)) {
            MetaData metaData;
            int offset = node.getStartPosition();
            int length = node.getLength();
            NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                    , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
            metaData = new MetaData(pos, node.toString(),getExpressionType(node));//ASTNode.nodeClassForType(node.getNodeType()).getName()
            metaDataASTNodeHashMap.put(metaData,node);
            metaDataList.add(metaData);
        }
        return super.preVisit2(node);
    }

    private boolean isSearchNode(ASTNode node){
        if(node.toString().equals(fSearchedNode)){
            ASTNode parent = node.getParent();
            while (parent!=null){
                if(parent instanceof MethodDeclaration || parent instanceof Initializer || parent instanceof LambdaExpression){
                    break;
                }
                parent=parent.getParent();
            }
            if(parent==null){
                return false;
            }
            int offset = parent.getStartPosition();
            int length = parent.getLength();
            NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                    , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
            if(pos.getStartLineNumber()<= fNodePosition.getStartLineNumber()
                    && pos.getEndLineNumber()>=fNodePosition.getEndLineNumber()){
                return true;
            }
        }
        return false;
    }

    public void reLoadMetaData(MetaData metaData, Expression initializer) {
        metaData.setNodeContext(initializer.toString());
        int offset = initializer.getStartPosition();
        int length = initializer.getLength();
        NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
        metaData.setNodePosition(pos);
        metaData.setNodeType(ASTNode.nodeClassForType(initializer.getNodeType()).getSimpleName());
        metaData.countASTNodeComplexity(initializer);
        metaData.setTokenLength();
    }
}
