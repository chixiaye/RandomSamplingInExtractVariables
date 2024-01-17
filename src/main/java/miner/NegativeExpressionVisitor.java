package miner;

import json.CurrentLineData;
import json.MetaData;
import json.ParentData;
import json.utils.NodePosition;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.Expression;
import org.eclipse.jdt.core.dom.Name;

import java.util.ArrayList;
import java.util.HashMap;


public class NegativeExpressionVisitor extends AbstractExpressionVisitor  {
    HashMap<String,ArrayList<MetaData>> recordMap;
    HashMap<MetaData,ASTNode > metaDataASTNodeHashMap;

    public NegativeExpressionVisitor(CompilationUnit cu) {
        super(cu);
        this.recordMap = new HashMap<>();
        this.metaDataASTNodeHashMap = new HashMap<>();
    }


    public void loadMetaData(MetaData metaData) {
        ASTNode node = this.metaDataASTNodeHashMap.get(metaData);
        metaData.setNodeType(ASTNode.nodeClassForType(node.getNodeType()).getSimpleName());
        metaData.countASTNodeComplexity(node);
        metaData.setTokenLength();
        if (isArithmetic(node))
            this.setArithmeticExpressionState(1);

        if (isStartWithGet(node))
            this.setTypeMethodState(1);

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
//        int currentLineContextIndex = Math.min(findCurrentLineContextIndex(node), parentDataList.size()-1);
//        ParentData parentData = parentDataList.get(currentLineContextIndex);
//        CurrentLineData currentLineData =  new CurrentLineData(parentData.getNodeContext(),parentData.getNodeType(),
//                parentData.getLocationInParent(),parentData.getNodePosition());
//        currentLineData.setTokenLength();
//        currentLineData.countASTNodeComplexity(parentNodes.get(currentLineContextIndex));
//        metaData.setCurrentLineData(currentLineData);
    }


    @Override
    public boolean preVisit2(ASTNode node) {
        if (node instanceof Expression || node instanceof Name) {
            if (canReplace(node, true)) {
                MetaData metaData;
                int offset = node.getStartPosition();
                int length = node.getLength();
                NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                        , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
                metaData = new MetaData(pos, node.toString(), getExpressionType(node));//ASTNode.nodeClassForType(node.getNodeType()).getName()
                metaDataASTNodeHashMap.put(metaData, node);
                if (recordMap.get(node.toString()) != null) {
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


}
