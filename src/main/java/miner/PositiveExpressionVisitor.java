package miner;

import json.CurrentLineData;
import json.MetaData;
import json.ParentData;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import org.eclipse.jdt.core.dom.*;

import java.util.ArrayList;


public class PositiveExpressionVisitor extends AbstractExpressionVisitor {
    @Getter
    @Setter
    private ArrayList<MetaData> metaDataList;

    private NodePosition fNodePosition;
    private String fSearchedNodeContext;

    @Getter
    private NodePosition fCoveredNodePosition;

    public PositiveExpressionVisitor(CompilationUnit cu, String node, NodePosition nodePosition) {
        super(cu);
        this.metaDataList = new ArrayList<>();
        this.fNodePosition = nodePosition;
        this.fSearchedNodeContext = node;
        this.fCoveredNodePosition = null;
        //        System.out.println(nodePosition);
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
        ASTNode parent = node.getParent();
        // 首先找到声明的位置
        if (fCoveredNodePosition == null && node instanceof SimpleName
                && parent instanceof VariableDeclarationFragment vdf &&
                node.equals(vdf.getName()) && fSearchedNodeContext.equals(node.toString())) {
            int offset = node.getStartPosition();
            int length = node.getLength();
            NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                    , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
//            System.out.println(pos);
            if (pos.getStartLineNumber() >= fNodePosition.getStartLineNumber()
                    && pos.getEndLineNumber() <= fNodePosition.getEndLineNumber()) {
                // 对覆盖范围做初始化
                while (parent != null) {
                    if (parent instanceof MethodDeclaration || parent instanceof Initializer || parent instanceof LambdaExpression) {
                        break;
                    }
                    parent = parent.getParent();
                }

                int o = parent.getStartPosition();
                int l = parent.getLength();
                fCoveredNodePosition = new NodePosition(fCU.getLineNumber(o), fCU.getColumnNumber(l)
                        , fCU.getLineNumber(o + l), fCU.getColumnNumber(o + l), l);

                if (isArithmetic(vdf.getInitializer())) {
                    this.arithmeticExpressionState = 1;
                }

                if (isStartWithGet(vdf.getInitializer())) {
                    this.typeMethodState = 1;
                }

                //处理表达式的右边
                MetaData metaData = new MetaData(pos, vdf.getInitializer().toString(), getExpressionType(vdf.getInitializer()));//ASTNode.nodeClassForType(node.getNodeType()).getName()
                reLoadMetaData(metaData, vdf.getInitializer());
                metaDataList.add(metaData);
                return true;
            }
        } else if (fCoveredNodePosition != null && isSearchNode(node)) {
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
        if (!(node instanceof SimpleName) || !node.toString().equals(fSearchedNodeContext) || node.getLocationInParent() != null
                && "name".equals(node.getLocationInParent().getId())) {
            return false;
        }

        int offset = node.getStartPosition();
        int length = node.getLength();
        NodePosition pos = new NodePosition(fCU.getLineNumber(offset), fCU.getColumnNumber(offset)
                , fCU.getLineNumber(offset + length), fCU.getColumnNumber(offset + length), length);
        if (pos.getStartLineNumber() <= fCoveredNodePosition.getStartLineNumber()
                || pos.getEndLineNumber() >= fCoveredNodePosition.getEndLineNumber()) {
            return false;
        }

        return true;
    }

}
