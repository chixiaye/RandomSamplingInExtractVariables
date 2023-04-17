package json;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.StructuralPropertyDescriptor;

@JsonPropertyOrder({"nodeContext", "nodeType", "ASTNodeNumber", "ASTHeight", "nodePosition"})
public abstract class AbstractNodeData {
    @Getter
    @Setter
    NodePosition nodePosition;
    @Getter
    @Setter
    String nodeContext;
    @Getter
    @Setter
    String nodeType;

    @Getter
    @Setter
    int astNodeNumber;
    @Getter
    @Setter
    int astHeight;

    @JsonIgnore
    @Getter
    @Setter
    boolean validFlag;

    public AbstractNodeData(String node, String nodeType, NodePosition nodePosition) {
        this.nodeContext = node;
        this.nodeType = nodeType;
        this.nodePosition = nodePosition;
        this.validFlag =true;
    }

    public AbstractNodeData() {
    }

    public void countASTNodeComplexity(ASTNode node) {
        //calculate the numbers of ASTNode of the Given ASTNode
        node.accept(new ASTVisitor() {
            @Override
            public boolean preVisit2(ASTNode node) {
                astNodeNumber++;
                return super.preVisit2(node);
            }
        });
        //calculate the height of ASTNode of the Given ASTNode
        astHeight = calculateHeight(node);
    }

    private int calculateHeight(ASTNode node) {
        int height = 0;
        for (Object o : node.structuralPropertiesForType()) {
            StructuralPropertyDescriptor pd = (StructuralPropertyDescriptor) o;
            if (pd.isChildProperty()) {
                Object child = node.getStructuralProperty(pd);
                if (child instanceof ASTNode) {
                    height = Math.max(height, calculateHeight((ASTNode) child));
                }
            } else if (pd.isChildListProperty()) {
                for (Object child : (Iterable<?>) node.getStructuralProperty(pd)) {
                    height = Math.max(height, calculateHeight((ASTNode) child));
                }
            }
        }
        return height + 1;
    }

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }
}

