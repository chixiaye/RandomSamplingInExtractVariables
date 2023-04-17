package json;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;

@JsonPropertyOrder({"locationInParent"})
public class ParentData extends AbstractNodeData {

    @Getter
    @Setter
    String locationInParent;
    public ParentData(String node, String nodeType, String locationInParent, NodePosition nodePosition) {
        super(node, nodeType, nodePosition);
        this.locationInParent=locationInParent;
    }

    public ParentData() {
    }
}
