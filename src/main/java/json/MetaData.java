package json;

import json.utils.NodePosition;
import json.utils.TokenComputable;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;

//@JsonPropertyOrder({"id","nodeContext","nodeType","projectName","commitID","filePath","nodePosition","parentDataList"})
public class MetaData extends AbstractNodeData implements TokenComputable {

    @Getter
    @Setter
    ArrayList<ParentData> parentDataList;


    @Getter
    @Setter
    int tokenLength;

    @Getter
    @Setter
    String type;


    public MetaData(NodePosition nodePosition, String node,String type) {
        super(node, "MetaData", nodePosition);
        this.type=type;
    }

    public MetaData() {
//        super(null, "MetaData", null);
    }

    public void setTokenLength() {
        this.tokenLength = computeToken(this.nodeContext);
    }


}

