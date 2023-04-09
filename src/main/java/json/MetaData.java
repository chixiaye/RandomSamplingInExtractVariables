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
    CurrentLineData currentLineData;

    @Getter
    @Setter
    int tokenLength;


    public MetaData(NodePosition nodePosition, String node) {
        super(node, "MetaData", nodePosition);
    }

    public void setTokenLength() {
        this.tokenLength = computeToken(this.nodeContext);
    }


}

