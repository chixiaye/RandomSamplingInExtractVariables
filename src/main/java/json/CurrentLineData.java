package json;

import json.utils.NodePosition;
import json.utils.TokenComputable;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class CurrentLineData extends ParentData implements TokenComputable {
    @Getter
    @Setter
    private int tokenLength;

    public CurrentLineData(String node, String nodeType, String locationInParent, NodePosition nodePosition) {
        super(node, nodeType, locationInParent, nodePosition);
    }


    public void setTokenLength() {
        this.tokenLength = computeToken(this.nodeContext);
//        log.info("{} token Length: {}" , this.nodeContext, tokenLength);
    }
}
