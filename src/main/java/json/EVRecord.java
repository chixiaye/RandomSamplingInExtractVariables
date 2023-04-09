package json;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.util.ArrayList;
@JsonPropertyOrder({"id", "expression","projectName", "commitID", "filePath", "occurrences","expressionList","positionList"})
public class EVRecord  {
    @Getter
    @Setter
    int id;

    @Getter
    @Setter
    String expression;

    @Getter
    @Setter
    String projectName;

    @Getter
    @Setter
    String commitID;

    @Getter
    @Setter
    String filePath;

    @Getter
    @Setter
    int occurrences;

    @Getter
    @Setter
    ArrayList<MetaData> expressionList;

    @Getter
    ArrayList<NodePosition> positionList;

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }

    public void setPositionList(ArrayList<MetaData> expressionList) {
        this.positionList = new ArrayList<>();
        for (MetaData metaData : expressionList) {
            this.positionList.add(metaData.getNodePosition());
        }
    }
}
