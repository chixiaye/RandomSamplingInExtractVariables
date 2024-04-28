package json;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.util.ArrayList;

//@JsonIgnoreProperties(ignoreUnknown = true)
@JsonPropertyOrder({"id", "expression", "projectName", "commitID", "filePath", "occurrences", "isArithmeticExpression",
        "isGetTypeMethod", "expressionList", "positionList"})
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
    @Setter
    ArrayList<LayoutRelationData> layoutRelationDataList;

    @Getter
    @Setter
    ArrayList<NodePosition> positionList;

    @Getter
    @Setter
    String name;

    @Getter
    @Setter
    NodePosition nodePosition;


    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }

    public void generatePositionList(ArrayList<MetaData> expressionList) {
        this.positionList = new ArrayList<>();
        for (MetaData metaData : expressionList) {
            this.positionList.add(metaData.getNodePosition());
        }
    }

    public void setLayoutRelationDataList() {
        this.layoutRelationDataList = new ArrayList<>();

        for (int i = 0; i < expressionList.size(); i++) {
            for (int j = 0; j < expressionList.size(); j++) {
                if (i == j) {
                    continue;
                }
                MetaData metaData1 = expressionList.get(i);
                MetaData metaData2 = expressionList.get(j);
                LayoutRelationData layoutRelationData = new LayoutRelationData(i, j);
                layoutRelationData.setRelationship(metaData1, metaData2);
                this.layoutRelationDataList.add(layoutRelationData);
            }
        }
    }

    public void initLayoutRelationDataListInit() {
        this.layoutRelationDataList = new ArrayList<>();

    }
}
