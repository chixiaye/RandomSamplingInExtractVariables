package json;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeId;
import lombok.Data;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class CaseStudy {
    private int no;

    @JsonProperty("originalName")
    private String originalName;

    @JsonProperty("originalExpressionList")
    private OriginalExpression[] originalExpressionList;

    @JsonProperty("originalCommitId")
    private String originalCommitId;

    private String path;

    @JsonProperty("refactoredName")
    private String refactoredName;

    @JsonProperty("refactoredCommitId")
    private String refactoredCommitId;

    @JsonProperty("projectName")
    private String projectName;

    private int fixed;
}


