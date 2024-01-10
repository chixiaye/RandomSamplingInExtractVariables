package json;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public
class OriginalExpression {
    private int line;
    private int column;

    @JsonProperty("needExtracted")
    private int needExtracted;
}
