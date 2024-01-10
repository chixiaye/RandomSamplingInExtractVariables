package json;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public
class Location {
    @JsonProperty("start line")
    private int startLine;

    @JsonProperty("start column")
    private int startColumn;

    @JsonProperty("end line")
    private int endLine;

    @JsonProperty("end column")
    private int endColumn;
}
