package json.utils;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.builder.ToStringBuilder;

@JsonPropertyOrder({"charLength","startLineNumber","startColumnNumber","endLineNumber","endColumnNumber"})
public class NodePosition {
    @Getter
    @Setter
    int startLineNumber;
    @Getter
    @Setter
    int startColumnNumber;

    @Getter
    @Setter
    int endLineNumber;
    @Getter
    @Setter
    int endColumnNumber;

    @Getter
    @Setter
    int charLength;

    public NodePosition(int startLineNumber, int startColumnNumber, int endLineNumber, int endColumnNumber, int charLength) {
        this.startLineNumber = startLineNumber;
        this.startColumnNumber = startColumnNumber;
        this.endLineNumber = endLineNumber;
        this.endColumnNumber = endColumnNumber;
        this.charLength = charLength;
    }
    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }
}
