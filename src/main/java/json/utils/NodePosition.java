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

    public NodePosition() {
    }

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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NodePosition that)) return false;

        if (getStartLineNumber() != that.getStartLineNumber()) return false;
        if (getStartColumnNumber() != that.getStartColumnNumber()) return false;
        if (getEndLineNumber() != that.getEndLineNumber()) return false;
        if (getEndColumnNumber() != that.getEndColumnNumber()) return false;
        return getCharLength() == that.getCharLength();
    }

    @Override
    public int hashCode() {
        int result = getStartLineNumber();
        result = 31 * result + getStartColumnNumber();
        result = 31 * result + getEndLineNumber();
        result = 31 * result + getEndColumnNumber();
        result = 31 * result + getCharLength();
        return result;
    }
}
