package json.utils;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j;
import utils.TokenParser;

public interface TokenComputable {
    default int computeToken(String s){
        int v= TokenParser.evaluateTokenLength(s);
        return v;
    }
}
