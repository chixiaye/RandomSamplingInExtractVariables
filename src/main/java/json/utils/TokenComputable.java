package json.utils;

import utils.TokenParser;

public interface TokenComputable {
    default int computeToken(String s){
        return TokenParser.evaluateTokenLength(s);
    }
}
