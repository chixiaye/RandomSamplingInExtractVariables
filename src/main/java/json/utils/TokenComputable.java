package json.utils;

import utils.TokenParser;

public interface TokenComputable {
    default int computeToken(String s){
        int v= TokenParser.evaluateTokenLength(s);
        return v;
    }
}
