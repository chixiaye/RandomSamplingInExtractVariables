package json.utils;

import utils.TokenParser;

public interface TokenComputable {
    default int computeToken(String s){
        int v = -1;//TokenParser.evaluateTokenLength(s);
        return v;
    }
}
