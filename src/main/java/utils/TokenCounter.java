package utils;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TokenCounter {
    /**
     * 分词，按照Java编程规范进行分词
     *
     * @param code Java代码字符串
     * @return 分词结果
     */
    public static List<String> tokenize(String code) {
        List<String> tokens = new ArrayList<>();

        // 匹配Java标识符
        Pattern identifierPattern = Pattern.compile("[a-zA-Z_$][a-zA-Z0-9_$]*");
        Matcher identifierMatcher = identifierPattern.matcher(code);

        // 匹配Java关键字
        Pattern keywordPattern = Pattern.compile("\\b(abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|extends|false|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|native|new|null|package|private|protected|public|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|true|try|void|volatile|while)\\b");
        Matcher keywordMatcher = keywordPattern.matcher(code);

        // 匹配Java数字字面量
        Pattern numberPattern = Pattern.compile("\\b(0[xX][0-9a-fA-F]+|[0-9]+\\.?[0-9]*([eE][+-]?[0-9]+)?)\\b");
        Matcher numberMatcher = numberPattern.matcher(code);

        // 逐个匹配
        while (identifierMatcher.find()) {
            String token = identifierMatcher.group();
            if (!isKeyword(token)) {
                tokens.add(token);
            }
        }
        while (keywordMatcher.find()) {
            tokens.add(keywordMatcher.group());
        }
        while (numberMatcher.find()) {
            tokens.add(numberMatcher.group());
        }

        return tokens;
    }

    /**
     * 判断字符串是否是Java关键字
     */
    private static boolean isKeyword(String s) {
        return s.matches("\\b(abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|extends|false|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|native|new|null|package|private|protected|public|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|true|try|void|volatile|while)\\b");
    }
}

