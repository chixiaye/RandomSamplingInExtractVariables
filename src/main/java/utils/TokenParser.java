package utils;

import java.util.StringTokenizer;

public class TokenParser {
    static String[] keywords = {"abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile", "while"};


    private static String preHandleContext(String s) {
        return s.replaceAll("[\\p{Punct}&&[^+\\-*/%<>!=~|&^!]]", " ");
    }

    public static int evaluateTokenLength(String context) {
//        EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
//        Encoding enc = registry.getEncoding(EncodingType.CL100K_BASE);
//        List<Integer> encoded = enc.encode(context);
//        return encoded.size();
        // encoded = [2028, 374, 264, 6205, 11914, 13]
//        String decoded = enc.decode(encoded);
        // decoded = "This is a sample sentence."
        // Or get the tokenizer based on the model type
//        Encoding secondEnc = registry.getEncodingForModel(ModelType.TEXT_EMBEDDING_ADA_002);
        // enc == secondEnc
        // 使用 StringTokenizer 分割文本
        StringTokenizer st = new StringTokenizer(preHandleContext(context));
        StringBuilder sb = new StringBuilder();

        while (st.hasMoreTokens()) {
            String token = st.nextToken();

            // 判断分割出来的单词是否是关键字
            boolean isKeyword = false;
            for (String keyword : keywords) {
                if (token.equals(keyword)) {
                    isKeyword = true;
                    break;
                }
            }

            // 如果不是关键字，则将其添加到 StringBuilder 中
            if (!isKeyword) {
                sb.append(token).append(" ");
            }
        }

        String result = sb.toString().trim();
        return result.split(" ").length;
    }
}
