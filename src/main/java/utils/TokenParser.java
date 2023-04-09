package utils;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.ModelType;

import java.util.List;

public class TokenParser {
    private  static String preHandleContext(String s){
        String s1 = s.replaceAll("[\\p{Punct}\\p{Cntrl}]", " ");
        return s1;
    }


    public static int evaluateTokenLength(String context){
        context=preHandleContext(context);
        EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
        Encoding enc = registry.getEncoding(EncodingType.CL100K_BASE);
        List<Integer> encoded = enc.encode(context);
        return encoded.size();
        // encoded = [2028, 374, 264, 6205, 11914, 13]
//        String decoded = enc.decode(encoded);
        // decoded = "This is a sample sentence."
        // Or get the tokenizer based on the model type
//        Encoding secondEnc = registry.getEncodingForModel(ModelType.TEXT_EMBEDDING_ADA_002);
        // enc == secondEnc
    }
}
