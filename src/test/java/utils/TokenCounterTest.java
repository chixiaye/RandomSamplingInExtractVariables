package utils;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.ModelType;
import junit.framework.TestCase;
import org.junit.Test;


public class TokenCounterTest extends TestCase {


    @Test
    public void testCountTokens() throws IOException {
        EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
        Encoding enc = registry.getEncoding(EncodingType.CL100K_BASE);
        List<Integer> encoded = enc.encode("This is a sample sentence.");
        // encoded = [2028, 374, 264, 6205, 11914, 13]
        String decoded = enc.decode(encoded);
        // decoded = "This is a sample sentence."
        // Or get the tokenizer based on the model type
        Encoding secondEnc = registry.getEncodingForModel(ModelType.TEXT_EMBEDDING_ADA_002);
        // enc == secondEnc
    }
    //
//    @Test
//    public void testCountTokens() throws IOException {
//        // Create a new pipeline
//        String text = "nextJavaCodeTokenizer";
//
//        // Set up Stanford CoreNLP pipeline
//        Properties props = new Properties();
//        props.setProperty("annotators", "tokenize");
//        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
//
//        // Create annotation for input text
//        Annotation annotation = new Annotation(text);
//
//        // Annotate the input text
//        pipeline.annotate(annotation);
//
//        // Get list of CoreMaps from the annotation
//        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
//
//        // Get list of tokens from the first CoreMap (only one in this case)
//        List<CoreLabel> tokens = sentences.get(0).get(CoreAnnotations.TokensAnnotation.class);
//
//        // Print tokens
//        for (CoreLabel token : tokens) {
//            System.out.println(token.word());
//        }
//    }
}