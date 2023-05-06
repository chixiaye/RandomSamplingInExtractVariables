package calculate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NodeVectorizer {
    private final Map<String, Integer> nodeTypes;

    public NodeVectorizer(List<String> nodeTypes) {
        this.nodeTypes = new HashMap<>();
        int index = 0;
        for (String type : nodeTypes) {
            if (!this.nodeTypes.containsKey(type)) {
                this.nodeTypes.put(type, index++);
            }
        }
    }

    public List<double[]> vectorize(List<String> nodes) {
        List<double[]> vectors = new ArrayList<>();
        for (String node : nodes) {
            double[] vector = new double[nodeTypes.size()];
            Integer index = nodeTypes.get(node);
            if (index != null) {
                vector[index] = 1;
            }
            vectors.add(vector);
        }
        return vectors;
    }
}
