package utils;

import junit.framework.TestCase;
import org.junit.Test;

import java.util.HashSet;

public class RandomSelectionTest extends TestCase {
    @Test
    public void testRandomSelection() {
        RandomSelection randomSelection = new RandomSelection(5);
        HashSet<Integer> set = new HashSet<>();
        set.add(1);
        set.add(2);
        set.add(3);
        set.add(4);
        set.add(5);
        set.add(6);
        Integer x = randomSelection.generateRandomObjectFromSet(set);
        System.out.println(x);
        set.remove(x);
        System.out.println(randomSelection.generateRandomObjectFromSet(set));
        System.out.println(randomSelection.generateRandomObjectFromSet(set));
        System.out.println(randomSelection.generateRandomObjectFromSet(set));
        System.out.println(randomSelection.generateRandomObjectFromSet(set));
        System.out.println(randomSelection.generateRandomObjectFromSet(set));
        System.out.println(randomSelection.generateRandomObjectFromSet(set));

    }

}