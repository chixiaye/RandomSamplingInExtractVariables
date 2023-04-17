package utils;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.util.Iterator;
import java.util.Random;
import java.util.Set;

@Slf4j
// for each file, we randomly select 1 record
public class RandomSelection {
    @Getter
    @Setter
    private int totalRecords;

    @Getter
    @Setter
    private int currentRecords;

    private ThreadLocal<Random> randomThreadLocal = ThreadLocal.withInitial(() -> new Random( ));

    public RandomSelection(int totalRecords) {
        this.totalRecords = totalRecords;
        this.currentRecords = 0;
    }

    public void incCurrentRecords() {
        currentRecords++;
    }

    public <E> E generateRandomObjectFromSet(Set<E> set) {
        if (set == null || set.size() == 0){
            return null;
        }
        int size = set.size();
        int v = randomThreadLocal.get().nextInt(0, size);

        Iterator<E> it = set.iterator();
        E e = null;
        for (int i = 0; it.hasNext(); ++i) {
            if (i == v) {
                e = it.next();
//                log.info("{} random value: {}", e, v);
                break;
            }
            it.next();
        }
        return e;
    }

}
