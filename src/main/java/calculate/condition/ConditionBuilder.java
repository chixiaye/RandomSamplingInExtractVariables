package calculate.condition;

import calculate.Correlation;
import json.EVRecord;
import lombok.Getter;

import java.util.Iterator;
import java.util.List;

/**
 * 建造者建立条件，用于过滤
 */
public class ConditionBuilder {

    @Getter
    private Condition condition = new Condition();

    public ConditionBuilder addOccurrences(Integer occurrences) {
        condition.setOccurrences(occurrences);
        return this;
    }

    public ConditionBuilder addCharLength(Integer charLength) {
        condition.setCharLength(charLength);
        return this;
    }

    public ConditionBuilder addTokenLength(Integer tokenLength) {
        condition.setTokenLength(tokenLength);
        return this;
    }

    public void filter(Correlation correlation) {
        Iterator<EVRecord> posIterator = correlation.getPosRecords().iterator();
        Iterator<EVRecord> negIterator = correlation.getNegRecords().iterator();
        while (posIterator.hasNext()) {
            EVRecord posRecord = posIterator.next();
            if (isMatch(posRecord)) {
                posIterator.remove();
            }
        }
        while (negIterator.hasNext()) {
            EVRecord posRecord = negIterator.next();
            if (isMatch(posRecord)) {
                negIterator.remove();
            }
        }
    }

    private boolean isMatch(EVRecord posRecord) {
        if (condition.getOccurrences() != null && condition.getOccurrences() <= posRecord.getOccurrences()) {
            return false;
        } else if (condition.getCharLength() != null && condition.getCharLength() <= posRecord.getPositionList().get(0).getCharLength()) {
            return false;
        } else if (condition.getTokenLength() != null && condition.getTokenLength() <= posRecord.getExpressionList().get(0).getTokenLength()) {
            return false;
        }
        return true;
    }


}
