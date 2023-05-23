package io.json;

import calculate.Correlation;
import calculate.condition.ConditionBuilder;
import io.excel.ExcelReader;
import junit.framework.TestCase;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import sample.Constants;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

@Slf4j
public class JSONReaderTest extends TestCase {

    @Test
    public void testDeserializeAsEVRecord() throws IOException, InvocationTargetException, NoSuchMethodException, IllegalAccessException {
        ExcelReader excelReader = new ExcelReader(Constants.EXCEL_PATH);
        excelReader.read();
        Correlation correlation = new Correlation(excelReader);
        ConditionBuilder conditionBuilder = new ConditionBuilder();
//            conditionBuilder.addOccurrences(2);
//            conditionBuilder.filter(correlation);
//            try {
//                correlation.calculate4RQ1();
//            } catch (IOException | InvocationTargetException | NoSuchMethodException | IllegalAccessException e) {
//                throw new RuntimeException(e);
//            }
        correlation.calculate4RQ3();
    }


}