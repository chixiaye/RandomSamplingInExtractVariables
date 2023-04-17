package io;

import json.EVRecord;
import junit.framework.TestCase;

import java.io.IOException;
import java.util.ArrayList;

public class RMDataReaderTest extends TestCase {

    public void testDoReadAction() {
        RMDataReader rmDataReader = new RMDataReader("C:\\Users\\30219\\Documents\\WeChat Files\\wxid_fdowsi9u80mi22\\FileStorage\\File\\2023-04\\records4Context\\ac-pm_Inspeckage.txt");
        try {
            ArrayList<EVRecord> variableData = rmDataReader.doReadAction();
            System.out.println(variableData.get(0).toString());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}