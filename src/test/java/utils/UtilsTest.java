package utils;

import junit.framework.TestCase;
import org.junit.Test;
import sample.Constants;

import java.io.File;
import java.util.ArrayList;

public class UtilsTest extends TestCase {

    @Test
    public void testRename() {
        ArrayList<File> arrayList = new ArrayList<>();
        Utils.getFileList(arrayList,Constants.PREFIX_RM_PATH,"txt");
        for (File file : arrayList) {
            String source = file.getName();
            String dest = replaceFirstUnderscore(source, "@");
//            String str = file.getName();
//            int count = str.split("_").length - 1;
//            if(count>1)
//                System.out.println(str + " 下划线出现的次数为：" + count);
        }
//        Utils.rename(str);
    }

    public static String replaceFirstUnderscore(String str, String replacement) {
        int index = str.indexOf("_"); // 查找第一个下划线的位置
        if (index != -1) {
            return str.substring(0, index) + replacement + str.substring(index + 1);
        }
        return str;
    }
}