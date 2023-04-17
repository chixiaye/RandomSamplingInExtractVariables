package io.excel;

import io.AbstractIOer;
import lombok.Getter;
import lombok.Setter;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

public class ExcelReader extends AbstractIOer {

    @Getter
    @Setter
    private ArrayList<ExcelData> excelDataList;



    public ExcelReader(String filePath) {
        excelDataList = new ArrayList<>();
        this.filePath= filePath;
    }

    public void read() {
        try (FileInputStream inputStream = new FileInputStream(filePath);
             XSSFWorkbook workbook = new XSSFWorkbook(inputStream)) {
            Sheet sheet = workbook.getSheetAt(0);
            Iterator<Row> iterator = sheet.rowIterator();
            iterator.next();
            while(iterator.hasNext()) {
                Row row=iterator.next();
                ExcelData data = readRow(row);
                if (data != null) {
                    excelDataList.add(data);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private ExcelData readRow(Row row) {
        Cell accountCell = row.getCell(0);
        Cell repositoryCell = row.getCell(1);
        Cell numberCell = row.getCell(2);
        if (accountCell != null && repositoryCell != null && numberCell != null) {
            String account = accountCell.getStringCellValue();
            String repository = repositoryCell.getStringCellValue();
            int number =  (int)numberCell.getNumericCellValue()  ;
            return new ExcelData(account, repository, number);
        }
        return null;
    }
}