//package io;
//import java.io.FileInputStream;
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.List;
//
//import org.apache.poi.ss.usermodel.Cell;
//import org.apache.poi.ss.usermodel.Row;
//import org.apache.poi.ss.usermodel.Sheet;
//import org.apache.poi.xssf.usermodel.XSSFWorkbook;
//
//public class ExcelReader {
//    public static void main(String[] args) {
//        String filePath = "/path/to/your/excel/file.xlsx";
//        ExcelDataHandler handler = new ExcelDataHandler();
//        ExcelReader reader = new ExcelReader(handler);
//        reader.read(filePath);
//    }
//
//    private ExcelDataHandler handler;
//
//    public ExcelReader(ExcelDataHandler handler) {
//        this.handler = handler;
//    }
//
//    public void read(String filePath) {
//        try (FileInputStream inputStream = new FileInputStream(filePath);
//             XSSFWorkbook workbook = new XSSFWorkbook(inputStream)) {
//            Sheet sheet = workbook.getSheetAt(0);
//            for (Row row : sheet) {
//                ExcelData data = readRow(row);
//                if (data != null) {
//                    handler.handle(data);
//                }
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
//
//    private ExcelData readRow(Row row) {
//        Cell accountCell = row.getCell(0);
//        Cell repositoryCell = row.getCell(1);
//        Cell numberCell = row.getCell(2);
//        if (accountCell != null && repositoryCell != null && numberCell != null) {
//            String account = accountCell.getStringCellValue();
//            String repository = repositoryCell.getStringCellValue();
//            int number = (int) numberCell.getNumericCellValue();
//            return new ExcelData(account, repository, number);
//        }
//        return null;
//    }
//}
//
//class ExcelData {
//    private String account;
//    private String repository;
//    private int number;
//
//    public ExcelData(String account, String repository, int number) {
//        this.account = account;
//        this.repository = repository;
//        this.number = number;
//    }
//
//    public String getAccount() {
//        return account;
//    }
//
//    public String getRepository() {
//        return repository;
//    }
//
//    public int getNumber() {
//        return number;
//    }
//}
//
//interface ExcelDataHandler {
//    void handle(ExcelData data);
//}
//
//class ConsoleDataHandler implements ExcelDataHandler {
//    public void handle(ExcelData data) {
//        System.out.printf("%s\t%s\t%d\n", data.getAccount(), data.getRepository(), data.getNumber());
//    }
//}