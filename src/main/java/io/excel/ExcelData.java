package io.excel;

public class ExcelData {
    private String account;
    private String repository;
    private int number;

    public ExcelData(String account, String repository, int number) {
        this.account = account;
        this.repository = repository;
        this.number = number;
    }

    public String getAccount() {
        return account;
    }

    public String getRepository() {
        return repository;
    }

    public int getNumber() {
        return number;
    }

    @Override
    public String toString() {
        return "ExcelData{" +
                "account='" + account + '\'' +
                ", repository='" + repository + '\'' +
                ", number=" + number +
                '}';
    }


}

