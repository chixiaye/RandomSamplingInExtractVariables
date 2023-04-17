package miner;

import ast.ProjectsParser;
import io.json.JsonFileSplitter;
import utils.RandomSelection;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public abstract class AbstractMinerThread extends Thread{

    protected final int fTotalRecords;
    protected ProjectsParser fProjectsParser;
    protected JsonFileSplitter fJsonFileSplitter;
    String fProjectName;
    String fCommitID;
    ArrayList<File> fFileList;
    RandomSelection fRandomSelection;

    public abstract void analyzeProject() throws IOException;

    public AbstractMinerThread(String projectName,int totalRecords) {
        fTotalRecords = totalRecords;
        fRandomSelection = new RandomSelection(totalRecords);
        fProjectName=projectName;
    }

    public abstract void init(String projectName) ;
}
