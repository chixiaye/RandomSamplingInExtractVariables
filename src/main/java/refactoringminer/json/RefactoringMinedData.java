package refactoringminer.json;

import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;

public class RefactoringMinedData {
    @Getter
    @Setter
    int id;

    @Getter
    @Setter
    String projectName;

    @Getter
    @Setter
    String originalCommitID;

    @Getter
    @Setter
    String refactoredCommitID;

    @Getter
    @Setter
    String originalFilePath;

    @Getter
    @Setter
    String refactoredFilePath;

    @Getter
    @Setter
    String originalName;

    @Getter
    @Setter
    String refactoredName;

    @Getter
    @Setter
    String type;

    @Getter
    @Setter
    ArrayList<NodePosition> originalPositionList;

    @Getter
    @Setter
    ArrayList<NodePosition> refactoredPositionList;

    public RefactoringMinedData(int id, String projectName, String originalCommitID, String refactoredCommitID, String originalFilePath, String refactoredFilePath, String originalName, String refactoredName, String type, ArrayList<NodePosition> originalPositionList, ArrayList<NodePosition> refactoredPositionList) {
        this.id = id;
        this.projectName = projectName;
        this.originalCommitID = originalCommitID;
        this.refactoredCommitID = refactoredCommitID;
        this.originalFilePath = originalFilePath;
        this.refactoredFilePath = refactoredFilePath;
        this.originalName = originalName;
        this.refactoredName = refactoredName;
        this.type = type;
        this.originalPositionList = originalPositionList;
        this.refactoredPositionList = refactoredPositionList;
    }
}
