package json;

import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;

/*
* json example
* {
  "id": 16,
  "projectName": "Tencent@tinker",
  "originalCommitID": "19f60dd760a6799f06651e29342b1c97ae3155f8",
  "refactoredCommitID": "737aee2e78e4901018eec2059f221218010f9784",
  "originalFilePath": "tinker-android/tinker-android-loader/src/main/java/com/tencent/tinker/loader/TinkerLoader.java",
  "refactoredFilePath": "tinker-android/tinker-android-loader/src/main/java/com/tencent/tinker/loader/TinkerLoader.java",
  "originalName": "SharePatchFileUtil.getPatchVersionFile(version)",
  "refactoredName": "patchVersionFileRelPath",
  "type": "String",
  "originalPositionList": [
    {
      "charLength": -1,
      "startLineNumber": 166,
      "startColumnNumber": 9,
      "endLineNumber": 166,
      "endColumnNumber": 136
    }
  ],
  "refactoredPositionList": [
    {
      "charLength": -1,
      "startLineNumber": 166,
      "startColumnNumber": 22,
      "endLineNumber": 166,
      "endColumnNumber": 95
    },
    {
      "charLength": -1,
      "startLineNumber": 167,
      "startColumnNumber": 9,
      "endLineNumber": 167,
      "endColumnNumber": 155
    }
  ],
  "refactoredURL": "https://github.com/Tencent/tinker/blob/737aee2e78e4901018eec2059f221218010f9784/tinker-android/tinker-android-loader/src/main/java/com/tencent/tinker/loader/TinkerLoader.java"
}
* */
public class LabelData {
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
    NodePosition[] originalPositionList;

    @Getter
    @Setter
    NodePosition[] refactoredPositionList;

    @Getter
    @Setter
    String refactoredURL;

    public LabelData(int id, String projectName, String originalCommitID, String refactoredCommitID,
                     String originalFilePath, String refactoredFilePath, String originalName,
                     String refactoredName, String type, NodePosition[] originalPositionList,
                     NodePosition[] refactoredPositionList, String refactoredURL) {
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
        this.refactoredURL = refactoredURL;
    }

    public LabelData() {
    }

    // toString method
    @Override
    public String toString() {
        return "LabelData{" +
                "id=" + id +
                ", projectName='" + projectName + '\'' +
                ", originalCommitID='" + originalCommitID + '\'' +
                ", refactoredCommitID='" + refactoredCommitID + '\'' +
                ", originalFilePath='" + originalFilePath + '\'' +
                ", refactoredFilePath='" + refactoredFilePath + '\'' +
                ", originalName='" + originalName + '\'' +
                ", refactoredName='" + refactoredName + '\'' +
                ", type='" + type + '\'' +
                ", originalPositionList=" + originalPositionList +
                ", refactoredPositionList=" + refactoredPositionList +
                ", refactoredURL='" + refactoredURL + '\'' +
                '}';
    }
}
