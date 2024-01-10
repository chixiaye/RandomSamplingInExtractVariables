package json;

import com.fasterxml.jackson.annotation.JsonProperty;

public class CaseStudy {
    @JsonProperty("before commit")
    private String beforeCommit;

    @JsonProperty("after commit")
    private String afterCommit;

    @JsonProperty("file path")
    private String filePath;

    @JsonProperty("old name")
    private String oldName;

    @JsonProperty("new name")
    private String newName;

    private String type;

    @JsonProperty("variable place")
    private VariablePlace variablePlace;

    @JsonProperty("before location list")
    private Location[] beforeLocationList;

    @JsonProperty("after location list")
    private Location[] afterLocationList;

    // Getters and setters
}

class VariablePlace {
    private int startLine;
    private int startColumn;
    private int endLine;
    private int endColumn;

    // Getters and setters
}

class Location {
    private int startLine;
    private int startColumn;
    private int endLine;
    private int endColumn;

    // Getters and setters
}
