package json;

import lombok.Getter;
import lombok.Setter;

// firstkey 向上多少个节点能找到secondkey,那么layout就是多少
public class LayoutRelationData {
    @Getter
    @Setter
    int firstKey;
    @Getter
    @Setter
    int secondKey;

    @Getter
    @Setter
    int layout;

    public LayoutRelationData(int firstKey, int secondKey) {
        this.firstKey = firstKey;
        this.secondKey = secondKey;
    }

    public LayoutRelationData() {
    }

    public void setRelationship(MetaData m1, MetaData m2){
         int index1=m1.parentDataList.size()-1;
         int index2=m2.parentDataList.size()-1;
         int res=-1;
         while(index1>=0 && index2 >=0){
            if(m1.getParentDataList().get(index1).getNodePosition().equals(
                    m2.getParentDataList().get(index2).getNodePosition())){
                res=index1;
            }else {
                break;
            }
            index1--;
            index2--;
         }
         this.layout= res;
    }
}
