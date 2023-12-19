package miner;

import json.MetaData;
import json.utils.NodePosition;
import lombok.Getter;
import lombok.Setter;
import org.eclipse.jdt.core.dom.*;

import java.util.HashMap;
import java.util.HashSet;

public class RatioVisitor extends AbstractExpressionVisitor {
    public RatioVisitor() {
        super(null);
    }

    @Getter
    @Setter
    int count=0;

    HashSet<String> nodeSet=new HashSet<>();

    @Override
    public boolean preVisit2(ASTNode node) {
        if (node instanceof Expression || node instanceof Name) {
            if( canReplace(node,false) &&
            nodeSet.add(node.toString()) ){
//                System.out.println(node);
                count++;
            }
//            System.out.println(node);
        }

        return super.preVisit2(node);
    }

}
