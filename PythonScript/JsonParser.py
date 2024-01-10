import json
import os
import re

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re


class JsonParser:
    def __init__(self, data_folder, flag):
        self.data_folder = data_folder
        self.data = self.load_data()
        self.enc = OneHotEncoder()
        self.types = []
        self.cnt = 0
        self.flag = flag  # positive : 1 negative: 0

    def load_data(self):
        data = {}
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        data[file] = json_data
        return data

    def get_value(self, keys):
        maps = {}
        for file, json_data in self.data.items():
            data = json_data
            value = []
            for key in keys:
                if key == "layoutRelationDataList":
                    arr = []
                    for k in np.array(data[key]):
                        arr.append(k['layout'])
                    value.append(0 if arr == [] else np.ptp(np.array(arr)))
                elif key == 'isArithmeticExp':
                    # value.append(1 if data['expressionList'][0][key] == "MethodInvocation" else 0)
                    value.append(1 if data['expressionList'][0]["nodeType"] in ["InfixExpression", "PrefixExpression",
                                                                                "ConditionalExpression"  # , "ConditionalExpression",
                                                                                "PostfixExpression"] else 0)
                elif key == 'isGetMethod':
                    # value.append(1 if data['expressionList'][0][key] == "MethodInvocation" else 0)
                    value.append(1 if data['expressionList'][0]["nodeType"] == "MethodInvocation" and (
                            '.get' in data['expression'] or data['expression'].startswith('get')) else 0)
                    # if    len(data['expression']) >30   :
                    #     self.cnt+=1
                    # elif data['expressionList'][0][key] == "MethodInvocation" and  "get" in data['expression'].lower():
                    #     print( data['expression'])
                elif key == 'isSimpleName':
                    value.append(1 if data['expressionList'][0]["nodeType"] == "SimpleName" else 0)
                elif key == 'isQualifiedName':  # new 的情况小于1
                    value.append(1 if data['expressionList'][0]["nodeType"] == "QualifiedName" else 0)
                elif key == 'isParenthesizedExpression':
                    value.append(1 if data['expressionList'][0]["nodeType"] == "ParenthesizedExpression" else 0)
                elif key == 'isClassInstanceCreation':
                    value.append(1 if data['expressionList'][0]["nodeType"] == "ClassInstanceCreation" else 0)
                elif key == 'isArrayCreation':
                    value.append(1 if data['expressionList'][0]["nodeType"] == "ArrayCreation" else 0)
                elif key == 'isContainNull':
                    value.append(1 if 'null' in data['expression'] else 0)
                elif key == 'isAccess':
                    value.append(1 if 'Access' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isMethodInvocation':
                    value.append(1 if data['expressionList'][0]["nodeType"] == "MethodInvocation" else 0)
                elif key == 'isContainStrConcat':
                    value.append(1 if bool(re.search(r'(["\']).*?\1\s*\+\s*\w+', data['expression'])) else 0)
                elif key == 'isNumberLiteral':
                    value.append(1 if 'NumberLiteral' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isNullLiteral':
                    value.append(1 if 'NullLiteral' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isCharacterLiteral':
                    value.append(1 if 'CharacterLiteral' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isNumberLiteral':
                    value.append(1 if 'NumberLiteral' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isBooleanLiteral':
                    value.append(1 if 'BooleanLiteral' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isStringLiteral':
                    value.append(1 if 'StringLiteral' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isArrayAccess':
                    value.append(1 if 'ArrayAccess' in data['expressionList'][0]["nodeType"] else 0)
                elif key == 'isLambda':
                    value.append(1 if '->' in data['expression'] else 0)
                elif key == 'locationInParentIsInitializer':
                    value.append(1 if data['expressionList'][0]["nodeType"] == "QualifiedName" else 0)
                elif key == 'numInitializerInlocationInParent':
                    v = 0
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            if 'initializer' in expression['parentDataList'][0]["locationInParent"]:
                                v += 1
                    else:
                        for expression in data['expressionList']:
                            if 'initializer' in expression['parentDataList'][0]["locationInParent"]:
                                v += 1
                                print(expression['parentDataList'][0]["locationInParent"], v)
                    value.append(v)
                elif key == 'numsParentThrowStatement':
                    v = 0
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            if expression['parentDataList'][0]["nodeType"] == "ThrowStatement":
                                v += 1
                    else:
                        for expression in data['expressionList']:
                            if expression['parentDataList'][0]["nodeType"] == "ThrowStatement":
                                v += 1
                    value.append(v)
                elif key == 'maxParentAstHeight':
                    data_get = data.get('expressionList', [])
                    if self.flag == 1:
                        max_parent_ast_height = max(
                            [node.get('parentDataList', {})[0].get('astHeight', 0) for node in data_get[1:]]
                        )
                        max_parent_ast_height = max_parent_ast_height - 1 + data_get[0]['astHeight']
                    else:
                        max_parent_ast_height = max(
                            [node.get('parentDataList', {})[0].get('astHeight', 0) for node in data_get]
                        )
                    value.append(max_parent_ast_height)
                elif key == 'numsParentReturnStatement':
                    v = 0
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            if expression['parentDataList'][0]["nodeType"] == "ReturnStatement":
                                v += 1
                    else:
                        for expression in data['expressionList']:
                            if expression['parentDataList'][0]["nodeType"] == "ReturnStatement":
                                v += 1
                    value.append(v)
                elif key == 'numsParentCall':
                    v = 0
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            if expression['parentDataList'][0]["nodeType"] == "MethodInvocation" or \
                                    expression['parentDataList'][0]["nodeType"] == "ClassInstanceCreation":
                                v += 1
                                # print("expression!",expression['parentDataList'][0]["nodeContext"],expression["nodeContext"])
                    else:
                        for expression in data['expressionList']:
                            if expression['parentDataList'][0]["nodeType"] == "MethodInvocation" or \
                                    expression['parentDataList'][0]["nodeType"] == "ClassInstanceCreation":
                                # and (
                                # expression["nodeContext"] + ')' in expression['parentDataList'][0]["nodeContext"] or
                                # expression["nodeContext"] + ',' in expression['parentDataList'][0]["nodeContext"]):
                                v += 1
                    value.append(v)
                elif key == 'numsInCond':
                    v = 0
                    # 在if块的个数
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            if expression['parentDataList'][0]["nodeType"] == "IfStatement":
                                v += 1
                    else:
                        for expression in data['expressionList']:
                            if expression['parentDataList'][0]["nodeType"] == "IfStatement":
                                v += 1
                    value.append(v)
                elif key == 'numsParentArithmeticExp':
                    v = 0
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            if expression['parentDataList'][0]["nodeType"] in ["InfixExpression"]:
                                v += 1
                    else:
                        for expression in data['expressionList']:
                            if expression['parentDataList'][0]["nodeType"] in ["InfixExpression"]:
                                v += 1
                    value.append(v)
                elif key == 'numsParentVariableDeclarationFragment':
                    v = 0
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            for i in range(len(expression[
                                                   'parentDataList'])):  # if expression['parentDataList'][0]["nodeType"] == "VariableDeclarationFragment":
                                if expression['parentDataList'][1]["nodeType"] == "VariableDeclarationFragment":
                                    v += 1
                                    break
                                # print(data['expressionList'][0]['nodeContext'], parentNode["nodeContext"])
                                # break
                    else:
                        for expression in data['expressionList']:
                            for i in range(len(expression[
                                                   'parentDataList'])):  # if expression['parentDataList'][0]["nodeType"] == "VariableDeclarationFragment":
                                if expression['parentDataList'][i]["nodeType"] == "VariableDeclarationFragment":
                                    v += 1
                                    break
                    if self.flag == 1:
                        cnt = data['occurrences'] - 1
                    else:
                        cnt = data['occurrences']
                    value.append(v / cnt)
                elif key == 'numsInAssignment':
                    v = 0
                    if self.flag == 1:
                        for expression in data['expressionList'][1:]:
                            if expression['parentDataList'][0]["nodeType"] == "Assignment":
                                v += 1
                    else:
                        for expression in data['expressionList']:
                            if expression['parentDataList'][0]["nodeType"] == "Assignment":
                                v += 1
                    value.append(v)
                elif key == 'currentLineData':
                    data_get = data.get('expressionList', [])
                    if self.flag == 1:
                        max_char_length = max(
                            [node.get('currentLineData', {}).get('nodePosition', {}).get('charLength', 0) for node in
                             data_get[1:]]
                        )
                        max_char_length = max_char_length + len(data['expressionList'][0]['nodeContext']) - \
                                          data['expressionList'][1]['nodePosition']['charLength']
                    else:
                        max_char_length = max(
                            [node.get('currentLineData', {}).get('nodePosition', {}).get('charLength', 0) for node in
                             data_get]
                        )
                    value.append(max_char_length)
                elif key == 'avgCurrentLineData':
                    data_get = data.get('expressionList', [])
                    if self.flag == 1:
                        array = [node.get('currentLineData', {}).get('nodePosition', {}).get('charLength', 0) for node
                                 in data_get[1:]]
                        avg_char_length = sum(array) / len(array) + (
                                len(data['expressionList'][0]['nodeContext']) - \
                                data['expressionList'][1]['nodePosition']['charLength'])
                        value.append(avg_char_length)
                    else:
                        array = [node.get('currentLineData', {}).get('nodePosition', {}).get('charLength', 0) for node
                                 in data_get]
                        avg_char_length = sum(array) / len(array)
                        value.append(avg_char_length)
                elif key in data:
                    if key == 'occurrences':
                        if self.flag == 1:
                            cnt = data['occurrences'] - 1
                        else:
                            cnt = data['occurrences']
                        value.append(cnt)
                        if 'isOccurOnce' in keys:
                            value.append(1 if cnt == 1 else 0)
                        # print(data[key] - 1)
                    else:
                        value.append(data[key])
                elif key in data['expressionList'][0]:
                    value.append(data['expressionList'][0][key])
                elif key in data['expressionList'][0]['nodePosition']:
                    # if data['expressionList'][0]['nodePosition'][key] <= 2:
                    #     print(data['expressionList'][0]['nodeContext'])
                    if key == 'charLength':
                        pattern = r'("[^"]*")|\s+'
                        result = re.sub(pattern, lambda m: m.group(1) if m.group(1) else '',
                                        data['expressionList'][0]['nodeContext'])
                        # if "\"" in data['expressionList'][0]['nodeContext']:
                        #     print(result)
                        value.append(len(result))
                    else:
                        value.append(data['expressionList'][0]['nodePosition'][key])
                elif key == 'largestLineGap':
                    if self.flag == 1:
                        start_line_numbers = [expr['nodePosition']['startLineNumber'] for expr in
                                              data['expressionList'][1:]]
                    else:
                        start_line_numbers = [expr['nodePosition']['startLineNumber'] for expr in
                                              data['expressionList']]
                    v = max(start_line_numbers) - min(start_line_numbers)
                    value.append(v)
                elif key == 'sumLineGap':
                    if self.flag == 1:
                        start_line_numbers = [expr['nodePosition']['startLineNumber'] for expr in
                                              data['expressionList'][1:]]
                    else:
                        start_line_numbers = [expr['nodePosition']['startLineNumber'] for expr in
                                              data['expressionList']]
                    value.append(sum(start_line_numbers))
                    # print(data['expressionList'][0]['nodeContext'],data['expressionList'][0]['nodePosition'][key])
                # elif key=="charLength_CurrentLineData":
                #     value.append(data['expressionList'][0]['nodePosition'][key])
                elif key == 'isPrimitiveType':
                    primitive_types = ['int', 'double', 'float', 'long', 'short', 'byte', 'char', 'boolean']
                    if data['expressionList'][0]['type'] in primitive_types:
                        value.append(1)
                    else:
                        value.append(0)
                elif key == 'isStreamMethod':
                    value.append(1 if '.stream(' in data['expression'] else 0)
                elif key == 'getNodeType':
                    # 'isClassInstanceCreation', 'isMethodInvocation', 'isSimpleName', 'isQualifiedName', 'isNumberLiteral',
                    # 'isCharacterLiteral', 'isStringLiteral',
                    type_dict = {"SimpleName": 200, "QualifiedName": 300, "NumberLiteral": 400,
                                 "CharacterLiteral": 500, "StringLiteral": 600, "MethodInvocation": 700,
                                 "ClassInstanceCreation": 800}
                    value.append(type_dict[data['expressionList'][0]["nodeType"]] if data['expressionList'][0][
                                                                                         "nodeType"] in type_dict else 0)
                elif key == 'maxStartColumnNumberIncurrentLineData':
                    if self.flag == 1:
                        start_line_numbers = [expr['currentLineData']['nodePosition']['startColumnNumber'] for expr in
                                              data['expressionList'][1:]]
                    else:
                        start_line_numbers = [expr['currentLineData']['nodePosition']['startColumnNumber'] for expr in
                                              data['expressionList']]
                    v = max(start_line_numbers)
                    value.append(v)
                elif key == 'maxEndColumnNumberInCurrentLine':
                    if self.flag == 1:
                        expressionList = data['expressionList'][1:]
                    else:
                        expressionList = data['expressionList']
                    v = 0
                    for expr in expressionList:
                        tmp = expr['nodePosition']['endColumnNumber']
                        for line in expr['parentDataList'] :
                            if line['nodePosition']['startLineNumber'] == expr['nodePosition']['startLineNumber']  \
                                    or line['nodePosition']['endLineNumber'] == expr['nodePosition']['endLineNumber']:
                                nodelen = len (line['nodeContext'].split('\n')[0])
                                tmp = max(tmp,line['nodePosition']['startColumnNumber']+nodelen)
                        v = max(v,tmp)
                    value.append(v)
            if value:
                maps[file.replace(".json", "_" + str(self.flag))] = value

        return maps
