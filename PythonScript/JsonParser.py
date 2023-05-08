import os
import json
import chardet
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class JsonParser:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data = self.load_data()
        self.enc = OneHotEncoder()
        self.types = []
        self.cnt = 0

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
        values = []
        for file, json_data in self.data.items():
            for data in json_data:
                value = []
                for key in keys:
                    if key == "layoutRelationDataList":
                        arr = []
                        for k in np.array(data[key]):
                            arr.append(k['layout'])
                        value.append(0 if arr == [] else np.ptp(np.array(arr)))
                    elif key == 'nodeType':
                        # value.append(1 if data['expressionList'][0][key] == "MethodInvocation" else 0)
                        value.append(1 if data['expressionList'][0]["nodeType"] == "PrefixExpression" or
                                          data['expressionList'][0]["nodeType"] == "InfixExpression"  or
                                          data['expressionList'][0]["nodeType"] == " ConditionalExpression" else 0)
                        # if    len(data['expression']) >30   :
                        #     self.cnt+=1
                        # elif data['expressionList'][0][key] == "MethodInvocation" and  "get" in data['expression'].lower():
                        #     print( data['expression'])
                    elif key == 'currentLineData':
                        max_char_length = max(
                            [node.get('currentLineData', {}).get('nodePosition', {}).get('charLength', 0) for node in
                             data.get('expressionList', [])]
                        )

                        value.append(max_char_length)
                    elif key in data:
                        value.append(data[key])
                    elif key in data['expressionList'][0]:
                        value.append(data['expressionList'][0][key])
                    elif key in data['expressionList'][0]['nodePosition']:
                        # if data['expressionList'][0]['nodePosition'][key] <= 2:
                        #     print(data['expressionList'][0]['nodeContext'])
                        value.append(data['expressionList'][0]['nodePosition'][key])

                    # elif key=="charLength_CurrentLineData":
                    #     value.append(data['expressionList'][0]['nodePosition'][key])
                if value:
                    values.append(value)
        return values
