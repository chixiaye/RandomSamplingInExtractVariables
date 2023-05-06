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
                    elif key in data:
                        value.append(data[key])
                    elif key in data['expressionList'][0]:
                        value.append(data['expressionList'][0][key])
                    elif key in data['expressionList'][0]['nodePosition']:
                        value.append(data['expressionList'][0]['nodePosition'][key])
                    # elif key=="charLength_CurrentLineData":
                    #     value.append(data['expressionList'][0]['nodePosition'][key])
                if value:
                    values.append(value)
        return values
