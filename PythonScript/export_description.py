import json
import os


# 读取文件夹下的所有json文件
def read_json_file(file_path):
    json_data = {}
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    json_data[file] = json.load(f)

    return json_data


# 逐行记录每个json的文件名和对应的“description”字段的内容到一个csv文件中
def export_description(json_data):
    with open("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\PythonScript\\description.csv", "w",
              encoding='utf-8') as f:
        for data in json_data:
            f.write("{0},{1}\n".format(data, json_data[data]['Description']))


def print_description(json_data):
    for data in json_data:
        print("File name: {0}".format(data))
        # print(json_data[i])
        print("Description: {0}".format(json_data[data]['Description']))
        print("")


file_path = "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\PythonScript\\read\\"
export_description(read_json_file(file_path))
