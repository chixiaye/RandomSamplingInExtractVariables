import os
import json

# 文件夹路径
folder_path = "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\PythonScript\\read\\"

# 遍历文件夹
for filename in os.listdir(folder_path):
    # 检查是否为json文件
    if filename.endswith(".json"):
        # 打开json文件并解析数据
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r",encoding='utf-8') as json_file:
            data = json.load(json_file)

        data_file_path = data['refactoredFilePath']
        refactored_name = data['refactoredName']
        commitID = data['refactoredCommitID']
        project_name = data['projectName']
        url = "https://github.com/{0}/blob/{1}/{2}".format(project_name.replace('@', '/'), commitID, data_file_path)
        # 添加refactoredURL字段
        data["refactoredURL"] = url

        # 保存修改后的json数据到同一文件中
        with open(file_path, "w", encoding='utf-8') as json_file:
            json.dump(data, json_file,indent=4, ensure_ascii=False)
