'''
按照项目顺序轮流 检查是不是确实是重构
'''

import json
import logging
import os
import webbrowser

import pyperclip


def save_json_file(project_name, id_num, json):
    # 拼接文件名
    file_name = f"{project_name}_{id_num}.json"
    # 写入文件
    with open(f"{PREFIX}/{file_name}", "w", encoding='utf-8') as f:
        f.write(json)


def load_data(file):
    if file.endswith('.json'):
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)


def open_github_page(project_name, version, json_data):
    file_path = json_data[version - 1]['refactoredFilePath']
    refactored_name = json_data[version - 1]['refactoredName']
    commitID = json_data[version - 1]['refactoredCommitID']
    pyperclip.copy(refactored_name)

    print(json.dumps(json_data[version - 1], indent=4, ensure_ascii=False))
    # 拼接GitHub页面的URL
    url = "https://github.com/{0}/blob/{1}/{2}".format(project_name.replace('@', '/'), commitID, file_path)

    # 打开浏览器
    webbrowser.open(url)

    label = input("是否重构，是的话输入1，否则输入其他：")
    if label != "1":
        return False

    # 添加refactoredURL字段
    json_data[version - 1]["refactoredURL"] = url

    # 使用json.dumps()方法将Python对象转换为JSON格式的字符串，并进行格式化
    json_str = json.dumps(json_data[version - 1], indent=4, ensure_ascii=False)

    save_json_file(project_name, version, json_str)
    return True


PREFIX = "./labeledData"

# 设置日志级别为INFO，只记录INFO级别以上的信息
logging.basicConfig(level=logging.INFO)
# 创建FileHandler并配置日志文件名
file_handler = logging.FileHandler('./resource/labelData.log')
# 将FileHandler添加到logger中
logger = logging.getLogger()
logger.filter(lambda record: record.levelno == logging.INFO)
logger.addHandler(file_handler)

if __name__ == '__main__':
    # 读取 resource/projects.csv 文件， 按star数量取前100
    projects_info_path = "./resource/projects.csv"
    projects_info = []
    with open(projects_info_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            projects_info.append(line.strip().split(',')[0])

    top100 = []
    for index in range(1, len(projects_info)):
        project_info = projects_info[index]
        local_path = "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\refactoring\\" + project_info + ".json"
        if not os.path.exists(local_path):
            continue
        top100.append(projects_info[index])  # top 100

    projects_info = top100[:100]
    # print(len(projects_info))
    # exit(projects_info)

    for index in range(0, len(projects_info)):  # 84
        project_info = projects_info[index]
        if project_info != 'bumptech@glide':
            continue
        if project_info == 'ben-manes@caffeine' or project_info == 'medcl@elasticsearch-analysis-ik' or project_info == 'iBotPeaches@Apktool' \
                or project_info == 'Tencent@VasSonic' or project_info == 'Yalantis@uCrop' or project_info == 'jfeinstein10@SlidingMenu' \
                or project_info == 'Konloch@bytecode-viewer' or project_info == 'LMAX-Exchange/disruptor' or project_info == 'permissions-dispatcher@PermissionsDispatcher' \
                or project_info == 'LMAX-Exchange@disruptor' or project_info == 'facebookarchive@stetho':
            continue

        local_path = "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\refactoring\\" + project_info + ".json"
        if not os.path.exists(local_path):
            continue

        json_data = load_data(local_path)

        # 显示当前的项目有多少个重构信息
        print("There are {0} refactoring information in project {1}. ( index {2} )".format(len(json_data), project_info,
                                                                                           index))

        max_current_version = 1
        for i in range(1, 1 + len(json_data)):
            file_name = f"{project_info}_{i}.json"
            file_exists = os.path.isfile(f"{PREFIX}/{file_name}")
            if file_exists:
                max_current_version = i

        # 每个重构放一个文件,如果存在则跳过
        for i in range(max_current_version + 1, 1 + len(json_data)):
            file_name = f"{project_info}_{i}.json"
            file_exists = os.path.isfile(f"{PREFIX}/{file_name}")
            if file_exists:
                continue
            flag = open_github_page(project_info, i, json_data)
            if flag:
                break
        break