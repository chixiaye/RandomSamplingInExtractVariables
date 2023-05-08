import json
import os
import webbrowser
import pyperclip
import re


def is_valid_java_variable(s):
    if not s:
        return False
    if not re.match(r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', s):
        return False
    if s in {'abstract', 'assert', 'boolean', 'break', 'byte', 'case',
             'catch', 'char', 'class', 'const', 'continue', 'default',
             'do', 'double', 'else', 'enum', 'extends', 'false', 'final',
             'finally', 'float', 'for', 'goto', 'if', 'implements',
             'import', 'instanceof', 'int', 'interface', 'long', 'native',
             'new', 'null', 'package', 'private', 'protected', 'public',
             'return', 'short', 'static', 'strictfp', 'super', 'switch',
             'synchronized', 'this', 'throw', 'throws', 'transient',
             'true', 'try', 'void', 'volatile', 'while'}:
        return False
    return True


def save_json_file(project_name, id_num, json):
    # 拼接文件名
    file_name = f"{project_name}_{id_num}.json"
    prefix = "./SimpleName"

    # 判断文件是否存在，存在则加上后缀
    file_exists = os.path.isfile(f"{prefix}/{file_name}")
    suffix_num = 1
    while file_exists:
        file_name = f"{project_name}_{id_num}_{suffix_num}.json"
        file_exists = os.path.isfile(f"{prefix}/{file_name}")
        suffix_num += 1

    # 写入文件
    with open(f"{prefix}/{file_name}", "w", encoding='utf-8') as f:
        f.write(json)


def load_data(file):
    if file.endswith('.json'):
        with open(file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            return json_data


def open_github_page(project_name, version):
    if "@" not in project_name:
        print("Invalid project name format. Please use the format 'owner@repository'.")
        return
    if not isinstance(version, int) or version <= 0:
        print("Invalid version number. Please enter a positive integer.")
        return

    local_path = "C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\data\\refactoring\\" + project_name + ".json"
    json_data = load_data(local_path)

    # 显示当前的项目有多少个重构信息
    print("There are {0} refactoring information in total.".format(len(json_data)))

    for i in range(len(json_data)):
        if is_valid_java_variable(json_data[i]['originalName']):
            version = i
            break

    if version > len(json_data):
        print("Version number out of range. Please enter a valid version number.")
        return

    file_path = json_data[version - 1]['refactoredFilePath']
    refactored_name = json_data[version - 1]['refactoredName']
    commitID = json_data[version - 1]['refactoredCommitID']
    pyperclip.copy(refactored_name)

    print(json.dumps(json_data[version - 1], indent=4, ensure_ascii=False))
    # 拼接GitHub页面的URL
    url = "https://github.com/{0}/blob/{1}/{2}".format(project_name.replace('@', '/'), commitID, file_path)

    # 打开浏览器
    webbrowser.open(url)

    description = input("请输入描述：")
    json_data[version - 1]["Description"] = description
    # 添加refactoredURL字段
    json_data[version - 1]["refactoredURL"] = url
    # 使用json.dumps()方法将Python对象转换为JSON格式的字符串，并进行格式化
    json_str = json.dumps(json_data[version - 1], indent=4, ensure_ascii=False)

    save_json_file(project_name, version, json_str)


# 循环读取多组project_name和v的输入，直到用户输入一个指定的结束标记（例如输入“quit”）为止
while True:
    project_name = input("Please enter the project name (in the format 'owner@repository'): ")
    if project_name.lower() == "quit":
        break
    v = input("Please enter the version number: ")
    if v.lower() == "quit":
        break
    try:
        v = int(v)
    except ValueError:
        print("Invalid version number. Please enter a positive integer.")
        continue
    open_github_page(project_name, v)
