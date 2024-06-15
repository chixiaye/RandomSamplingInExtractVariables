import json
import os

import git
from datetime import datetime

# JRRT 实验，用于按照commit顺序找到重构

def load_data(data_folder):
    data = {}
    for root, dirs,files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    data[file] = json_data
    return data


if __name__ == "__main__":
    data = load_data("C:\\Users\\30219\\IdeaProjects\\RandomSamplingInExtractVariables\\final-data\\positive")
    # 字典 ， 项目名 到 json_DATA的映射
    dict_data = {}
    # 项目名到排除文件的映射
    exclude_dict = {}

    for item in data:
        project_name = item.split("_")[0]
        if project_name not in dict_data:
            dict_data[project_name] = [item]
        else:
            dict_data[project_name].append(item)
    for key in dict_data:
        values = dict_data[key]
        # Git 仓库路径
        repo_path = "E://34-projects//" + key+"//"
        if key in[ "commons-codec","antlr@antlr4","apache@rocketmq","apache@shardingsphere"] :
            continue
        # 初始化 Git 仓库对象
        repo = git.Repo(repo_path)
        earliest_time = None
        earliest_commit = None
        earliest_file = None
        for v in values:
            if key == "alibaba@spring-cloud-alibaba" and v == "alibaba@spring-cloud-alibaba_12.json" or  key == "projectlombok@lombok" and v == "projectlombok@lombok_1.json" :
                continue
            commit_id = data[v]["commitID"]
            # 根据 commit ID 获取提交对象
            commit = repo.commit(commit_id)
            # 获取提交的时间
            commit_time = datetime.fromtimestamp(commit.committed_date)
            # print(commit_time,earliest_time)
            if earliest_time is None or commit_time < earliest_time:
                earliest_time = commit_time
                earliest_commit = commit
                earliest_file = v
        print(f"Project: {key}, File Name: {earliest_file}, Earliest Commit ID: {earliest_commit.hexsha}, Commit Time: {earliest_time}")
            # print(f"Commit ID: {commit_id}, Commit Time: {commit_time}")

