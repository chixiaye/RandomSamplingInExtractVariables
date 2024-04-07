import csv


class CSVReader:
    def __init__(self, file_path, flag):
        self.file_path = file_path
        self.flag = flag  # positive : 1 negative: 0

    def read_csv(self):
        results = {}
        # unicode_escape是对编码后存储的文本，读取时进行反向转换，就能直接得到原始文本数据
        with open(self.file_path, 'r', newline='', encoding='unicode_escape') as file:
            # reader = csv.reader(file)
            # 将空字符全部替换掉
            reader = csv.reader((line.replace('\0', '') for line in file))
            for row in reader:
                try:
                    # Assuming the CSV format is consistent
                    # ID,Project Name,SHA,New Name,Label,Approach,Position
                    id, project_info, version, _, _, approach, position = row
                    if approach != 'ours':  # or "*"  in position
                        continue
                    x_ = [int(x) for x in position.split('*') if x != '']
                    # 如果X内的是不连续的 打印警告
                    # if len(x_) > 0:
                        # print(row)
                        # for i in range(len(x_) - 1):
                        #     if x_[i ] - x_[i] != 1:
                        #         x_ = x_[0:i + 1]  # 有点问题 应该调整一下
                    results[f'{project_info}_{id}'] = x_
                    # print(row)
                except ValueError as e:
                    print(row, e)

        return results
