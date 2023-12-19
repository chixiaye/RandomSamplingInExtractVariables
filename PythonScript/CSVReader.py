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
                    project_info, version, _, before, after, status, timestamp = row
                    results[f'{project_info}_{version}_{str(self.flag)}'] = [status,before,after]
                except ValueError as e:
                    print(row, e)


        return results
