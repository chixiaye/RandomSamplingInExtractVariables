import os
import shutil


def copy_folders(input_filenames, destination_directory):
    # 获取程序所在的路径
    script_directory = os.path.abspath("D:\Top1K\dataset")

    # 构建目标目录的绝对路径
    destination_directory = os.path.join(script_directory, destination_directory)

    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for filename in input_filenames:
        # 构建文件夹的绝对路径
        folder_path = os.path.join(script_directory, filename)

        # 检查文件夹是否存在
        if os.path.exists(folder_path):
            # 获取文件夹名称
            folder_name = os.path.basename(folder_path)

            # 构建目标文件夹路径
            destination_path = os.path.join(destination_directory, folder_name)

            # 复制文件夹及其内容
            try:
                shutil.copytree(folder_path, destination_path)
                print(f"成功复制文件夹 {folder_path} 到 {destination_path}")
            except shutil.Error as e:
                print(f"复制文件夹 {folder_path} 失败: {e}")
        else:
            print(f"文件夹 {folder_path} 不存在")


if __name__ == "__main__":
    # 输入文件夹名列表
    input_filenames = input("请输入文件夹名列表（以空格分隔）: ").split()

    # 指定目标目录
    destination_directory = input("请输入目标目录: ")

    # 调用函数进行复制操作
    copy_folders(input_filenames, destination_directory)
