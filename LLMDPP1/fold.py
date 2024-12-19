import os
import shutil

# 设置你的文件夹路径
source_folder = r'C:\Users\wangk\Desktop\download'

# 确保源文件夹存在
if not os.path.exists(source_folder):
    print("Source folder does not exist.")
    exit()

# 创建一个新的目录来存放所有新的子文件夹
new_folder_base = r'C:\Users\wangk\Desktop\download1'
if not os.path.exists(new_folder_base):
    os.makedirs(new_folder_base)

# 获取源文件夹中所有文件的列表
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# 按每1000个文件分组
for i, file in enumerate(files, start=1):
    # 确定新子文件夹的名称
    folder_name = f"{i // 1000 + 1}"
    new_folder_path = os.path.join(new_folder_base, folder_name)

    # 如果子文件夹不存在，则创建它
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # 获取文件的完整路径
    file_path = os.path.join(source_folder, file)

    # 移动文件到新的子文件夹
    shutil.move(file_path, new_folder_path)

    print(f"Moved {file} to {new_folder_path}")