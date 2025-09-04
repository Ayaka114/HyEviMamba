import os
import pandas as pd
import shutil
import glob
from tqdm import tqdm

def organize_single_label_images(csv_path, image_source_dir, output_base_dir):
    """
    根据CSV文件，将“正常”和“单标签”的图片分类到新的文件夹中。

    参数:
    - csv_path (str): 标签CSV文件的路径。
    - image_source_dir (str): 原始图片所在的文件夹路径。
    - output_base_dir (str): 将要创建分类文件夹并存放图片的根目录。
    """
    # 检查输入路径
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件未找到 -> {csv_path}")
        return
    if not os.path.exists(image_source_dir):
        print(f"错误: 图片源文件夹未找到 -> {image_source_dir}")
        return

    # 读取CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # 获取所有疾病列
    disease_columns = df.columns[2:]

    # 遍历CSV的每一行
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"整理 {os.path.basename(image_source_dir)}"):
        image_id = row['ID']
        disease_risk = row['Disease_Risk']
        
        # 查找图片文件，兼容不同扩展名
        source_image_paths = glob.glob(os.path.join(image_source_dir, f"{image_id}.*"))
        if not source_image_paths:
            # 如果找不到图片，打印警告并跳过
            # print(f"\n警告: 未找到ID为 {image_id} 的图片，已跳过。") # 开启此行可查看详细跳过信息
            continue
        source_image_path = source_image_paths[0]

        target_category = None

        # 1. 判断是否为“正常”
        if disease_risk == 0:
            target_category = "Normal"
        
        # 2. 如果有疾病风险，判断是否为“单标签”
        else:
            # 提取疾病标签行
            disease_labels = row[disease_columns]
            # 计算1的个数
            num_diseases = (disease_labels == 1).sum()
            
            # 如果恰好只有一个疾病
            if num_diseases == 1:
                # 找到那个疾病的名称
                # idxmax() 会返回第一个值为True的列名
                target_category = disease_labels[disease_labels == 1].idxmax()

        # 3. 如果是“正常”或“单标签”，则执行复制操作
        if target_category:
            # 创建目标文件夹
            destination_folder = os.path.join(output_base_dir, target_category)
            os.makedirs(destination_folder, exist_ok=True)
            
            # 复制文件
            destination_image_path = os.path.join(destination_folder, os.path.basename(source_image_path))
            shutil.copy(source_image_path, destination_image_path)
            
    print(f"整理完成: {output_base_dir}")


if __name__ == "__main__":
    # --- 基础路径配置 ---
    base_folder = '/home1/zhouhao/MedMamba_ywj/datasets/RFMiD2.0'
    
    # --- 新的输出目录 ---
    # 我们将所有整理后的单标签数据放入一个新文件夹，以保持原数据集的完整性
    output_dataset_folder = '/home1/zhouhao/MedMamba_ywj/datasets/RFMiD2.0/new'
    os.makedirs(output_dataset_folder, exist_ok=True)

    # --- 训练集路径配置 ---
    training_csv = os.path.join(base_folder, 'Training_Set', 'RFMiD_Training_Labels.csv')
    training_source_images = os.path.join(base_folder, 'Training_Set', 'Training')
    # 新的训练集输出路径
    training_output_dir = os.path.join(output_dataset_folder, 'Training')
    os.makedirs(training_output_dir, exist_ok=True)
    
    # --- 测试集路径配置 ---
    testing_csv = os.path.join(base_folder, 'Test_Set', 'RFMiD_Testing_Labels.csv')
    testing_source_images = os.path.join(base_folder, 'Test_Set', 'Test')
    # 新的测试集输出路径
    testing_output_dir = os.path.join(output_dataset_folder, 'Test')
    os.makedirs(testing_output_dir, exist_ok=True)

    # --- 开始执行 ---
    print("--- 开始整理训练集 (单标签和正常) ---")
    organize_single_label_images(training_csv, training_source_images, training_output_dir)
    
    print("\n--- 开始整理测试集 (单标签和正常) ---")
    organize_single_label_images(testing_csv, testing_source_images, testing_output_dir)
    
    print(f"\n所有单标签数据已整理完毕，请查看 '{output_dataset_folder}' 文件夹。")