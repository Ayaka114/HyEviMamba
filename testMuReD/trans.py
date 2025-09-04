import os
import pandas as pd
import shutil
import glob
from tqdm import tqdm

def organize_new_format_images(csv_path, image_source_dir, output_base_dir):
    """
    根据新的CSV格式，将“正常”和“单标签”的图片分类到新的文件夹中。
    新格式特点：有一个明确的 'NORMAL' 列。
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
        print(f"读取CSV文件 '{csv_path}' 时出错: {e}")
        return

    # 获取所有疾病列的名称
    # 关键：从所有列中排除 'ID' 和 'NORMAL'
    all_columns = df.columns.tolist()
    disease_columns = [col for col in all_columns if col.upper() not in ['ID', 'NORMAL']]
    print(f"已识别出 {len(disease_columns)} 个疾病标签。")
    
    # 遍历CSV的每一行
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"整理 {os.path.basename(image_source_dir)}"):
        image_id = row['ID']
        
        # 查找图片文件，兼容不同扩展名
        source_image_paths = glob.glob(os.path.join(image_source_dir, f"{image_id}.*"))
        if not source_image_paths:
            continue  # 如果找不到图片，静默跳过
        source_image_path = source_image_paths[0]

        target_category = None

        # 1. 检查是否为“正常”类别 (NORMAL 列为 1)
        if 'NORMAL' in row and row['NORMAL'] == 1:
            target_category = "NORMAL"
        
        # 2. 如果不是“正常”，则检查是否为“单标签疾病”
        else:
            # 提取所有疾病标签的数据
            disease_labels = row[disease_columns]
            # 计算1的个数
            num_diseases = (disease_labels == 1).sum()
            
            # 如果恰好只有一个疾病
            if num_diseases == 1:
                # 找到那个疾病的名称
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
    # --- !! 重要：请根据你的新数据集修改以下路径 !! ---
    
    # 假设你的新数据集文件夹名为 'New_Dataset'
    base_folder = '/home1/zhouhao/MedMamba_ywj/datasets/MuReD/Multi-Label Retinal Diseases (MuReD) Dataset' 
    
    # 新的输出目录
    output_dataset_folder = '/home1/zhouhao/MedMamba_ywj/datasets/MuReD/Multi-Label Retinal Diseases (MuReD) Dataset'
    os.makedirs(output_dataset_folder, exist_ok=True)

    # --- 训练集路径配置 ---
    # 假设训练CSV名为 'train_labels.csv'，图片在 'train_images' 文件夹
    training_csv = os.path.join(base_folder, 'train_data.csv')
    training_source_images = os.path.join(base_folder, 'images')
    training_output_dir = os.path.join(output_dataset_folder, 'Training')
    os.makedirs(training_output_dir, exist_ok=True)
    
    # --- 测试集路径配置 ---
    # 假设测试CSV名为 'test_labels.csv'，图片在 'test_images' 文件夹
    testing_csv = os.path.join(base_folder, 'val_data.csv')
    testing_source_images = os.path.join(base_folder,'images')
    testing_output_dir = os.path.join(output_dataset_folder, 'Test')
    os.makedirs(testing_output_dir, exist_ok=True)

    # --- 开始执行 ---
    print("--- 开始整理训练集 (新格式) ---")
    organize_new_format_images(training_csv, training_source_images, training_output_dir)
    
    print("\n--- 开始整理测试集 (新格式) ---")
    organize_new_format_images(testing_csv, testing_source_images, testing_output_dir)
    
    print(f"\n所有单标签数据已整理完毕，请查看 '{output_dataset_folder}' 文件夹。")