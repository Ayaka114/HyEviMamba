import os
from PIL import Image
from tqdm import tqdm

def find_corrupted_images(image_folder):
    """
    遍历指定文件夹中的所有图片，找出损坏或无法打开的文件。
    """
    corrupted_files = []
    # 获取文件夹下所有文件的列表
    image_list = [os.path.join(root, file)
                  for root, _, files in os.walk(image_folder)
                  for file in files]
    
    print(f"开始检查文件夹 '{image_folder}' 中的 {len(image_list)} 个文件...")

    # 使用tqdm显示进度条
    for image_path in tqdm(image_list, desc="正在检查图片"):
        try:
            # 尝试打开图片并加载数据
            with Image.open(image_path) as img:
                img.load()
        except Exception as e:
            # 如果出现任何异常（如OSError, SyntaxError等），则认为是损坏的
            print(f"\n发现损坏文件: {image_path}")
            print(f"  错误原因: {e}")
            corrupted_files.append(image_path)
            
    if not corrupted_files:
        print("\n检查完成，未发现损坏的图片文件。")
    else:
        print(f"\n检查完成！共发现 {len(corrupted_files)} 个损坏文件。")
    
    return corrupted_files

if __name__ == "__main__":
    # !!! 重要：请将这里的路径修改为你的实际训练集图片路径 !!!
    # 例如，根据你之前的脚本，可能是 'Single_Label_Dataset/Training'
    training_data_path = "/home1/zhouhao/MedMamba_ywj/datasets/RFMiD2.0/new/Training" 
    
    find_corrupted_images(training_data_path)