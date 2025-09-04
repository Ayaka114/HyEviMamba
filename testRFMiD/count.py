import pandas as pd
import os

def count_rows_with_multiple_diseases(csv_path):
    """
    统计CSV文件中，除前两列外，值为1的个数大于1的行数。

    参数:
    - csv_path (str): 要分析的CSV文件的路径。

    返回:
    - int: 满足条件的行数。如果文件不存在或出错则返回-1。
    """
    if not os.path.exists(csv_path):
        print(f"错误: 文件未找到 -> {csv_path}")
        return -1

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 选取除了'ID'和'Disease_Risk'之外的所有疾病列（即从第2列索引开始到最后）
        disease_columns = df.iloc[:, 2:]
        
        # 沿着行的方向（axis=1）计算每一行中1的总数
        # 这会为每一行生成一个“疾病总数”
        sum_of_diseases = disease_columns.sum(axis=1)
        
        # 判断这个总数是否大于1，并计算满足条件的行的数量
        count = (sum_of_diseases > 1).sum()
        
        return count
        
    except Exception as e:
        print(f"处理文件 {csv_path} 时发生错误: {e}")
        return -1

if __name__ == "__main__":
    # --- 配置文件路径 ---
    base_folder = '/home1/zhouhao/MedMamba_ywj/datasets/RFMiD2.0'
    
    # 训练集和测试集的CSV文件路径
    training_csv = os.path.join(base_folder, 'Training_Set', 'RFMiD_Training_Labels.csv')
    testing_csv = os.path.join(base_folder, 'Test_Set', 'RFMiD_Testing_Labels.csv')

    # --- 执行并打印结果 ---
    print("开始统计多标签图片数量...")

    # 统计训练集
    training_count = count_rows_with_multiple_diseases(training_csv)
    if training_count != -1:
        print(f"训练集 ({os.path.basename(training_csv)}) 中，患有超过一种疾病的行数: {training_count}")

    # 统计测试集
    testing_count = count_rows_with_multiple_diseases(testing_csv)
    if testing_count != -1:
        print(f"测试集 ({os.path.basename(testing_csv)}) 中，患有超过一种疾病的行数: {testing_count}")