import pandas as pd

# 读取结果文件
df = pd.read_csv("eval_results.csv")

# 找出所有包含 "Macro" 的列
macro_cols = [c for c in df.columns if "Macro" in c]

# 只保留需要的列：Model、Macro相关列、Accuracy
sub_df = df[["Model"] + macro_cols + ["Accuracy"]]
sub_df = sub_df.sort_values(by="Model")
# 保留小数点后四位
pd.options.display.float_format = "{:.4f}".format

# 打印结果
print(sub_df.to_string(index=False))
