import re
import matplotlib.pyplot as plt
import numpy as np

# 提取各个 F1 值（binary_f1, weighted_f1, roc_auc, accuracy）
def extract_f1_score(tag, section):
    pattern = r'"' + tag + '"\s*:\s*([\d.]+)'
    return re.search(pattern, section).group(1)

methods = ["TabDDPM", "TabDDPM+TameSyn", "TabDDPM+ClassifierGuided", "TabDDPM+TameSyn_0"]

path_list = [
    'eval/result/tabddpm_shoppers_ori.txt',
    'eval/result/tabddpm_shoppers_-5_90w.txt',
    'eval/result/tabddpm_shoppers_05_005_05.txt',
    'eval/result/shoppers_-3_0.txt'
]

datas = {}

for i, path in enumerate(path_list):
    with open(path, 'r') as f:
        txt_content = f.read()
        # 提取mem_all、cat_ori、num_ori
        mem_all = re.search(r'mem_all: ([\d.]+)', txt_content).group(1)
        cat_ori = re.search(r'cat_ori: ([\d.]+)', txt_content).group(1)
        num_ori = re.search(r'num_ori: ([\d.]+)', txt_content).group(1)

        # 从各个部分提取 F1 等值
        best_f1 = extract_f1_score("binary_f1", re.search(r'"best_f1_scores": {.*?}', txt_content, re.DOTALL).group())
        best_weighted_f1 = extract_f1_score("binary_f1", re.search(r'"best_weighted_scores": {.*?}', txt_content, re.DOTALL).group())
        best_auroc_f1 = extract_f1_score("binary_f1", re.search(r'"best_auroc_scores": {.*?}', txt_content, re.DOTALL).group())
        best_acc_f1 = extract_f1_score("binary_f1", re.search(r'"best_acc_scores": {.*?}', txt_content, re.DOTALL).group())
        best_avg_f1 = extract_f1_score("binary_f1", re.search(r'"best_avg_scores": {.*?}', txt_content, re.DOTALL).group())


        # 提取Density的Shape和Trend
        shape = re.search(r'Shape: ([\d.]+)', txt_content).group(1)
        trend = re.search(r'Trend: ([\d.]+)', txt_content).group(1)

        # 提取Alpha_Precision_all和Beta_Recall_all
        alpha_precision = re.search(r'Alpha_Precision_all = ([\d.]+)', txt_content).group(1)
        beta_recall = re.search(r'Beta_Recall_all = ([\d.]+)', txt_content).group(1)

        # 组织数据
        data = {
            "Mem": mem_all,
            "F1": best_f1,
            "WE": best_weighted_f1,
            "AUR": best_auroc_f1,
            "ACC": best_acc_f1,
            "AVG": best_avg_f1,
            "Shape": shape,
            "Trend": trend,
            "Alpha": alpha_precision,
            "Beta": beta_recall
        }
        datas[methods[i]] = data
    
# print(datas)

# 数据处理
metrics = ['Mem', 'F1', 'WE', 'AUR', 'ACC', 'AVG', 'Shape', 'Trend', 'Alpha', 'Beta']
models = list(datas.keys())
values = {model: [float(datas[model][metric]) for metric in metrics] for model in models}

# 可视化设置
plt.figure(figsize=(18, 10))
x = np.arange(len(metrics))
width = 0.2  # 柱宽
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 绘制柱状图
for i, model in enumerate(models):
    offset = width * (i - 1.5)  # 居中排列
    plt.bar(x + offset, values[model], width=width, label=model, color=colors[i])

# 图表装饰
plt.title('Model Performance Comparison', fontsize=16)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.xticks(x, metrics, rotation=45)
plt.ylim(0, 1.2)
plt.grid(axis='y', alpha=0.4)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# 显示图表
plt.tight_layout()
plt.show()
plt.savefig('bar_chart.pdf')