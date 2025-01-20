import matplotlib.pyplot as plt

# 初始化数据列表
steps = []
mems = []
cat_mems = []
num_mems = []

# 读取文件并解析数据
with open('/home/lxl/TabCutMix/eval/result/tabddpm_default_mem_ori.txt', 'r') as file:
    for line in file:
        if line.startswith('step:'):
            parts = line.split(',')
            step = int(parts[0].split(':')[1].strip())
            mem = float(parts[1].split(':')[1].strip())
            cat_mem = float(parts[2].split(':')[1].strip())
            num_mem = float(parts[3].split(':')[1].strip())
            
            steps.append(step)
            mems.append(mem)
            cat_mems.append(cat_mem)
            num_mems.append(num_mem)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制mem曲线
plt.plot(steps, mems, label='mem', color='blue')

# 绘制cat_mem曲线
# plt.plot(steps, cat_mems, label='cat_mem', color='red')

# 绘制num_mem曲线
# plt.plot(steps, num_mems, label='num_mem', color='green')

# 添加标题和标签
plt.title('Memory Usage Over Steps')
plt.xlabel('Step')
plt.ylabel('Memory')
plt.legend()

# 显示图表
plt.show()
plt.savefig('mem.png')