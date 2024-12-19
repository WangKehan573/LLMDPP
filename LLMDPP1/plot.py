import matplotlib.pyplot as plt
import numpy as np

# 数据
systems = ['20 shots', '50 shots', '100 shots']
ga_values = [0.6297, 0.7822,0.8932]  # GA values for each system
pa_values = [0.6894, 0.8229,0.9168]  # PA values for each system
ga_values_em = [0.6962,0.8260,0.9081]
pa_values_em = [0.8048,0.8854,0.9163]





# 创建图形和轴
fig, ax = plt.subplots()

# 条形图的位置
bar_width = 0.2
index = np.arange(len(systems))

# 绘制条形图
bar1 = ax.bar(index, ga_values, bar_width, label='LMMDPP_tfidf_GA')
bar2 = ax.bar(index + bar_width, pa_values, bar_width, label='LMMDPP_tfidf_PA')
bar1 = ax.bar(index+ bar_width*2, ga_values_em, bar_width, label='LMMDPP_em_GA')
bar2 = ax.bar(index + bar_width*3, pa_values_em, bar_width, label='LMMDPP_em_PA')

# 添加标签、标题和图例
ax.set_xlabel('Number of shots')
ax.set_ylabel('Scores')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(systems)
ax.legend()
ax.set_ylim(0.0,1.0)
# 显示图形
plt.show()