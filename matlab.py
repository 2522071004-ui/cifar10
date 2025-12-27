from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import scipy.io

# 定义TensorBoard日志目录路径
logdir = 'logs/Adam'

# 创建EventAccumulator对象
event_acc = EventAccumulator(logdir)
event_acc.Reload()

# 获取所有标签（Summary标量的标签名称）
tags = event_acc.Tags()['scalars']

# 创建字典来存储每个标签的数据
data = {}

# 逐个读取每个标签的数据
for tag in tags:
    # 从EventAccumulator中提取标签的值（步数和对应的标量值）
    scalar_events = event_acc.Scalars(tag)
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]

    # 存储数据到字典
    data[tag] = {'steps': steps, 'values': values}

# 将数据保存为MAT文件
mat_data = {}
for tag, values in data.items():
    mat_data[tag] = {'steps': np.array(values['steps']),
                     'values': np.array(values['values'])}

scipy.io.savemat('Adamdata.mat', mat_data)