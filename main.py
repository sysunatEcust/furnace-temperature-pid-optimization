import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('temperature.csv')
time = data['time'].values
temperature = data['temperature'].values
voltage = data['volte'].values

# 绘制输入输出曲线
plt.figure(figsize=(10, 6))
plt.plot(time, temperature, label='Output (Temperature)', color='b')
plt.plot(time, voltage, label='Input (Voltage)', color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Temperature / Voltage')
plt.title('Step Response of Heating Furnace')
plt.legend()
plt.grid(True)
plt.show()

# =======================
# 两点法参数辨识
# =======================

# 阶跃输入起始与结束值
delta_U = voltage[0]

# 阶跃响应起始与稳态值
Y1 = temperature[0]
Y2 = temperature[21600]
delta_Y = Y2 - Y1

# 系统增益 K
K = delta_Y / delta_U
print(f'系统增益 K = {K:.4f}')

# 估算延迟时间 L 和时间常数 T
# 方法：找到输出达到 28.3% 和 63.2% 的时间点
Y_28 = Y1 + 0.283 * delta_Y
Y_63 = Y1 + 0.632 * delta_Y

t_28 = time[np.argmin(np.abs(temperature - Y_28))]
t_63 = time[np.argmin(np.abs(temperature - Y_63))]

L = t_28
T = 1.5 * (t_63 - t_28)

print(f'延迟时间 L = {L:.4f} s')
print(f'时间常数 T = {T:.4f} s')

# =======================
# 一阶惯性+纯滞后模型验证
# G(s) = K * exp(-Ls) / (Ts + 1)
# =======================
from scipy.signal import lti, step

system = lti([K], [T, 1])
t_sim, y_sim = step(system, T=time)
y_sim = np.where(t_sim + L >= 0, y_sim, 0)  # 添加纯滞后效应
y_sim = y_sim + Y1  # 加上初始值

# 绘制辨识模型输出与真实数据对比
plt.figure(figsize=(10, 6))
plt.plot(time, temperature, label='Real Output', color='b')
plt.plot(time, y_sim, label='Model Output', color='g', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Temperature')
plt.title('Model Verification')
plt.legend()
plt.grid(True)
plt.show()
