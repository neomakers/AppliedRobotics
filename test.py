
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider, HBox, VBox, FloatText
from IPython.display import display

# 设置连杆长度
l1 = 1.0  # 连杆1长度
l2 = 0.5  # 连杆2长度

# 计算连杆末端位置的正运动学函数
def forward_kinematics(theta1, theta2):
    theta1 = np.radians(theta1)
    theta2 = np.radians(theta2)
    
    # 第一个关节的位置
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    
    # 末端执行器的位置
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    
    return x1, y1, x2, y2

# 绘制函数，用于更新图形
def plot_robot(theta1, theta2):
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2)
    
    # 清除前一帧
    plt.clf()
    
    # 设置绘图区域和坐标系限制
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    delta=0.1    
    # 绘制机器人连杆
    plt.plot([0, x1, x2], [0, y1, y2], 'o-', lw=4)
    
    # 标注P0、P1、P2点的位置
    plt.text(0, 0+ delta, 'P0', fontsize=12, ha='right', color='blue')  # 原点 P0
    plt.text(x1, y1+ delta, 'P1', fontsize=12, ha='right', color='red')  # 关节点 P1
    plt.text(x2+ delta, y2+ delta, 'P2', fontsize=12, ha='right', color='green')  # 末端执行器 P2
    
    # 绘制并标注连杆长度

    plt.text(x1 / 2, y1 / 2 + delta, 'l1', fontsize=12, color='purple')  # 标注连杆1的长度
    plt.text((x1 + x2) / 2, (y1 + y2) / 2 + delta, 'l2', fontsize=12, color='orange')  # 标注连杆2的长度
    
    # 绘制关键点
    plt.plot([x1], [y1], 'ro')  # 关节1
    plt.plot([x2], [y2], 'bo')  # 末端执行器
    
    plt.title(f"Theta1: {theta1}°, Theta2: {theta2}°")
    
    # 显示图像
    plt.show()

# 创建滑块和输入框组合的交互控件
joint0_t0=53
joint1_t0=-26
theta1_slider = FloatSlider(min=-180, max=180, step=1, value=joint0_t0, description='Theta1')
theta1_input = FloatText(value=joint0_t0)
theta1_box = HBox([theta1_slider, theta1_input])

theta2_slider = FloatSlider(min=-180, max=180, step=1, value=joint1_t0, description='Theta2')
theta2_input = FloatText(value=joint1_t0)
theta2_box = HBox([theta2_slider, theta2_input])

# 将滑块与输入框联动
def update_theta1(*args):
    theta1_input.value = theta1_slider.value
def update_slider1(*args):
    theta1_slider.value = theta1_input.value

def update_theta2(*args):
    theta2_input.value = theta2_slider.value
def update_slider2(*args):
    theta2_slider.value = theta2_input.value

theta1_slider.observe(update_theta1, 'value')
theta1_input.observe(update_slider1, 'value')

theta2_slider.observe(update_theta2, 'value')
theta2_input.observe(update_slider2, 'value')

# 组合滑块与输入框的交互
interactive_plot = interactive(plot_robot, theta1=theta1_slider, theta2=theta2_slider)
display(interactive_plot)