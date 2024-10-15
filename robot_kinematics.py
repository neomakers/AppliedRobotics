import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from ipywidgets import interactive, FloatSlider, HBox, VBox, FloatText
from IPython.display import display

# 设置连杆长度
l1 = 1.72  # 连杆1长度
l2 = 1.0  # 连杆2长度

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

# 添加角度绘制的辅助函数
def plot_angle(ax, center, theta1, theta2, radius=0.3, label=None, color='black',fontsize=12):
    # 绘制圆弧表示角度
    if theta2 >= 0:
        arc = Arc(center, radius * 2, radius * 2, angle=0, theta1=theta1, theta2=theta2 + theta1, color=color, lw=2)
    else:
        arc = Arc(center, radius * 2, radius * 2, angle=0, theta1=theta2 + theta1, theta2=theta1, color=color, lw=2)
    ax.add_patch(arc)
    
    # 计算起始边和终止边的端点
    start_x = center[0] + radius * np.cos(np.radians(theta1))
    start_y = center[1] + radius * np.sin(np.radians(theta1))
    end_x = center[0] + radius * np.cos(np.radians(theta1 + theta2))
    end_y = center[1] + radius * np.sin(np.radians(theta1 + theta2))
    delta=0.2
    theta_x = center[0] + (radius+delta) * np.cos(np.radians(theta1 + theta2 / 2))
    theta_y = center[1] + (radius+delta) * np.sin(np.radians(theta1 + theta2 / 2))
    
    # 绘制角度的初始边和终止边
    ax.plot([center[0], start_x], [center[1], start_y], color=color, lw=1, linestyle='--')  # 初始边
    ax.plot([center[0], end_x], [center[1], end_y], color=color, lw=1, linestyle='--')  # 终止边
    
    # 标注角度的标签
    if label:
        ax.text(theta_x, theta_y, label, fontsize=fontsize, color=color, ha='center', va='center')


def plot_angle_single(ax, center, theta1, theta2, radius=0.3, label=None, color='black',fontsize=12):
    # 绘制圆弧表示角度
    arc = Arc(center, radius * 2, radius * 2, angle=0, theta1=theta1, theta2=theta2, color=color, lw=2)
    ax.add_patch(arc)
    
    # 计算起始边和终止边的端点
    start_x = center[0] + radius * np.cos(np.radians(theta1))
    start_y = center[1] + radius * np.sin(np.radians(theta1))
    end_x = center[0] + radius * np.cos(np.radians(theta2))
    end_y = center[1] + radius * np.sin(np.radians(theta2))
    theta_x = center[0] - (1/12) *fontsize* radius * np.cos(np.radians(theta1+theta2 / 2))
    theta_y = center[1] - (1/12) *fontsize * radius * np.sin(np.radians(theta1+theta2 / 2))
    
    # 绘制角度的初始边和终止边
    ax.plot([center[0], start_x], [center[1], start_y], color=color, lw=1, linestyle='--')  # 初始边
    ax.plot([center[0], end_x], [center[1], end_y], color=color, lw=1, linestyle='--')  # 终止边
    
    # 标注角度的标签
    if label:
        ax.text(theta_x, theta_y, label, fontsize=fontsize, color=color, ha='center', va='center')


# 绘制函数，用于更新图形
def plot_robot_forward(theta1, theta2):
    font_size=12
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2)
    fig, ax = plt.subplots()  # 创建新的绘图上下文
    # 设置绘图区域和坐标系限制
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # 添加X轴和Y轴的标签
    ax.set_xlabel(r'$X$', fontsize=font_size)
    ax.set_ylabel(r'$Y$', fontsize=font_size)

    # 绘制X轴和Y轴
    ax.axhline(0, color='black', linewidth=0.5)  # X轴
    ax.axvline(0, color='black', linewidth=0.5)  # Y轴
    
    delta = 0.1    
    # 绘制机器人连杆
    ax.plot([0, x1, x2], [0, y1, y2], '--',color="orange", lw=4)
    
    # 标注P0、P1、P2点的位置
    ax.text(0, 0 + delta, r'$P_0$', fontsize=font_size, ha='right', color='green')  # 原点 P0
    ax.text(x1, y1 + delta, r'$P_1$', fontsize=font_size, ha='right', color='red')  # 关节点 P1
    ax.text(x2 + delta, y2 + delta, r'$P_2$', fontsize=font_size, ha='right', color='blue')  # 末端执行器 P2
    
    # 绘制并标注连杆长度
    ax.text(x1 / 2, y1 / 2 + delta, r'$l_1$', fontsize=font_size, color='red')  # 标注连杆1的长度
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='green')  # 标注连杆2的长度
    
    # 绘制关键点
    ax.plot([x1], [y1], 'ro')  # 关节1
    ax.plot([x2], [y2], 'bo')  # 末端执行器
    
    # 绘制 P1 和 P2 点的投影线
    ax.plot([x1, x1], [0, y1], 'r--')  # P1到X轴的投影线
    ax.plot([0, x1], [y1, y1], 'r--')  # P1到Y轴的投影线
    ax.plot([x2, x2], [0, y2], 'b--')  # P2到X轴的投影线
    ax.plot([0, x2], [y2, y2], 'b--')  # P2到Y轴的投影线

    # 使用 LaTeX 渲染投影点
    delta=delta
    if y1 >= 0:
        ax.text(x1, 0 - delta * 2, r'$P_{1_x}=l_1\cos(\theta_1)=%.2f$' % x1, fontsize=10, ha='center', color='red')
    else:
        ax.text(x1, 0 + delta * 2, r'$P_{1_x}=l_1\cos(\theta_1)=%.2f$' % x1, fontsize=10, ha='center', color='red')
    if y2 >= 0:
        ax.text(x2, 0 - delta * 5, r'$x=P_{2_x}=l_1\cos(\theta_{ii})=%.2f$' % x2, fontsize=10, ha='center', color='blue')
    else:
        ax.text(x2, 0 + delta * 5, r'$x=P_{2_x}=l_1\cos(\theta_{ii})=%.2f$' % x2, fontsize=10, ha='center', color='blue')

    if x1 >= 0 and y1 > 0:
        ax.text(0 - delta * 16, y1, r'$P_{1_y}=l_1\sin(\theta_1)=%.2f$' % y1, fontsize=10, va='center', color='red')
    elif x1 > 0 and y1 < 0:
        ax.text(0 - delta * 18, y1, r'$P_{1_y}=l_1\sin(\theta_1)=%.2f$' % y1, fontsize=10, va='center', color='red')
    else:
        ax.text(0 + delta * 0.5, y1, r'$P_{1_y}=l_1\sin(\theta_1)=%.2f$' % y1, fontsize=10, va='center', color='red')

    if x2 >= 0 and y2 > 0:
        ax.text(0 - delta * 16, y2, r'$y=P_{2_y}=sin(\theta_{ii})=%.2f$' % y2, fontsize=10, va='center', color='blue')
    elif y2 < 0:
        ax.text(0 - delta * 18, y2, r'$y=P_{2_y}=sin(\theta_{ii})=%.2f$' % y2, fontsize=10, va='center', color='blue')
    else:
        ax.text(0 + delta * 0.5, y2, r'$y=P_{2_y}=sin(\theta_{ii})=%.2f$' % y2, fontsize=10, va='center', color='blue')

    # 绘制角度
    plot_angle(ax, (0, 0), 0, theta1, label=r'$\theta_1$', color='red')
    plot_angle(ax, (x1, y1), theta1, theta2, label=r'$\theta_2$', color='green')
    plot_angle(ax, (x1, y1), 0,theta2+theta1, radius=0.7, label=r'$\theta_{ii}$', color='blue')

    
    ax.set_title(r'$x$: %.2f, $y$: %.2f' % (x2, y2))
    
    # 显示图像
    plt.show()


# 创建交互函数
def create_interactive_forward():
    joint0_t0 = 35
    joint1_t0 = 71
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
    interactive_plot = interactive(plot_robot_forward, theta1=theta1_slider, theta2=theta2_slider)
    display(interactive_plot)

# 绘制函数，用于更新图形
def plot_robot_forward_with_r(theta1, theta2):
    font_size=12
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2)
    fig, ax = plt.subplots()  # 创建新的绘图上下文
    # 设置绘图区域和坐标系限制
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # 添加X轴和Y轴的标签
    ax.set_xlabel(r'$X$', fontsize=font_size)
    ax.set_ylabel(r'$Y$', fontsize=font_size)

    # 绘制X轴和Y轴
    ax.axhline(0, color='black', linewidth=0.5)  # X轴
    ax.axvline(0, color='black', linewidth=0.5)  # Y轴
    
    delta = 0.1    
    # 绘制机器人连杆
    ax.plot([0, x1, x2], [0, y1, y2], 'o-', lw=4)
    ax.plot([0,x2],[0,y2],'g--',lw=2)
    
    # 标注P0、P1、P2点的位置
    ax.text(0, 0 + delta, r'$P_0$', fontsize=font_size, ha='right', color='green')  # 原点 P0
    ax.text(x1, y1 + delta, r'$P_1$', fontsize=font_size, ha='right', color='red')  # 关节点 P1
    ax.text(x2 + delta, y2 + delta, r'$P_2$', fontsize=font_size, ha='right', color='blue')  # 末端执行器 P2
    
    # 绘制并标注连杆长度
    ax.text(x1 / 2, y1 / 2 + delta, r'$l_1$', fontsize=font_size, color='red')  # 标注连杆1的长度
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='orange')  # 标注连杆2的长度
    ax.text(x2 / 2, y2 / 2 + delta, r'$r$', fontsize=font_size, color='green')  # 标注连杆1的长度
    # 绘制关键点
    ax.plot([x1], [y1], 'ro')  # 关节1
    ax.plot([x2], [y2], 'bo')  # 末端执行器
    
    # 绘制 P1 和 P2 点的投影线
    ax.plot([x1, x1], [0, y1], 'r--')  # P1到X轴的投影线
    ax.plot([0, x1], [y1, y1], 'r--')  # P1到Y轴的投影线
    ax.plot([x2, x2], [0, y2], 'b--')  # P2到X轴的投影线
    ax.plot([0, x2], [y2, y2], 'b--')  # P2到Y轴的投影线

    # 使用 LaTeX 渲染投影点
    delta=delta
    if y1 >= 0:
        ax.text(x1, 0 - delta * 2, r'$P_{1_x}=l_1\cos(\theta_1)=%.2f$' % x1, fontsize=10, ha='center', color='red')
    else:
        ax.text(x1, 0 + delta * 2, r'$P_{1_x}=l_1\cos(\theta_1)=%.2f$' % x1, fontsize=10, ha='center', color='red')
    if y2 >= 0:
        ax.text(x2, 0 - delta * 5, r'$x=%.2f$' % x2, fontsize=10, ha='center', color='blue')
    else:
        ax.text(x2, 0 + delta * 5, r'$x=%.2f$' % x2, fontsize=10, ha='center', color='blue')

    if x1 >= 0 and y1 > 0:
        ax.text(0 - delta * 16, y1, r'$P_{1_y}=l_1\sin(\theta_1)=%.2f$' % y1, fontsize=10, va='center', color='red')
    elif x1 > 0 and y1 < 0:
        ax.text(0 - delta * 18, y1, r'$P_{1_y}=l_1\sin(\theta_1)=%.2f$' % y1, fontsize=10, va='center', color='red')
    else:
        ax.text(0 + delta * 0.5, y1, r'$P_{1_y}=l_1\sin(\theta_1)=%.2f$' % y1, fontsize=10, va='center', color='red')

    if x2 >= 0 and y2 > 0:
        ax.text(0 - delta * 16, y2, r'$y=%.2f$' % y2, fontsize=10, va='center', color='blue')
    elif y2 < 0:
        ax.text(0 - delta * 18, y2, r'$y=%.2f$' % y2, fontsize=10, va='center', color='blue')
    else:
        ax.text(0 + delta * 0.5, y2, r'y=%.2f$' % y2, fontsize=10, va='center', color='blue')

    # 绘制角度
    plot_angle(ax, (0, 0), 0, theta1, label=r'$\theta_1$', color='red')
    plot_angle(ax, (x1, y1), theta1, theta2, label=r'$\theta_2$', color='orange')
    plot_angle(ax, (x1, y1), 0,theta2+theta1, radius=0.7, label=r'$\theta_{ii}=\theta_1+\theta_2$', color='blue')

    
    ax.set_title(r'$x$: %.2f, $y$: %.2f' % (x2, y2))
    
    # 显示图像
    plt.show()

# 创建交互函数
def create_interactive_forward_with_r():
    joint0_t0 = 35
    joint1_t0 = 71
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
    interactive_plot = interactive(plot_robot_forward_with_r, theta1=theta1_slider, theta2=theta2_slider)
    display(interactive_plot)


# 绘制函数，用于更新图形
def plot_robot_forward_with_range(theta1, theta2):
    font_size=12
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2)
    fig, ax = plt.subplots()  # 创建新的绘图上下文
    # 设置绘图区域和坐标系限制
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # 添加X轴和Y轴的标签
    ax.set_xlabel(r'$X$', fontsize=font_size)
    ax.set_ylabel(r'$Y$', fontsize=font_size)

    # 绘制X轴和Y轴
    ax.axhline(0, color='black', linewidth=0.5)  # X轴
    ax.axvline(0, color='black', linewidth=0.5)  # Y轴
    
    delta = 0.1    
    # 绘制机器人连杆
    ax.plot([0, x1, x2], [0, y1, y2], 'o-', lw=4)
    ax.plot([0,x2],[0,y2],'g--',lw=2)
    
    # 标注P0、P1、P2点的位置
    ax.text(0, 0 + delta, r'$P_0$', fontsize=font_size, ha='right', color='green')  # 原点 P0
    ax.text(x1, y1 + delta, r'$P_1$', fontsize=font_size, ha='right', color='red')  # 关节点 P1
    ax.text(x2 + delta, y2 + delta, r'$P_2$', fontsize=font_size, ha='right', color='blue')  # 末端执行器 P2
    
    # 绘制并标注连杆长度
    ax.text(x1 / 2, y1 / 2 + delta, r'$l_1$', fontsize=font_size, color='red')  # 标注连杆1的长度
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='orange')  # 标注连杆2的长度
    ax.text(x2 / 2, y2 / 2 + delta, r'$r$', fontsize=font_size, color='green')  # 标注连杆2的长度
    
    # 绘制关键点
    ax.plot([x1], [y1], 'ro')  # 关节1
    ax.plot([x2], [y2], 'bo')  # 末端执行器

    # 绘制角度
    plot_angle(ax, (0, 0), 0, theta1, label=r'$\theta_1$', color='red')
    plot_angle(ax, (x1, y1), theta1, theta2, label=r'$\theta_2$', color='orange')
    plot_angle(ax, (x1, y1), 0,theta2+theta1, radius=0.7, label=r'$\theta_{ii}=\theta_1+\theta_2$', color='blue')

    center=(0,0)
    radius_min=np.abs(l1-l2)
    # 创建一个圆对象
    circle_innner = plt.Circle(center, radius_min, color='purple', linestyle='-.',fill=False, linewidth=2)
    # 将圆添加到坐标系中
    ax.add_artist(circle_innner)
    ax.text(0,radius_min/2,r'inner boundary',fontsize=font_size,color='purple')
    
    radius_max=np.abs(l1+l2)
    # 创建一个圆对象
    circle_outer = plt.Circle(center, radius_max, color='red', linestyle='-.',fill=False, linewidth=2)
    # 将圆添加到坐标系中
    ax.add_artist(circle_outer)
    ax.text(0,radius_max+delta,r'outter boundary',fontsize=font_size,color='red')
    
    ax.set_title(r'$x$: %.2f, $y$: %.2f' % (x2, y2))
    
    # 显示图像
    plt.show()


def create_interactive_forward_with_range():
    joint0_t0 = 35
    joint1_t0 = 71
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
    interactive_plot = interactive(plot_robot_forward_with_range, theta1=theta1_slider, theta2=theta2_slider)
    display(interactive_plot)

# 定义逆运动学函数
def inverse_kinematics(x, y):
    # 计算 cos(theta_2)
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    # 检查 cos_theta2 是否在有效范围内（-1 到 1）
    if cos_theta2 < -1 or cos_theta2 > 1:
        raise ValueError("无法到达该位置：超出连杆的可到达范围。")
    
    # 计算两个 theta_2 的解
    theta_2_a = np.arccos(cos_theta2)
    theta_2_b = -np.arccos(cos_theta2)
    
    # 计算 theta_1 对应的解
    k1 = l1 + l2 * np.cos(theta_2_a)
    k2_a = l2 * np.sin(theta_2_a)
    k2_b = l2 * np.sin(theta_2_b)
    
    theta_1_a = np.arctan2(y, x) - np.arctan2(k2_a, k1)
    theta_1_b = np.arctan2(y, x) - np.arctan2(k2_b, k1)

    # 将弧度转换为角度
    theta_1_a_deg = np.degrees(theta_1_a)
    theta_2_a_deg = np.degrees(theta_2_a)
    
    theta_1_b_deg = np.degrees(theta_1_b)
    theta_2_b_deg = np.degrees(theta_2_b)
    
    return [theta_1_a_deg, theta_2_a_deg], [theta_1_b_deg, theta_2_b_deg]


# 绘制两个解的机器人姿态
def plot_robot_inverse(x, y):
    font_size=24
    [theta_1_a, theta_2_a], [theta_1_b, theta_2_b] = inverse_kinematics(x, y)

    # 使用 forward_kinematics 计算连杆的关节位置
    x1_a, y1_a, x2_a, y2_a = forward_kinematics(theta_1_a, theta_2_a)
    x1_b, y1_b, x2_b, y2_b = forward_kinematics(theta_1_b, theta_2_b)

    # 创建图形并绘制两个子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 第一个解（肘关节向上）
    axs[0].plot([0, x1_a, x2_a], [0, y1_a, y2_a], 'o-', lw=4)
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-3, 3)
    axs[0].set_title(f'Elbow Up: $θ_1$ = {theta_1_a:.2f}°, $θ_2$ = {theta_2_a:.2f}°',fontsize=font_size/1.5)
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    
    delta=0.5
    # 标注点P0、P1、P2
    axs[0].text(0-delta, 0, '$P_0$', fontsize=font_size, color='blue')
    axs[0].text(x1_a-delta, y1_a, '$P_1$', fontsize=font_size, color='red')
    axs[0].text(x2_a, y2_a, '$P_2$', fontsize=font_size, color='green')
        # 绘制X轴和Y轴
    axs[0].axhline(0, color='black', linewidth=0.5)  # X轴
    axs[0].axvline(0, color='black', linewidth=0.5)  # Y轴

    # 绘制角度
    # 绘制角度
    plot_angle(axs[0], (0, 0), 0, theta_1_a, label=r'$\theta_1$', color='blue',fontsize=font_size)
    plot_angle(axs[0], (x1_a, y1_a), theta_1_a, theta_2_a, label=r'$\theta_2$', color='red',fontsize=font_size)

    # 第二个解（肘关节向下）
    axs[1].plot([0, x1_b, x2_b], [0, y1_b, y2_b], 'o-', lw=4)
    axs[1].set_xlim(-3, 3)
    axs[1].set_ylim(-3, 3)
    axs[1].set_title(f'Elbow Down: $θ_1$ = {theta_1_b:.2f}°, $θ_2$ = {theta_2_b:.2f}°',fontsize=font_size/1.5)
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    
    
    # 标注点P0、P1、P2
    axs[1].text(0-delta, 0, '$P_0$', fontsize=font_size, color='blue')
    axs[1].text(x1_b-delta, y1_b, '$P_1$', fontsize=font_size, color='red')
    axs[1].text(x2_b, y2_b, '$P_2$', fontsize=font_size, color='green')

    axs[1].axhline(0, color='black', linewidth=0.5)  # X轴
    axs[1].axvline(0, color='black', linewidth=0.5)  # Y轴    
    
    # 绘制角度
    plot_angle(axs[1], (0, 0), 0, theta_1_b, label=r'$\theta_1$', color='blue',fontsize=font_size)
    plot_angle(axs[1], (x1_b, y1_b), theta_1_b, theta_2_b, label=r'$\theta_2$', color='red',fontsize=font_size)

    # 显示图形
    plt.tight_layout()
    plt.show()

def create_interactive_inverse():
    x0 = 2
    y0 = -1
    x_slider = FloatSlider(min=-3, max=3, step=.01, value=x0, description='x')
    x_input = FloatText(value=x0)
    x_box = HBox([x_slider, x_input])

    y_slider = FloatSlider(min=-3, max=3, step=.01, value=y0, description='y')
    y_input = FloatText(value=y0)
    y_box = HBox([y_slider, y_input])

    # 将滑块与输入框联动
    def update_x(*args):
        x_input.value = x_slider.value
    def update_slider1(*args):
        x_slider.value = x_input.value

    def update_y(*args):
        y_input.value = y_slider.value
    def update_slider2(*args):
        y_slider.value = y_input.value

    x_slider.observe(update_x, 'value')
    x_input.observe(update_slider1, 'value')

    y_slider.observe(update_y, 'value')
    y_input.observe(update_slider2, 'value')

    # 组合滑块与输入框的交互
    interactive_plot = interactive(plot_robot_inverse, x=x_slider, y=y_slider)
    # interactive_plot = interactive(plot_robot, theta1=x_slider, theta2=y_slider)
    display(interactive_plot)

def plot_robot_inverse_for_theta(x, y):
    font_size=20
    [theta_1_a, theta_2_a], [theta_1_b, theta_2_b] = inverse_kinematics(x, y)

    # 使用 forward_kinematics 计算连杆的关节位置
    x1_a, y1_a, x2_a, y2_a = forward_kinematics(theta_1_a, theta_2_a)


    # 创建图形并绘制两个子图
    fig, axs = plt.subplots()
    delta=0.1
    # 第一个解（肘关节向上）
    axs.plot([0, x1_a, x2_a], [0, y1_a, y2_a], 'o-', lw=4)
    axs.plot([0, x2_a], [0, y2_a], '-.', lw=4)
    axs.set_xlim(-3, 3)
    axs.set_ylim(-3, 3)
    axs.set_title(f'Elbow Up: $θ_1$ = {theta_1_a:.2f}°, $θ_2$ = {theta_2_a:.2f}°',fontsize=font_size/1.5)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    
    
    axs.text(x2_a, y2_a, '$P_2$', fontsize=font_size, color='green')
        # 绘制X轴和Y轴
    axs.axhline(0, color='black', linewidth=0.5)  # X轴
    axs.axvline(0, color='black', linewidth=0.5)  # Y轴

    
    # 绘制并标注连杆长度
    x1=x1_a
    y1=y1_a
    x2=x2_a
    y2=y2_a
    if x1>0 and y1<0:
        delta = -0.5
    else :
        delta = 0.1
    axs.text(x1 / 2, y1 / 2 + delta, r'$l_1$', fontsize=font_size, color='red')  # 标注连杆1的长度
    
    delta=0.1
    if x1+x2>0:
        axs.text((x1 + x2) / 2-2*delta, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='green')  # 标注连杆2的长度
    else:
        axs.text((x1 + x2) / 2, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='green')  # 标注连杆2的长度
    axs.text((x2) / 2, (y2) / 2 + delta, r'$r$', fontsize=font_size, color='orange')  # 标注连杆2的长度
    
    # 绘制角度
    # 绘制角度
    delta=0.1
    plot_angle(axs, (0, 0), 0, theta_1_a, label=r'$\theta_1$', color='blue',fontsize=font_size)
    delta=0.1
    plot_angle(axs, (x1_a, y1_a), theta_1_a, theta_2_a, label=r'$\theta_2$', color='red',fontsize=font_size)
    end_angle=theta_1_a+theta_2_a
    start_angle=-180+theta_1_a
    # plot_angle(axs, (x1_a, y1_a), start_angle, -start_angle+end_angle, label=r'$\phi$', color='orange',fontsize=font_size)
    plot_angle_single(axs, (x1_a, y1_a), theta_2_a+theta_1_a ,-180+theta_1_a, label=r'$\phi$', color='orange',fontsize=font_size)
    # 显示图形
    plt.tight_layout()
    plt.show()
def create_interactive_inverse_for_theta():
    x0 = 2
    y0 = -1
    x_slider = FloatSlider(min=-3, max=3, step=.01, value=x0, description='x')
    x_input = FloatText(value=x0)
    x_box = HBox([x_slider, x_input])

    y_slider = FloatSlider(min=-3, max=3, step=.01, value=y0, description='y')
    y_input = FloatText(value=y0)
    y_box = HBox([y_slider, y_input])

    # 将滑块与输入框联动
    def update_x(*args):
        x_input.value = x_slider.value
    def update_slider1(*args):
        x_slider.value = x_input.value

    def update_y(*args):
        y_input.value = y_slider.value
    def update_slider2(*args):
        y_slider.value = y_input.value

    x_slider.observe(update_x, 'value')
    x_input.observe(update_slider1, 'value')

    y_slider.observe(update_y, 'value')
    y_input.observe(update_slider2, 'value')

    # 组合滑块与输入框的交互
    interactive_plot = interactive(plot_robot_inverse_for_theta, x=x_slider, y=y_slider)
    # interactive_plot = interactive(plot_robot, theta1=x_slider, theta2=y_slider)
    display(interactive_plot)



def plot_robot_inverse_for_psi_theta(x, y):
    font_size=20
    [theta_1_a, theta_2_a], [theta_1_b, theta_2_b] = inverse_kinematics(x, y)

    # 使用 forward_kinematics 计算连杆的关节位置
    x1_a, y1_a, x2_a, y2_a = forward_kinematics(theta_1_a, theta_2_a)
    l1_ext=l1+l2*np.cos(np.deg2rad(theta_2_a))
    K=[l1_ext*np.cos(np.deg2rad(theta_1_a)),l1_ext*np.sin(np.deg2rad(theta_1_a))]

    psi=np.rad2deg(np.arctan2(y2_a,x2_a))
    epsilon=psi-theta_1_a
    # 创建图形并绘制两个子图
    fig, axses = plt.subplots(1, 2, figsize=(12, 6))
    delta=0.1
    # 第一个解（肘关节向上）
    axses[0].plot([0, x1_a, x2_a], [0, y1_a, y2_a],'-',color='orange', lw=3)
    axses[0].plot([0, x2_a], [0, y2_a], 'k-.', lw=1)
    axses[0].plot([x2_a, x2_a], [0, y2_a], '--',color='black', lw=1)
    axses[0].plot([0, x2_a], [y2_a, y2_a], '--',color='black', lw=1)
    axses[0].set_xlim(-3, 3)
    axses[0].set_ylim(-3, 3)
    axses[0].set_title(f'Elbow Up: $θ_1$ = {theta_1_a:.2f}°, $θ_2$ = {theta_2_a:.2f}°',fontsize=font_size/1.5)
    axses[0].set_xlabel('X')
    axses[0].set_ylabel('Y')
    
    
    axses[0].text(x2_a, y2_a, '$P_2$', fontsize=font_size, color='green')
        # 绘制X轴和Y轴
    axses[0].axhline(0, color='black', linewidth=0.5)  # X轴
    axses[0].axvline(0, color='black', linewidth=0.5)  # Y轴

    
    # 绘制并标注连杆长度
    x1=x1_a
    y1=y1_a
    x2=x2_a
    y2=y2_a
    if x1>0:
        delta = -0.5
    else :
        delta = 0.1
    axses[0].text(x1 / 2, y1 / 2 + delta, r'$l_1$', fontsize=font_size, color='red')  # 标注连杆1的长度
    
    delta=0.1
    if x1+x2>0:
        axses[0].text((x1 + x2) / 2-2*delta, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='green')  # 标注连杆2的长度
    else:
        axses[0].text((x1 + x2) / 2, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='green')  # 标注连杆2的长度
    
    axses[0].text(x2,-0.1, r'$x$', fontsize=font_size, color='black')  # 标注连杆2的长度
    axses[0].text(-0.1,y2, r'$y$', fontsize=font_size, color='black')  # 标注连杆2的长度
    # 绘制角度
    # 绘制角度
    delta=0.1
    plot_angle(axses[0], (0, 0), 0, theta_1_a, label=r' $\theta_1$', color='blue',fontsize=font_size)
    delta=0.1
    plot_angle(axses[0], (x1_a, y1_a), theta_1_a, theta_2_a, label=r' $\theta_2$', color='red',fontsize=font_size)
    end_angle=psi
    start_angle=theta_1_a
    plot_angle(axses[0], (0, 0), start_angle, -start_angle+end_angle,radius=0.65, label=r'$\epsilon$', color='green',fontsize=font_size)
    plot_angle(axses[0], (0, 0), 0,np.rad2deg(np.arctan2(y2_a,x2_a)),radius=1, label=r'             $\psi=\epsilon+\theta_1$', color='black',fontsize=font_size)
    # 显示图形
    axses[1].plot([0, x1_a, x2_a], [0, y1_a, y2_a],'-',color='orange', lw=2)
    axses[1].plot([0, x2_a], [0, y2_a], '-.',color='black',lw=1)
    axses[1].plot([x1_a, K[0]], [y1_a, K[1]], '-', color='red',lw=4)
    
    if x1>0 and x>0:
        delta=-0.3
    else:
        delta=0.3
    axses[1].text((x1_a+K[0])/2, (y1_a+K[1])/2+delta, r'$l_{2}\cos(\theta_2$)', fontsize=font_size, color='red')

    axses[1].text((x2_a+K[0])/2, (y2_a+K[1])/2, r'  $l_2\sin(\theta_2$)', fontsize=font_size, color='green')

    axses[1].plot([K[0],x2_a], [K[1],y2_a], '-', color='green',lw=4)
    axses[1].set_xlim(-3, 3)
    axses[1].set_ylim(-3, 3)
    axses[1].set_title(f'Elbow Up: $θ_1$ = {theta_1_a:.2f}°, $θ_2$ = {theta_2_a:.2f}°',fontsize=font_size/1.5)
    axses[1].set_xlabel('X')
    axses[1].set_ylabel('Y')
    
    
    axses[1].text(x2_a, y2_a, '$P_2$', fontsize=font_size, color='green')
        # 绘制X轴和Y轴
    axses[1].axhline(0, color='black', linewidth=0.5)  # X轴
    axses[1].axvline(0, color='black', linewidth=0.5)  # Y轴

    
    # 绘制并标注连杆长度
    x1=x1_a
    y1=y1_a
    x2=x2_a
    y2=y2_a
    if x1>0:
        delta = -0.5
    else :
        delta = 0.1
    axses[1].text(x1 / 2, y1 / 2 + delta, r'$l_1$', fontsize=font_size, color='red')  # 标注连杆1的长度
    
    delta=0.1
    if x1+x2>0:
        axses[1].text((x1 + x2) / 2-2*delta, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='green')  # 标注连杆2的长度
    else:
        axses[1].text((x1 + x2) / 2, (y1 + y2) / 2 + delta, r'$l_2$', fontsize=font_size, color='green')  # 标注连杆2的长度
    
    # 绘制角度
    # 绘制角度

    delta=0.1
    plot_angle(axses[1], (0, 0), 0, theta_1_a, label=r' $\theta_1$', color='blue',fontsize=font_size)
    delta=0.1
    plot_angle(axses[1], (x1_a, y1_a), theta_1_a, theta_2_a, label=r' $\theta_2$', color='red',fontsize=font_size)
    end_angle=psi
    start_angle=theta_1_a
    plot_angle(axses[1], (0, 0), start_angle, -start_angle+end_angle, label=r'$\epsilon$', radius=0.65,color='green',fontsize=font_size)
    # plot_angle(axses[1], (0, 0), 0,psi,radius=1, label=r'$\psi$', color='black',fontsize=font_size)
    
    plt.tight_layout()
    plt.show()


def create_interactive_inverse_for_psi_theta():
    x0 = 1.95
    y0 = 1.36
    x_slider = FloatSlider(min=-3, max=3, step=.01, value=x0, description='x')
    x_input = FloatText(value=x0)
    x_box = HBox([x_slider, x_input])

    y_slider = FloatSlider(min=-3, max=3, step=.01, value=y0, description='y')
    y_input = FloatText(value=y0)
    y_box = HBox([y_slider, y_input])

    # 将滑块与输入框联动
    def update_x(*args):
        x_input.value = x_slider.value
    def update_slider1(*args):
        x_slider.value = x_input.value

    def update_y(*args):
        y_input.value = y_slider.value
    def update_slider2(*args):
        y_slider.value = y_input.value

    x_slider.observe(update_x, 'value')
    x_input.observe(update_slider1, 'value')

    y_slider.observe(update_y, 'value')
    y_input.observe(update_slider2, 'value')

    # 组合滑块与输入框的交互
    interactive_plot = interactive(plot_robot_inverse_for_psi_theta, x=x_slider, y=y_slider)
    # interactive_plot = interactive(plot_robot, theta1=x_slider, theta2=y_slider)
    display(interactive_plot)