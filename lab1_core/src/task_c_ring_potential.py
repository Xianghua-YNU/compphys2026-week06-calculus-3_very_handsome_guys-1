try:
    import numpy as np
    import matplotlib.pyplot as plt
    print("基础库 Numpy 和 Matplotlib 已找到。")
except ImportError as e:
    print(f"缺失库: {e}")
    print("请在终端运行: pip install numpy matplotlib")
    import sys
    sys.exit()

def ring_potential_grid_no_scipy(y_grid, z_grid, a=1.0, q=1.0, num_phi=100):
    """
    使用基础 numpy 实现数值积分计算电势（不依赖 scipy）
    """
    V = np.zeros_like(y_grid)
    phi = np.linspace(0, 2 * np.pi, num_phi)
    dphi = phi[1] - phi[0]
    
    # 向量化计算数值积分
    for p in phi:
        # 距离公式: sqrt((x-a*cosφ)^2 + (y-a*sinφ)^2 + z^2), 这里取 x=0
        dist = np.sqrt(a**2 * np.cos(p)**2 + (y_grid - a*np.sin(p))**2 + z_grid**2)
        V += 1.0 / (dist + 1e-9) # 加上 epsilon 避免除零
    
    return (q / (2 * np.pi)) * V * dphi

def plot_ring_field():
    # 设置参数
    a = 1.0
    q = 1.0
    
    # 定义 yz 平面的网格
    limit = 2.0
    num_points = 50 # 降低点数以确保速度
    y = np.linspace(-limit, limit, num_points)
    z = np.linspace(-limit, limit, num_points)
    Y, Z = np.meshgrid(y, z)
    
    print("正在计算电势分布（不依赖 scipy）...")
    V = ring_potential_grid_no_scipy(Y, Z, a=a, q=q)
    
    print("正在计算电场...")
    # 使用 numpy.gradient 计算数值梯度
    dZ, dY = np.gradient(V, z, y)
    Ey = -dY
    Ez = -dZ
    
    # 3. 可视化
    print("正在生成图像...")
    plt.figure(figsize=(10, 8))
    
    # 绘制等势线
    levels = np.logspace(-0.5, 1, 15)
    contour = plt.contour(Y, Z, V, levels=levels, cmap='plasma', alpha=0.8)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # 绘制电场线 (移除了旧版本 matplotlib 不支持的 alpha 参数)
    plt.streamplot(Y, Z, Ey, Ez, color='darkblue', linewidth=1, density=1.2, arrowstyle='->')
    
    # 绘制圆环在 yz 平面的投影点
    plt.plot([a, -a], [0, 0], 'ro', markersize=10, label='Ring Section', markeredgecolor='black')
    
    plt.title(f'Electric Potential and Field Lines (No SciPy Version)')
    plt.xlabel('y (m)')
    plt.ylabel('z (m)')
    plt.colorbar(contour, label='Potential V (V)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    
    # 保存结果
    output_file = 'ring_potential_field.png'
    plt.savefig(output_file, dpi=150)
    print(f"计算完成，图像已保存为 {output_file}")
    
    # 尝试显示
    plt.show()

if __name__ == "__main__":
    plot_ring_field()
