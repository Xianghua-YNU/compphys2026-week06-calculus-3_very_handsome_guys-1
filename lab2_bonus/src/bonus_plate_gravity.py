import numpy as np


G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    """使用二维高斯-勒让德积分实现双重积分"""
    # 获取一维高斯-勒让德积分的节点和权重
    nodes, weights = np.polynomial.legendre.leggauss(n)
    
    # 将 [-1, 1] 区间的节点映射到目标区间 [a, b]
    # x = (a+b)/2 + (b-a)/2 * t
    # dx = (b-a)/2 * dt
    x_nodes = (ax + bx) / 2.0 + (bx - ax) / 2.0 * nodes
    y_nodes = (ay + by) / 2.0 + (by - ay) / 2.0 * nodes
    
    # 计算权重因子
    factor = (bx - ax) / 2.0 * (by - ay) / 2.0
    
    integral = 0.0
    # 二维累加: Σ Σ w_i * w_j * f(x_i, y_j)
    for i in range(n):
        for j in range(n):
            integral += weights[i] * weights[j] * func(x_nodes[i], y_nodes[j])
            
    return integral * factor


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    """计算方板中心正上方 z 位置的 Fz"""
    # 面密度 σ = M_plate / L^2
    sigma = M_plate / (L**2)
    
    # 定义被积函数 f(x, y) = 1 / (x^2 + y^2 + z^2)^(3/2)
    def integrand(x, y):
        return 1.0 / (x**2 + y**2 + z**2)**1.5
    
    # 积分范围为 [-L/2, L/2]
    a_lim = -L / 2.0
    b_lim = L / 2.0
    
    integral_val = gauss_legendre_2d(integrand, a_lim, b_lim, a_lim, b_lim, n)
    
    # 根据公式 Fz = G * sigma * m_particle * z * integral
    force = G * sigma * m_particle * z * integral_val
    return force


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    """返回 z_values 对应的 Fz 数组"""
    forces = [plate_force_z(z, L, M_plate, m_particle, n) for z in z_values]
    return np.array(forces)

if __name__ == "__main__":
    # 设定方板参数
    L = 10.0
    M_plate = 1.0e4
    m_particle = 1.0
    
    # 对 z ∈ [0.2, 10] 进行计算
    # 选择几个代表性的 z 值进行输出
    test_z = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"--- 方板引力场 Fz 计算结果 (L={L}m, M={M_plate}kg) ---")
    print(f"{'z (m)':<10} | {'Fz (N)':<15}")
    print("-" * 30)
    
    for z in test_z:
        fz = plate_force_z(z, L, M_plate, m_particle)
        print(f"{z:<10.1f} | {fz:<15.4e}")
    
    # 物理分析说明
    print("\n--- 物理自检 ---")
    print("1. 当 z 较小时，方板近似于无限大平面，Fz 趋于常数 2πGσ。")
    print("2. 当 z 较大时，方板近似于质点，Fz 趋于 G*M*m/z^2。")
