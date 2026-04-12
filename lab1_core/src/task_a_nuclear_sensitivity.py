import numpy as np

def rate_3alpha(T: float) -> float:
    """计算 3-alpha 反应率 q(T)"""
    if T <= 0:
        return 0.0
    T8 = T / 1.0e8
    # 避免数值溢出或过小
    try:
        return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)
    except OverflowError:
        return np.inf

def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    """使用前向差分计算 dq/dT | T0"""
    # 增量 ΔT = h * T0
    delta_T = h * T0
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T0 + delta_T)
    return (q1 - q0) / delta_T

def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    """计算温度敏感性指数 nu(T0) = (T0 / q(T0)) * (dq/dT)"""
    q0 = rate_3alpha(T0)
    if q0 == 0 or np.isinf(q0):
        return np.nan
    
    dq_dT = finite_diff_dq_dT(T0, h)
    # nu = (T0 / q0) * dq_dT
    return (T0 / q0) * dq_dT

def nu_table(T_values, h: float = 1e-8):
    """计算并打印温度点列表的 nu 值"""
    results = []
    for T in T_values:
        nu = sensitivity_nu(T, h)
        results.append((T, nu))
    return results

if __name__ == "__main__":
    # A.4 必算温度点
    T_targets = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    
    print("--- 3-alpha 反应率温度敏感性指数 nu 结果 ---")
    table = nu_table(T_targets)
    
    for T, nu in table:
        # A.4 输出格式要求: 1.000e+08 K : nu = 41.03
        print(f"{T:.3e} K : nu = {nu:.2f}")

    # A.5 结果自检趋势说明
    print("\n--- 结果自检趋势 ---")
    print("1. T0 = 10^8 K 附近: nu 很大 (约 41)")
    print("2. 随温度升高: nu 显著降低")
    print("3. 高温端可能出现负值 (约 -1 到 -2 量级)")
