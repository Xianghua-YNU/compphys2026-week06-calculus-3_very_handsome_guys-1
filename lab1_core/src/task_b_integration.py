import math


def debye_integrand(x: float) -> float:
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)


def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    # TODO B1: 实现复合梯形积分
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h


def simpson_composite(f, a: float, b: float, n: int) -> float:
    # TODO B2: 实现复合 Simpson 积分，并检查 n 为偶数
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals (n).")
    
    h = (b - a) / n
    s = f(a) + f(b)
    
    # 奇数项系数为 4
    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
        
    # 偶数项系数为 2
    for i in range(2, n, 2):
        s += 2 * f(a + i * h)
        
    return s * h / 3.0


def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    # TODO B3: 计算 Debye 积分 I(theta_d/T)
    y = theta_d / T
    if method == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method == "simpson":
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError("Unknown integration method. Use 'trapezoid' or 'simpson'.")


if __name__ == "__main__":
    # TODO B4: 比较两种方法在相同 n 下的误差差异
    # 我们以 x^4 在 [0, 1] 上的积分为例，精确值为 0.2
    f_test = lambda x: x**4
    exact = 0.2
    n_list = [10, 20, 40, 80, 160]
    
    print(f"{'n':>5} | {'Trapezoid Error':>20} | {'Simpson Error':>20} | {'Ratio (T/S)':>10}")
    print("-" * 65)
    
    for n in n_list:
        val_t = trapezoid_composite(f_test, 0.0, 1.0, n)
        val_s = simpson_composite(f_test, 0.0, 1.0, n)
        
        err_t = abs(val_t - exact)
        err_s = abs(val_s - exact)
        
        ratio = err_t / err_s if err_s > 0 else float('inf')
        
        print(f"{n:5d} | {err_t:20.10e} | {err_s:20.10e} | {ratio:10.2f}")

    print("\n结论：")
    print("- 复合梯形公式具有二阶精度 O(h^2)，当 n 翻倍时误差约减小为原来的 1/4。")
    print("- 复合 Simpson 公式具有四阶精度 O(h^4)，当 n 翻倍时误差约减小为原来的 1/16。")
    print("- 在相同 n 下，Simpson 法的精度远高于梯形法。")
