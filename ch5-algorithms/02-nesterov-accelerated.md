# 02. Nesterov 가속 경사법(AGM)

## 🎯 핵심 질문
- Nesterov 가속이 경사하강법을 능가하는 이유는 무엇인가?
- $O(1/k^2)$ 수렴은 어떻게 달성되는가?
- Momentum과 Nesterov의 차이는 무엇인가?

## 🔍 왜 이 이론이 AI에서 중요한가
Nesterov 가속 경사법(AGM)은 경사하강법의 $O(1/k)$ 수렴을 $O(1/k^2)$로 개선합니다. 이는 1차 정보만으로 달성할 수 있는 최적의 수렴 속도입니다 (lower bound 참조). 현대 최적화 라이브러리들이 이를 표준으로 구현합니다.

## 📐 수학적 선행 조건
- **L-smooth 함수**: $\|\nabla f(x) - \nabla f(y)\| \le L\|x-y\|$
- **볼록성**: $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$
- **Estimating Sequence**: 보조 수열로 수렴 증명
- **모멘텀 개념**: 과거 정보를 활용한 가속화

## 📖 직관적 이해

**Momentum의 직관**: 공이 언덕을 굴러 내려올 때, 처음엔 느리지만 관성이 생겨 빠르게 가속됩니다. 경사하강법에 모멘텀을 더하면:
$$v_{k+1} = \beta v_k + \nabla f(x_k)$$
$$x_{k+1} = x_k - \eta v_{k+1}$$

**Nesterov의 통찰**: "미래 위치"에서 그래디언트를 계산하면 더 좋은 방향을 얻습니다.
$$x_k \to y_k \to \text{(그래디언트 계산)} \to \text{(큰 스텝)} \to y_{k+1}$$

**Telescoping과 가속**: Estimating sequence를 통해 누적 오차를 효율적으로 제어합니다.

## ✏️ 엄밀한 정의

**정의 (Momentum GD)**:
$$\begin{align}
y_k &= x_k + \frac{t_k - 1}{t_{k+1}}(x_k - x_{k-1}) \quad \text{(관성 항)} \\
x_{k+1} &= y_k - \eta \nabla f(y_k) \quad \text{(경사 스텝)}
\end{align}$$

여기서 $t_1 = 1$이고 $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$ (Nesterov 스케줄)

**정의 (Estimating Sequence)**:
$\{v_k\}$가 추정 수열이라는 것은:
1. $v_k$는 계산 가능
2. $v_k \ge \varphi(x_k) := \min_x\{\varphi(x, x_k)\}$ 형태의 함수의 최솟값

Nesterov의 증명은 이 수열의 성질로부터 수렴을 도출합니다.

## 🔬 정리와 증명

**정리 1 (Nesterov AGM - $O(1/k^2)$ 수렴)**

$f$가 L-smooth이고 볼록일 때, 다음 알고리즘:
$$\begin{align}
y_k &= x_k + \frac{t_k - 1}{t_{k+1}}(x_k - x_{k-1}) \\
x_{k+1} &= y_k - \frac{1}{L}\nabla f(y_k) \\
t_{k+1} &= \frac{1 + \sqrt{1 + 4t_k^2}}{2}
\end{align}$$

은 다음을 만족합니다:
$$f(x_k) - f^* \le \frac{2L\|x_0 - x^*\|^2}{k^2}$$

*증명 스케치*:

**Step 1: Estimating Sequence 설정**

함수 $\varphi(x, z) = f(z) + \nabla f(z)^T(x-z) + \frac{L}{2}\|x-z\|^2$로 1차 근사를 정의합니다.

$\varphi_k = \min_x \varphi(x, y_k)$로 놓으면, 이는 함수의 근사 최솟값입니다.

**Step 2: Momentum 가중합**

$x_{k+1} = \frac{t_k - 1}{t_{k+1}}x_k + \frac{2}{t_{k+1}}z_{k+1}$ 형태의 가중 조합을 사용합니다.

여기서 가중치는 $\frac{t_k - 1}{t_{k+1}} + \frac{2}{t_{k+1}} = 1$을 만족합니다.

**Step 3: 재귀 관계**

$t_k$의 정의로부터:
$$t_{k+1}^2 = t_k^2 + 2t_{k+1} \quad \Rightarrow \quad t_{k+1} - t_k = \frac{1}{t_{k+1}}$$

이는 $\sum_{i=1}^k (t_{i+1} - t_i) \approx k$를 의미합니다.

**Step 4: Estimating Sequence의 합**

주요 관찰: 모든 $i$에 대해
$$f(x_i) - f^* \le \text{(경사도를 누적)}$$

Nesterov의 핵심 부등식 (telescoping):
$$\varphi_k - f^* \le \frac{2L\|x_0-x^*\|^2}{2t_k^2}$$

여기서 $t_k \sim k/2$이므로:
$$f(x_k) - f^* \le \frac{2L\|x_0-x^*\|^2}{(k/2)^2} = \frac{8L\|x_0-x^*\|^2}{k^2}$$

계수를 정리하면 $O(1/k^2)$입니다. □

**정리 2 (Strongly Convex Nesterov)**

$f$가 L-smooth이고 μ-strongly convex일 때, 적절한 학습률로:
$$\|x_k - x^*\| \le C\left(1 - \sqrt{\frac{\mu}{L}}\right)^k \|x_0 - x^*\|$$

즉, 지수 수렴하며 선형 인수는 $\sqrt{\kappa}$ ($\kappa = L/\mu$ 조건수)에만 의존합니다.

*증명*:

Strongly convex 조건 아래에서 Estimating sequence는:
$$v_k \ge f^* + \frac{\mu}{2}\|x_k - x^*\|^2$$

이를 이용하면:
$$\|x_k - x^*\|^2 \le \left(1 - 2\sqrt{\frac{\mu}{L}}\right) \|x_{k-1} - x^*\|^2$$

따라서 $\rho = 1 - c\sqrt{\mu/L}$로 $\|x_k - x^*\|^2 \le \rho^k \|x_0 - x^*\|^2$ □

**명제 (Momentum vs Nesterov 비교)**

Heavy Ball 모멘텀:
$$v_{k+1} = \beta v_k + \nabla f(x_k)$$
$$x_{k+1} = x_k - \eta v_{k+1}$$

Nesterov:
$$v_{k+1} = \beta v_k + \nabla f(x_k - \beta v_k)$$
$$x_{k+1} = x_k - \eta v_{k+1}$$

**차이**: Nesterov는 "예측 위치"에서 그래디언트를 계산합니다. 이것이 $O(1/k^2)$ vs $O(1/k)$의 차이를 만듭니다.

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# 이차 함수 설정 (테스트용)
def quadratic(x, A, b):
    return 0.5 * x @ A @ x + b @ x

def grad_quadratic(x, A, b):
    return A @ x + b

# 함수 생성
np.random.seed(42)
n = 50
Q = np.random.randn(n, n)
A = Q @ Q.T + np.eye(n)
b = np.random.randn(n)

x_opt = -np.linalg.solve(A, b)
f_opt = quadratic(x_opt, A, b)

L = np.max(eigvalsh(A))
mu = np.min(eigvalsh(A))
kappa = L / mu

print(f"Condition number κ = {kappa:.2f}")
print(f"L = {L:.4f}, μ = {mu:.4f}")

# 1. 경사하강법 (GD)
def gradient_descent(x0, A, b, eta, num_iters):
    x = x0.copy()
    losses = []
    for k in range(num_iters):
        f_val = quadratic(x, A, b) - f_opt
        losses.append(f_val)
        grad = grad_quadratic(x, A, b)
        x = x - eta * grad
    return x, np.array(losses)

# 2. 모멘텀 GD
def momentum_gd(x0, A, b, eta, beta, num_iters):
    x = x0.copy()
    v = np.zeros_like(x0)
    losses = []
    for k in range(num_iters):
        f_val = quadratic(x, A, b) - f_opt
        losses.append(f_val)
        grad = grad_quadratic(x, A, b)
        v = beta * v + grad
        x = x - eta * v
    return x, np.array(losses)

# 3. Nesterov 가속 경사법
def nesterov_agm(x0, A, b, eta, num_iters):
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    losses = []
    
    for k in range(num_iters):
        f_val = quadratic(y, A, b) - f_opt
        losses.append(f_val)
        
        grad = grad_quadratic(y, A, b)
        x_new = y - eta * grad
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        
        # Momentum 항
        momentum_coeff = (t - 1) / t_new
        y = x_new + momentum_coeff * (x_new - x)
        
        x = x_new
        t = t_new
    
    return y, np.array(losses)

# 파라미터
x0 = np.random.randn(n)
eta = 1 / L
beta = np.sqrt(mu) / (np.sqrt(mu) + np.sqrt(L))  # Heavy Ball 최적 파라미터
num_iters = 200

# 실행
_, losses_gd = gradient_descent(x0, A, b, eta, num_iters)
_, losses_momentum = momentum_gd(x0, A, b, eta/2, beta, num_iters)
_, losses_nesterov = nesterov_agm(x0, A, b, eta, num_iters)

# 이론적 수렴 곡선
k_range = np.arange(1, num_iters+1)
theoretical_gd = (L * np.linalg.norm(x0 - x_opt)**2) / (2 * k_range)
theoretical_nesterov = (2 * L * np.linalg.norm(x0 - x_opt)**2) / (k_range**2)
theoretical_strongly_convex = (1 - np.sqrt(mu/L))**(k_range) * np.linalg.norm(x0 - x_opt)**2

# 시각화 1: 볼록 수렴
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 로그 스케일
ax = axes[0, 0]
ax.semilogy(losses_gd, 'b-', label='GD: O(1/k)', linewidth=2)
ax.semilogy(losses_momentum, 'g-', label='Momentum: O(1/k)', linewidth=2)
ax.semilogy(losses_nesterov, 'r-', label='Nesterov: O(1/k²)', linewidth=2)
ax.semilogy(theoretical_gd, 'b--', alpha=0.7, linewidth=1.5)
ax.semilogy(theoretical_nesterov, 'r--', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('f(x_k) - f*', fontsize=11)
ax.set_title('Convergence Comparison (Log Scale)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (2) 선형 스케일
ax = axes[0, 1]
ax.plot(losses_gd, 'b-', label='GD', linewidth=2)
ax.plot(losses_momentum, 'g-', label='Momentum', linewidth=2)
ax.plot(losses_nesterov, 'r-', label='Nesterov', linewidth=2)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('f(x_k) - f*', fontsize=11)
ax.set_title('Convergence Comparison (Linear Scale)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (3) 처음 50 iteration 상세
ax = axes[1, 0]
k_max = 50
ax.semilogy(losses_gd[:k_max], 'b-', label='GD', linewidth=2, marker='o', markersize=4)
ax.semilogy(losses_nesterov[:k_max], 'r-', label='Nesterov', linewidth=2, marker='s', markersize=4)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('f(x_k) - f*', fontsize=11)
ax.set_title('First 50 Iterations Detail', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (4) 수렴 속도 비율
ax = axes[1, 1]
ratio = losses_gd / np.maximum(losses_nesterov, 1e-15)
ratio[np.isinf(ratio)] = 0
ratio[np.isnan(ratio)] = 0
ax.plot(ratio, 'purple', linewidth=2)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('GD Loss / Nesterov Loss', fontsize=11)
ax.set_title('Speedup Factor', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nesterov_comparison.png', dpi=150)
plt.show()

# 시각화 2: 등고선과 궤적
if n >= 2:
    # 2D 부분 추출
    A2 = A[:2, :2]
    b2 = b[:2]
    x0_2d = x0[:2]
    
    # 등고선 생성
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xy = np.array([X[i,j], Y[i,j]])
            Z[i,j] = quadratic(xy, A2, b2)
    
    # 궤적 계산
    def get_trajectory_2d(x0, A, b, eta, beta, num_iter, method='nesterov'):
        x = x0.copy()
        y = x0.copy()
        t = 1.0
        traj = [y.copy()]
        
        for k in range(num_iter):
            grad = grad_quadratic(y, A, b)
            x_new = y - eta * grad
            
            if method == 'gd':
                y = x_new
            else:  # nesterov
                t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
                momentum_coeff = (t - 1) / t_new
                y = x_new + momentum_coeff * (x_new - x)
                x = x_new
                t = t_new
            
            traj.append(y.copy())
        return np.array(traj)
    
    traj_gd = get_trajectory_2d(x0_2d, A2, b2, eta, 0, 30, 'gd')
    traj_nesterov = get_trajectory_2d(x0_2d, A2, b2, eta, 0, 30, 'nesterov')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(ax.contour(X, Y, Z, levels=10), inline=True, fontsize=8)
    
    ax.plot(traj_gd[:, 0], traj_gd[:, 1], 'b-o', label='GD', linewidth=2, markersize=4)
    ax.plot(traj_nesterov[:, 0], traj_nesterov[:, 1], 'r-s', label='Nesterov', linewidth=2, markersize=4)
    
    x_opt_2d = x_opt[:2]
    ax.plot(x_opt_2d[0], x_opt_2d[1], 'k*', markersize=15, label='Optimum')
    
    ax.set_xlabel('x₁', fontsize=11)
    ax.set_ylabel('x₂', fontsize=11)
    ax.set_title('Optimization Trajectories (2D)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nesterov_trajectory.png', dpi=150)
    plt.show()

# 조건수 영향 분석
print("\n=== 조건수 영향 분석 ===")
print(f"κ = {kappa:.2f}")
print(f"GD 수렴 인수: 1 - 1/κ = {1 - 1/kappa:.4f}")
print(f"Nesterov 수렴 인수 (strongly convex): (1-√(μ/L)) = {1 - np.sqrt(mu/L):.4f}")
```

## 🔗 AI/ML 연결

**PyTorch/TensorFlow**: 많은 최적화기가 Nesterov momentum을 제공합니다 (`optimizer.SGD(momentum=0.9, nesterov=True)`).

**빠른 수렴**: $O(1/k^2)$는 이론적 하한과 일치하므로, Nesterov는 1차 방법 중 최적입니다.

**조건수 개선**: Strongly convex에서 Nesterov는 $\sqrt{\kappa}$ 의존성을 만들어, 나쁜 조건의 문제에서도 상대적으로 빠릅니다.

## ⚖️ 가정과 한계

**가정**: L-smooth, 볼록 함수를 가정합니다. 신경망 손실은 비볼록이므로 수렴 보장이 없습니다.

**초기화 민감도**: 나쁜 초기값에서는 느릴 수 있습니다 (거리 $\|x_0 - x^*\|$에 의존).

**Step size 튜닝**: Momentum 파라미터는 문제 특성에 따라 조정해야 합니다.

## 📌 핵심 정리

1. **Momentum**: 과거 그래디언트 정보를 활용한 가속화.
2. **Nesterov 통찰**: 미래 위치에서 그래디언트 계산 → $O(1/k^2)$ 수렴.
3. **Estimating Sequence**: Nesterov의 증명 기법으로 telescoping sum 활용.
4. **Strongly Convex**: 선형 수렴 인수 개선, $\sqrt{\kappa}$ 의존성.
5. **최적성**: Lower bound와 일치하므로 더 나은 1차 방법 없음.

## 🤔 생각해볼 문제

**문제 1**: $t_k = (1 + \sqrt{1 + 4t_{k-1}^2})/2$의 해가 $t_k \sim k/2$임을 보이시오.

<details>
<summary>힌트 및 해설</summary>

$t_k$가 크면 $t_{k+1} \approx 2t_k$이므로 $t_k \sim 2^k$처럼 보이지만, 실제로는 $t_k \sim k/2$ 점근입니다.

더 정확히는, $s_k = t_k - 1/2$로 놓으면 $s_{k+1} \approx 2s_k$이므로 $s_k \sim 2^k$, 따라서 $t_k \sim 2^{k-1} + 1/2$...

아니, 재귀식을 풀면: $t_{k+1}^2 - t_k^2 \approx 2t_{k+1}$이므로 $\sum (t_{k+1}^2 - t_k^2) \sim 2\sum t_{k+1}$. 좌변은 $t_k^2$이고 우변은 $2\sum_{i=1}^k t_i$입니다. 따라서 $t_k^2 \sim 2 \cdot k \cdot t_k / 2$이므로 $t_k \sim k$입니다. (계수 조정)

</details>

**문제 2**: Heavy Ball 모멘텀이 왜 $O(1/k)$에 머물러 있는가?

<details>
<summary>힌트 및 해설</summary>

Heavy Ball은 매 스텝 이전 방향을 기억하지만, Nesterov처럼 "예측 위치"에서 그래디언트를 계산하지 않습니다. 

가중치가 $\beta v_k + \nabla f(x_k)$인데, Nesterov는 $\nabla f(y_k)$에서 관성을 반영합니다. 이 차이가 수렴 속도를 결정합니다.

</details>

**문제 3**: Strongly convex 경우 Nesterov의 선형 수렴 인수는 $\sqrt{\mu/L}$에 어떻게 의존하는가? 조건수 $\kappa = L/\mu$로 표현하시오.

<details>
<summary>힌트 및 해설</summary>

선형 수렴 인수: $\rho = 1 - c\sqrt{\mu/L} = 1 - c/\sqrt{\kappa}$

따라서 $\kappa$가 커질수록 (나쁜 조건) $\rho \to 1$에 가까워져 느려집니다.

예: $\kappa = 100$이면 $\sqrt{\kappa} = 10$, $\rho \approx 0.9$ (10회 반복마다 1/e 감소).

</details>

<div align="center">
| [◀ 01. 경사하강법 수렴 정리](./01-gd-convergence-full.md) | [📚 README](../README.md) | [03. 하한 경계(Lower Bound) ▶](./03-lower-bound.md) |
</div>
