# 03. 하한 경계(Lower Bound)

## 🎯 핵심 질문
- 1차 방법(gradient 기반)으로 도달할 수 있는 최고의 수렴 속도는 무엇인가?
- Nesterov 가속이 정말 최적인가?
- 더 나은 1차 방법이 존재할 수 있는가?

## 🔍 왜 이 이론이 AI에서 중요한가
Lower bound는 알고리즘이 이론적으로 도달할 수 없는 성능의 벽을 정의합니다. Nesterov 가속이 lower bound를 달성한다는 것은 "더 이상 개선의 여지가 없다"는 뜻입니다. 이는 왜 특정 알고리즘을 쓰는지, 언제 포기할지를 결정하게 합니다.

## 📐 수학적 선행 조건
- **First-order oracle**: 알고리즘은 $\nabla f(x_0), \ldots, \nabla f(x_{k-1})$만 접근 가능
- **Worst-case 분석**: 모든 가능한 함수에 대한 최악의 경우
- **Nesterov의 보조정리**: 함수 구성 기법
- **행렬 고유값**: 이차형식의 성질

## 📖 직관적 이해

**Oracle 모델**: 알고리즘은 "검은 상자" 함수 $f$에 어떤 점 $x$를 주면 $\nabla f(x)$를 받습니다. 이 정보만으로 작동합니다.

**최악의 함수 구성**: 특정 함수를 설계하여, 아무리 똑똑한 알고리즘도 $k$번의 쿼리 후 최솟값을 찾을 수 없도록 만듭니다.

**정보-이론적 해석**: $k$번의 쿼리는 최대 $k$차원의 정보를 제공합니다. 하지만 $n$차원 공간에서는 부족합니다.

## ✏️ 엄밀한 정의

**정의 (First-order Oracle)**:
주어진 점 $x$에서 $(f(x), \nabla f(x))$를 반환하는 Oracle. 알고리즘은 이 정보를 누적하여 최적해를 추정합니다.

**정의 (하한)**:
함수 클래스 $\mathcal{F}$와 알고리즘이 주어졌을 때, 모든 알고리즘에 대해
$$\inf_{\text{alg}} \sup_{f \in \mathcal{F}} (f(x_k) - f^*) \ge \text{Lower Bound}$$

**정의 (Worst-case Complexity)**:
$k$번의 gradient 쿼리로 달성 가능한 최소 오차.

## 🔬 정리와 증명

**정리 1 (Nemirovski-Yudin 하한 - 볼록 경우)**

L-smooth이고 볼록인 함수의 클래스에서, 모든 1차 알고리즘은
$$\inf_{\text{alg}} \sup_{f \in \mathcal{F}} (f(x_k) - f^*) \ge \frac{3L\|x_0 - x^*\|^2}{32k^2}$$

를 만족합니다.

*증명 스케치 (최악의 함수 구성)*:

Nesterov의 구성 방법을 사용합니다. 다음과 같은 함수를 정의합니다:

$$f(x) = \frac{\gamma}{4}\sum_{i=1}^{k}(k-i+1)(x_i - x_{i+1})^2 + \frac{\gamma}{2}\|x\|^2$$

여기서 변수는 $(x_1, \ldots, x_k)$입니다.

**Step 1: 함수 분석**

이 함수는:
- 강볼록 (λ = γ)
- 부드러움 (L = O(γ·k))
- 최솟값은 $x^* = 0$

**Step 2: 정보 전파 지연**

$\nabla f$를 계산하면:
$$\frac{\partial f}{\partial x_i} \propto (x_i - x_{i+1}) + (x_{i-1} - x_i)$$

첫 번째 쿼리 $\nabla f(x^0)$는 $x^0_1$에만 의존합니다.
두 번째 쿼리 $\nabla f(x^1)$는 $x^0_1, x^0_2$의 정보만 반영합니다.

일반적으로 $k$번의 쿼리는 최대 처음 $k$개 변수에만 영향을 줄 수 있습니다.

**Step 3: 크기 추정**

$x_{k+1}, \ldots, x_n$의 위치에 대한 정보가 $k$번의 쿼리로는 전파되지 않습니다.

따라서 $\|x - x^*\| \gtrsim \|x_{k+1:n}\| \gtrsim 1$

**Step 4: 오차 하한**

함수값의 오차:
$$f(x) - f^* \ge C \cdot L \|x_k - x^*\|^2 \ge C' \cdot \frac{L\|x_0 - x^*\|^2}{k^2}$$

상수들을 정리하면 $O(1/k^2)$ 하한입니다. □

**정리 2 (Strongly Convex 하한)**

L-smooth이고 μ-strongly convex인 함수에 대해:
$$\inf_{\text{alg}} \sup_{f \in \mathcal{F}} \|x_k - x^*\|^2 \ge \left(1 - \sqrt{\frac{\mu}{L}}\right)^{2k} C\|x_0 - x^*\|^2$$

*증명*:

Strongly convex 경우의 최악의 함수:
$$f(x) = \frac{\mu}{2}x_1^2 + \sum_{i=1}^{k-1} \left[\frac{\mu}{2}x_{i+1}^2 + \frac{L}{2}(x_i - x_{i+1})^2\right]$$

정보 전파 논리는 동일하지만, 강볼록성으로 인한 감소 요인이 추가됩니다.

결과: 선형 수렴 인수 $(1 - \sqrt{\mu/L})$가 하한입니다. □

**명제 (Nesterov의 최적성)**

Nesterov 가속 경사법은:
- 볼록에서 $O(1/k^2)$ 달성 (정리 1과 일치)
- Strongly convex에서 $(1 - \sqrt{\mu/L})^k$ 달성 (정리 2와 일치)

따라서 Nesterov는 두 하한을 모두 달성하는 최적 알고리즘입니다.

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# Nesterov가 구성한 최악의 함수 (단순화 버전)
# f(x) = (1/2) * (x^T A x) where A는 특별한 구조

def construct_worst_case_matrix(k, L=1.0, mu=1.0):
    """
    Worst-case 함수: f(x) = (1/2) x^T A x
    A의 고유값: [μ, μ + ε₁, μ + ε₂, ..., L]
    정보 전파 지연으로 인해 낮은 고유값 방향으로 수렴이 느림
    """
    n = k + 5  # 여유 차원 추가
    
    # Tridiagonal matrix with delayed information propagation
    A = np.diag(np.linspace(mu, L, n))
    
    # 트라이디애곤 구조: 정보가 천천히 전파됨
    for i in range(n-1):
        A[i, i+1] = A[i+1, i] = (L - mu) / (2 * n)
    
    return A

def quadratic_f(x, A):
    return 0.5 * x @ A @ x

def grad_quadratic_f(x, A):
    return A @ x

# 알고리즘들
def gd_oracle_limited(x0, A, k_queries):
    """k번의 gradient 쿼리만 사용하는 GD"""
    x = x0.copy()
    L = np.max(eigvalsh(A))
    eta = 1 / L
    
    queries = []
    losses = []
    
    for q in range(k_queries):
        grad = grad_quadratic_f(x, A)
        queries.append((x.copy(), grad.copy()))
        
        loss = quadratic_f(x, A)
        losses.append(loss)
        
        x = x - eta * grad
    
    return x, np.array(losses), queries

def nesterov_oracle_limited(x0, A, k_queries):
    """k번의 gradient 쿼리를 사용하는 Nesterov"""
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    
    L = np.max(eigvalsh(A))
    eta = 1 / L
    
    queries = []
    losses = []
    
    for q in range(k_queries):
        grad = grad_quadratic_f(y, A)
        queries.append((y.copy(), grad.copy()))
        
        loss = quadratic_f(y, A)
        losses.append(loss)
        
        x_new = y - eta * grad
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        momentum = (t - 1) / t_new
        y = x_new + momentum * (x_new - x)
        
        x = x_new
        t = t_new
    
    return y, np.array(losses), queries

# 시뮬레이션
np.random.seed(42)
k_queries = 50
L_param = 100.0
mu_param = 1.0

A = construct_worst_case_matrix(k_queries, L=L_param, mu=mu_param)
x0 = np.random.randn(A.shape[0])

x_gd, losses_gd, queries_gd = gd_oracle_limited(x0, A, k_queries)
x_nest, losses_nest, queries_nest = nesterov_oracle_limited(x0, A, k_queries)

# 이론적 하한과 상한
k_range = np.arange(1, k_queries + 1)
lower_bound_convex = (3 * L_param * np.linalg.norm(x0)**2) / (32 * k_range**2)
upper_bound_nesterov = (2 * L_param * np.linalg.norm(x0)**2) / (k_range**2)

# Strongly convex 하한
rho = 1 - np.sqrt(mu_param / L_param)
lower_bound_strongly_convex = rho**(2*k_range) * np.linalg.norm(x0)**2

# 시각화 1: 수렴 곡선 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 로그-로그 (O(1/k²) 확인)
ax = axes[0, 0]
ax.loglog(k_range, losses_gd, 'b-o', label='GD', linewidth=2, markersize=4)
ax.loglog(k_range, losses_nest, 'r-s', label='Nesterov', linewidth=2, markersize=4)
ax.loglog(k_range, lower_bound_convex, 'g--', label='Lower Bound O(1/k²)', linewidth=2)
ax.loglog(k_range, upper_bound_nesterov, 'r--', label='Nesterov Upper Bound', linewidth=2, alpha=0.7)
ax.set_xlabel('Iterations k', fontsize=11)
ax.set_ylabel('f(x_k) - f*', fontsize=11)
ax.set_title('Worst-case: Log-Log Scale', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# (2) 반로그 (선형성 확인)
ax = axes[0, 1]
ax.semilogy(k_range, losses_gd, 'b-o', label='GD', linewidth=2, markersize=4)
ax.semilogy(k_range, losses_nest, 'r-s', label='Nesterov', linewidth=2, markersize=4)
ax.semilogy(k_range, lower_bound_convex, 'g--', label='Lower Bound', linewidth=2)
ax.set_xlabel('Iterations k', fontsize=11)
ax.set_ylabel('f(x_k) - f* (log)', fontsize=11)
ax.set_title('Worst-case: Semi-log Scale', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (3) GD vs Lower Bound 비율
ax = axes[1, 0]
ratio_gd = losses_gd / np.maximum(lower_bound_convex, 1e-15)
ax.plot(k_range, ratio_gd, 'b-o', linewidth=2, markersize=4)
ax.axhline(y=1, color='r', linestyle='--', label='Lower Bound', linewidth=2)
ax.set_xlabel('Iterations k', fontsize=11)
ax.set_ylabel('GD Loss / Lower Bound', fontsize=11)
ax.set_title('GD Gap to Optimality', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (4) Nesterov vs Lower Bound
ax = axes[1, 1]
ratio_nest = losses_nest / np.maximum(lower_bound_convex, 1e-15)
ax.plot(k_range, ratio_nest, 'r-s', linewidth=2, markersize=4)
ax.axhline(y=1, color='g', linestyle='--', label='Lower Bound', linewidth=2)
ax.set_xlabel('Iterations k', fontsize=11)
ax.set_ylabel('Nesterov Loss / Lower Bound', fontsize=11)
ax.set_title('Nesterov Achieves Lower Bound', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lower_bound_verification.png', dpi=150)
plt.show()

# 분석: Nesterov가 lower bound에 얼마나 가까운가?
print("=" * 60)
print("Lower Bound 달성도 분석")
print("=" * 60)

for k_check in [10, 20, 30, 40, 50]:
    lb = lower_bound_convex[k_check-1]
    nest = losses_nest[k_check-1]
    gap = nest / lb if lb > 1e-15 else np.inf
    print(f"k={k_check:2d}: GD Loss={losses_gd[k_check-1]:.2e}, "
          f"Nest={nest:.2e}, LB={lb:.2e}, Gap={gap:.2f}x")

# 시뮬레이션 2: 정보 전파 시각화
print("\n" + "=" * 60)
print("정보 전파 분석 (처음 10개 차원에서의 오차)")
print("=" * 60)

# GD에서 처음 k 차원이 업데이트되는 정도
for q in range(min(10, k_queries)):
    x_gd_q, _, _ = gd_oracle_limited(x0, A, q+1)
    x_nest_q, _, _ = nesterov_oracle_limited(x0, A, q+1)
    
    # 처음 q+1 차원의 오차
    error_gd = np.linalg.norm(x_gd_q[:q+1] - np.zeros(q+1))
    error_nest = np.linalg.norm(x_nest_q[:q+1] - np.zeros(q+1))
    
    print(f"Query {q+1:2d}: GD error={error_gd:.4f}, Nesterov error={error_nest:.4f}")
```

## 🔗 AI/ML 연결

**알고리즘 개발 한계**: Lower bound는 새로운 알고리즘의 필요성을 보여줍니다. $O(1/k^2)$보다 빠른 1차 방법은 불가능하므로, 2차 정보(Hessian)가 필요합니다 (Newton 방법).

**문제 구조 활용**: 특수한 구조를 가진 문제 (강볼록, 합계 구조)는 더 나은 알고리즘이 가능합니다.

**현실의 갭**: 신경망 손실은 비볼록이므로 lower bound가 직접 적용되지 않지만, 개념은 알고리즘의 근본적 한계를 이해하는 데 도움됩니다.

## ⚖️ 가정과 한계

**가정 1**: L-smooth, 볼록 함수 클래스에만 적용됩니다.

**가정 2**: 첫 번째 미분 정보만 접근 가능하다고 가정합니다. 고차 정보가 있으면 더 빠를 수 있습니다.

**가정 3**: Worst-case 분석이므로, 실제 문제는 더 빠를 수 있습니다 (평균적으로).

**한계**: 비볼록, 비smooth 함수에는 직접 적용 불가능합니다.

## 📌 핵심 정리

1. **First-order oracle 제약**: 그래디언트 정보만으로 최고 성능은 제한됩니다.
2. **Convex lower bound**: $\Omega(1/k^2)$ — 더 빠를 수 없습니다.
3. **Strongly convex lower bound**: $\Omega((1-\sqrt{\mu/L})^k)$ 지수 수렴.
4. **Nesterov 최적성**: 두 경우 모두 lower bound 달성.
5. **2차 방법 필요**: 더 빠른 수렴 → Hessian 활용 (Newton).

## 🤔 생각해볼 문제

**문제 1**: Nemirovski-Yudin 구성에서 정보가 왜 천천히 전파되는가? Tridiagonal 구조의 역할을 설명하시오.

<details>
<summary>힌트 및 해설</summary>

함수 $f(x) = \sum (x_i - x_{i+1})^2$에서 $\nabla f$는:
$$\frac{\partial f}{\partial x_i} \propto (x_i - x_{i-1}) + (x_i - x_{i+1})$$

$\nabla f(x^0)$는 $x^0_i, x^0_{i-1}, x^0_{i+1}$의 위치에만 의존합니다. 이는 인접 차원 간의 의존성만 있다는 뜻입니다.

$k$번의 쿼리 후에도 맨 끝 차원 $x_n$의 영향은 시작점까지 "전파"되어야 하는데, 선형 체인에서는 $k$ 스텝이 필요합니다.

</details>

**문제 2**: Lower bound가 $O(1/k^2)$인데 왜 GD는 $O(1/k)$에 머무르는가?

<details>
<summary>힌트 및 해설</summary>

GD는 매 스텝 현재 위치에서만 그래디언트를 봅니다. Nesterov는 "미래 위치"를 예측하므로, 더 효율적으로 정보를 활용합니다.

Lower bound 함수에서, GD는 $k$번째 쿼리 후에도 뒤쪽 차원들을 업데이트하지 못합니다. 반면 Nesterov의 모멘텀은 과거 방향을 적극 활용하여 거리를 더 크게 나아갑니다.

</details>

**문제 3**: Strong convexity가 있으면 lower bound가 선형 수렴 $(1-\sqrt{\mu/L})^k$로 바뀐다. 왜 지수 형태인가?

<details>
<summary>힌트 및 해설</summary>

Strong convexity는 최솟값 주변에서 "용수철" 같은 복원력을 제공합니다. 

집합 $\{x: \|x - x^*\| \le r\}$ 밖에서 시작하면, 경사가 강하므로 빠르게 $r$로 들어갑니다.

한 번 안에 들어오면, 강볼록성으로 인해 매 스텝 오차가 기하급수적으로 감소합니다: $\|x_{k+1} - x^*\| \le \rho \|x_k - x^*\|$.

따라서 지수 형태 $\rho^k$가 나옵니다.

</details>

<div align="center">
| [◀ 02. Nesterov 가속 경사법](./02-nesterov-accelerated.md) | [📚 README](../README.md) | [04. Newton 방법 ▶](./04-newton-method.md) |
</div>
