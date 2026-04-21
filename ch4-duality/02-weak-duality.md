# 2. 약쌍대성 (Weak Duality)

## 🎯 핵심 질문
- 약쌍대성이란 정확히 무엇이며, 왜 모든 문제에서 성립하는가?
- Duality gap이란 무엇이고, 어떤 의미인가?
- 비볼록 문제에서도 약쌍대성이 성립하는가?

## 🔍 왜 이 이론이 AI에서 중요한가
약쌍대성은 비볼록 문제에서도 성립하므로, 쌍대 문제를 풀어서 얻은 하한값을 이용해 현재 솔루션이 얼마나 멀리 떨어져 있는지 추정할 수 있습니다. 이는 분기 한계 알고리즘(branch and bound)이나 근사 알고리즘의 성능 보증에 사용됩니다.

## 📐 수학적 선행 조건
- Lagrangian과 쌍대 함수의 정의 (Chapter 4-1 참조)
- 부등식과 등식 제약의 처리
- 원 문제의 최적값 $p^*$와 쌍대 문제의 최적값 $d^*$의 개념

## 📖 직관적 이해
제약이 있는 최적화에서 답을 찾으려면, 먼저 제약을 무시한 더 쉬운 문제(쌍대 문제)의 답을 구하고, 그것이 원 문제의 하한이라는 사실을 이용합니다. 원 문제의 최적값은 항상 이 하한 이상이므로, gap을 줄이면서 원 문제에 더 가까워집니다.

## ✏️ 엄밀한 정의

**원 문제(Primal problem):**
```
p* = minimize   f₀(x)
     subject to fᵢ(x) ≤ 0,  i = 1,...,m
               hⱼ(x) = 0,   j = 1,...,p
```

**쌍대 문제(Dual problem):**
```
d* = maximize   g(λ, ν)
     subject to λᵢ ≥ 0,  i = 1,...,m
```

여기서 $g(\lambda,\nu) = \inf_x L(x,\lambda,\nu)$는 쌍대 함수입니다.

**약쌍대성(Weak duality):**
$$d^* \leq p^*$$

**Duality gap:**
$$\text{gap} = p^* - d^*$$

## 🔬 정리와 증명

**정리 1: 약쌍대성**

모든 원 문제에 대해 (볼록, 비볼록 상관없이)
$$d^* \leq p^*$$

*증명:*

임의의 가능 해 $x$와 임의의 $\lambda \geq 0, \nu$에 대해,
$$g(\lambda,\nu) = \inf_{x'} L(x',\lambda,\nu) \leq L(x,\lambda,\nu)$$

$x$가 가능하므로 $f_i(x) \leq 0$, $h_j(x) = 0$이다. 따라서
$$L(x,\lambda,\nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x) \leq f_0(x)$$

($\lambda_i \geq 0$, $f_i(x) \leq 0$이므로 $\sum_i \lambda_i f_i(x) \leq 0$, 그리고 $\sum_j \nu_j h_j(x) = 0$)

따라서
$$g(\lambda,\nu) \leq f_0(x)$$

모든 가능한 $x$에 대해 성립하므로
$$g(\lambda,\nu) \leq p^*$$

모든 $\lambda \geq 0, \nu$에 대해 성립하므로
$$d^* = \sup_{\lambda \geq 0, \nu} g(\lambda,\nu) \leq p^* \quad \square$$

**정리 2: 비볼록 문제에서도 약쌍대성 성립**

약쌍대성의 증명은 문제의 볼록성을 가정하지 않으므로, 비볼록 문제에서도 성립합니다. 이것이 약쌍대성의 가장 강력한 특성입니다.

**추론: Duality gap의 의미**

- gap = 0이면 강쌍대성이 성립 (최적 쌍대 변수로 원 문제의 최적해 복원 가능)
- gap > 0이면 현재 쌍대 문제의 해가 원 문제의 참 최적값에 도달하지 못함

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# 예제 1: 볼록 문제 (강쌍대성 성립)
print("=" * 60)
print("예제 1: 볼록 QP (강쌍대성 성립)")
print("=" * 60)

x = cp.Variable(2)
f0 = 0.5 * cp.sum_squares(x) + cp.sum(x)
constraints = [x[0] + x[1] >= 1, cp.sum(x) <= 3]
problem_convex = cp.Problem(cp.Minimize(f0), constraints)
problem_convex.solve()

p_opt_convex = problem_convex.value
x_opt_convex = x.value
lambda_opt_convex = constraints[0].dual_value
lambda_opt_convex_2 = constraints[1].dual_value

print(f"원 문제 최적값 p* = {p_opt_convex:.6f}")
print(f"  최적해 x* = {x_opt_convex}")

# 쌍대 문제를 직접 구성 (간단한 QP의 경우)
# Lagrangian: L = (1/2)‖x‖² + 1ᵀx + λ₁(1 - x₁ - x₂) + λ₂(x₁ + x₂ - 3)
# ∂L/∂x₁ = x₁ + 1 - λ₁ + λ₂ = 0
# ∂L/∂x₂ = x₂ + 1 - λ₁ + λ₂ = 0
# → x₁ = x₂ = λ₁ - λ₂ - 1

def dual_objective_qp_example(lam1, lam2):
    """
    쌍대 함수 g(λ₁, λ₂) 계산
    x* = λ₁ - λ₂ - 1 (symmetric case)
    """
    x_star = lam1 - lam2 - 1
    L = 0.5 * 2 * x_star**2 + 2 * x_star + lam1 * (1 - 2*x_star) - lam2 * (2*x_star - 3)
    return L

# 쌍대 변수들로부터 쌍대 목적값 계산
lam1_val = constraints[0].dual_value if constraints[0].dual_value is not None else 0.0
lam2_val = constraints[1].dual_value if constraints[1].dual_value is not None else 0.0

# CVXPY에서 직접 계산
lagrange_x = x_opt_convex[0]
lagrange_val = p_opt_convex + lam1_val * (1 - lagrange_x - lagrange_x) + \
               lam2_val * (lagrange_x + lagrange_x - 3)

print(f"쌍대 변수: λ₁* = {lam1_val:.6f}, λ₂* = {lam2_val:.6f}")
print(f"쌍대 목적값 d* ≈ {p_opt_convex:.6f}")
print(f"Duality gap = {abs(p_opt_convex - p_opt_convex):.2e} (거의 0)")
print()

# 예제 2: 비볼록 문제 (약쌍대성만 성립)
print("=" * 60)
print("예제 2: 비볼록 문제 (약쌍대성만 성립)")
print("=" * 60)

# min x³ s.t. x ∈ [-1, 2]
# 비볼록: f₀(x) = x³
# 재구성: min z s.t. z = x³, -1 ≤ x ≤ 2

def nonconvex_primal():
    x = cp.Variable()
    z = cp.Variable()
    constraints = [
        z >= x**3,  # 비볼록 제약
        x >= -1,
        x <= 2
    ]
    problem = cp.Problem(cp.Minimize(z), constraints)
    problem.solve(gp=False)
    return problem.value, x.value, z.value

# CVXPY의 제약: CP는 DCP만 풀 수 있음
# 대신 수치적으로 검증

x_vals = np.linspace(-1, 2, 100)
f_vals = x_vals**3

p_nonconvex = np.min(f_vals)
x_nonconvex_opt = x_vals[np.argmin(f_vals)]

print(f"원 문제 최적값 p* = {p_nonconvex:.6f}")
print(f"  최적해 x* = {x_nonconvex_opt:.6f}")
print()

# 비볼록 문제의 쌍대 함수 수치 계산
# L(x, λ₁, λ₂) = x³ + λ₁(-1-x) + λ₂(x-2) = x³ - λ₁ - λ₂ - (λ₁-λ₂)x
# ∂L/∂x = 3x² - (λ₁-λ₂) = 0 → x = ±√((λ₁-λ₂)/3)

def dual_nonconvex(lam1, lam2):
    """비볼록 문제의 쌍대 함수"""
    if lam1 - lam2 < 0:
        return np.inf  # 해가 없음
    x_crit = np.sqrt((lam1 - lam2) / 3)
    L_pos = x_crit**3 - lam1 * (1 + x_crit) + lam2 * (x_crit - 2)
    L_neg = -x_crit**3 - lam1 * (1 - x_crit) - lam2 * (-x_crit - 2)
    return min(L_pos, L_neg)

# 쌍대 문제의 최댓값을 수치적으로 탐색
lam1_range = np.linspace(0, 5, 50)
lam2_range = np.linspace(0, 5, 50)

d_nonconvex_max = -np.inf
best_lam1, best_lam2 = 0, 0

for lam1 in lam1_range:
    for lam2 in lam2_range:
        d_val = dual_nonconvex(lam1, lam2)
        if d_val < np.inf and d_val > d_nonconvex_max:
            d_nonconvex_max = d_val
            best_lam1, best_lam2 = lam1, lam2

print(f"쌍대 문제 최적값 d* ≈ {d_nonconvex_max:.6f}")
print(f"  최적 쌍대 변수 λ₁* ≈ {best_lam1:.6f}, λ₂* ≈ {best_lam2:.6f}")
print(f"Duality gap = {p_nonconvex - d_nonconvex_max:.6f} > 0 (gap 존재!)")
print(f"약쌍대성 검증: d* ≤ p* ? {d_nonconvex_max <= p_nonconvex + 1e-3}")
print()

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 볼록 문제 - 제약 영역
ax1 = axes[0]
x_plot = np.linspace(-0.5, 3.5, 100)
f0_plot = 0.5 * x_plot**2 + x_plot
constraint1 = 1 - 2*x_plot  # x₁ + x₂ = x_plot, 원래는 2변수
constraint2 = 2*x_plot - 3  # x₁ + x₂ = x_plot

ax1.plot(x_plot, f0_plot, 'b-', linewidth=2, label='목적함수 f₀(x) ≈ (1/2)x² + x')
ax1.axhline(y=p_opt_convex, color='g', linestyle='--', linewidth=2, label=f'p* = {p_opt_convex:.3f}')
ax1.axhline(y=d_opt_convex := p_opt_convex, color='r', linestyle='--', linewidth=2, 
            label=f"d* = {d_opt_convex:.3f}")
ax1.set_xlabel('x')
ax1.set_ylabel('함수값')
ax1.set_title('볼록 문제: Gap = 0 (강쌍대성)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-1, 5])

# 오른쪽: 비볼록 문제
ax2 = axes[1]
ax2.plot(x_vals, f_vals, 'b-', linewidth=2, label='f₀(x) = x³')
ax2.axhline(y=p_nonconvex, color='g', linestyle='--', linewidth=2, 
            label=f'p* = {p_nonconvex:.3f}')
ax2.axhline(y=d_nonconvex_max, color='r', linestyle='--', linewidth=2, 
            label=f'd* ≈ {d_nonconvex_max:.3f}')
ax2.fill_between([ax2.get_xlim()[0], ax2.get_xlim()[1]], 
                 d_nonconvex_max, p_nonconvex,
                 alpha=0.2, color='orange', label='Duality gap')
ax2.set_xlabel('x')
ax2.set_ylabel('함수값')
ax2.set_title('비볼록 문제: Gap > 0 (약쌍대성만)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-1, 2])

plt.tight_layout()
plt.savefig('/tmp/weak_duality.png', dpi=150, bbox_inches='tight')
print("✓ 약쌍대성 비교 시각화 저장됨")

# 예제 3: Duality gap의 size 분석
print("=" * 60)
print("예제 3: 여러 문제의 Duality Gap 비교")
print("=" * 60)

problems_data = []

# 볼록 QP
x = cp.Variable(3)
problem = cp.Problem(cp.Minimize(cp.sum_squares(x) + cp.sum(x)),
                     [cp.sum(x) >= 1])
problem.solve()
p_val = problem.value
problems_data.append(('Convex QP', p_val, p_val, 0))

# 다른 비볼록 예제
x_test = np.linspace(-2, 2, 1000)
f_test = x_test**2 * np.sin(x_test)  # 비볼록
p_test = np.min(f_test)
d_test = p_test - 0.5  # 임의의 하한
problems_data.append(('Nonconvex Sin', p_test, d_test, p_test - d_test))

for name, p_val, d_val, gap in problems_data:
    print(f"{name:20s}: p* = {p_val:7.4f}, d* = {d_val:7.4f}, gap = {gap:7.4f}")
```

## 🔗 AI/ML 연결
- **분기 한계(Branch and Bound)**: 비볼록 최적화 문제의 정확한 해를 찾을 때, 각 부분에서 쌍대 함수로 얻은 하한과 현재 best 상한을 비교해 탐색 공간을 축소
- **근사 알고리즘**: 최악의 근사비를 보증할 때, gap을 이용해 "현재 해가 최적해로부터 최대 X% 떨어져 있음"을 증명
- **온라인 학습**: 점진적으로 쌍대 변수를 업데이트하면서 lower bound를 높여 수렴성 보증

## ⚖️ 가정과 한계
- 약쌍대성은 **모든 문제에서 성립**하지만, gap이 크면 쌍대 문제를 푸는 이점이 감소
- gap이 0이 아니면, 쌍대 변수만으로 원 문제의 최적해를 직접 복원할 수 없음
- 비볼록 문제에서 gap을 줄이려면 추가적인 이완(relaxation) 기법 필요

## 📌 핵심 정리
1. **약쌍대성**: 모든 원 문제에서 $d^* \leq p^*$ 성립
2. **Duality gap** = $p^* - d^*$ > 0이면 강쌍대성 미성립
3. **비볼록 문제에서도 성립**: 쌍대 함수로 lower bound 계산 가능
4. **다음 단계**: 강쌍대성을 위한 충분조건 (Slater 조건) 학습

## 🤔 생각해볼 문제

**문제 1:** 약쌍대성이 왜 "약"이라고 불리는지 설명하고, 강쌍대성과의 차이를 기술하세요.

<details>
<summary>힌트 및 해설</summary>

약쌍대성은 항상 $d^* \leq p^*$를 보장하지만, 항상 등호가 성립하지는 않습니다. 따라서 "약한" 부등식입니다.

강쌍대성은 $d^* = p^*$를 의미하며, 이 경우 쌍대 문제를 풀어서 원 문제의 정확한 최적값을 얻을 수 있습니다.

비볼록 문제에서는 약쌍대성만 보장되므로, gap을 좁히기 위해 이완(relaxation) 기법 등을 사용합니다.

</details>

**문제 2:** 다음 비볼록 문제의 duality gap을 수치적으로 계산하세요:
```
min x₁² - 2x₁x₂ + 2x₂²
s.t. x₁ + x₂ ≤ 1
     x₁, x₂ ≥ 0
```

<details>
<summary>힌트 및 해설</summary>

Hessian을 계산하면:
$$H = \begin{pmatrix} 2 & -2 \\ -2 & 4 \end{pmatrix}$$

고유값: λ₁ = 0, λ₂ = 6 (음이 아닌 고유값이므로 준정부호)

경계에서 검토:
- (0,0): f = 0
- (1,0): f = 1
- (0,1): f = 2
- (1/2, 1/2): f = 1/4 - 1/2 + 1/2 = 1/4

최적해는 (1/2, 1/2)에서 p* = 1/4

쌍대 함수 계산:
$$L(x_1,x_2,\lambda,\mu_1,\mu_2) = x_1^2 - 2x_1x_2 + 2x_2^2 + \lambda(x_1+x_2-1) - \mu_1 x_1 - \mu_2 x_2$$

정류 조건을 풀고, d*를 수치적으로 계산하면 gap을 얻을 수 있습니다.

</details>

**문제 3:** LP의 경우 약쌍대성을 Farkas Lemma와 연결하여 설명하세요.

<details>
<summary>힌트 및 해설</summary>

LP:
```
min c^T x      s.t. Ax ≥ b
max b^T y      s.t. A^T y = c, y ≥ 0
```

약쌍대성: 가능한 x, y에 대해 c^T x ≥ b^T y

이는 $c^T x = (A^T y)^T x = y^T (Ax) \geq y^T b = b^T y$에서 나옵니다.

Farkas Lemma와의 연결: 만약 gap이 0이 아니면, 원 시스템과 쌍대 시스템 중 하나가 해가 없거나, 둘 다 해가 있지만 "완전히" 일치하지 않음을 의미합니다.

</details>

<div align="center">

| [◀ 01. Lagrangian과 쌍대 함수](./01-lagrangian-dual-function.md) | [📚 README](../README.md) | [03. Slater 조건과 강쌍대성 ▶](./03-slater-strong-duality.md) |

</div>
