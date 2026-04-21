# 3. Slater 조건과 강쌍대성

## 🎯 핵심 질문
- 강쌍대성이 성립하는 조건은 무엇인가?
- Slater 조건이란 정확히 무엇이며, 왜 필요한가?
- 분리 초평면 정리를 이용한 강쌍대성 증명의 핵심은 무엇인가?

## 🔍 왜 이 이론이 AI에서 중요한가
강쌍대성은 최적화 문제를 푸는 알고리즘의 correctness를 보증합니다. 예를 들어, SVM에서 쌍대 문제를 풀어서 얻은 α가 원 문제(w, b)의 최적해를 유일하게 결정하는 이유가 강쌍대성입니다. 또한 내부점 방법(interior point methods)의 이론적 기초도 강쌍대성입니다.

## 📐 수학적 선행 조건
- 볼록 함수와 볼록 집합
- Lagrangian과 쌍대 함수 (Chapter 4-1)
- 약쌍대성 (Chapter 4-2)
- 분리 초평면 정리 (Separating Hyperplane Theorem)

## 📖 직관적 이해
원 문제가 "충분히 안정적"이라면 (Slater 조건), 쌍대 문제의 최적값이 원 문제의 최적값과 정확히 같습니다. Slater 조건은 "제약들이 너무 답답하게 모여있지 않다"는 의미로, 엄격히 가능한 내부점이 존재하면 됩니다.

## ✏️ 엄밀한 정의

**원 문제:**
```
minimize   f₀(x)
subject to fᵢ(x) ≤ 0,  i = 1,...,m
          hⱼ(x) = 0,   j = 1,...,p
```

**Slater 조건 (제약 유효 조건, CQ):**

존재한다 $\tilde{x} \in \text{relint(dom)}$ such that
$$f_i(\tilde{x}) < 0 \quad \forall i=1,...,m$$
$$h_j(\tilde{x}) = 0 \quad \forall j=1,...,p$$

즉, 부등식 제약을 **엄격히** 만족하는 등식 가능 점이 존재합니다.

**특수 경우: 아핀 부등식**

$f_i(x) = a_i^T x - b_i$ (선형)일 때:
$$f_i(\tilde{x}) \leq 0 \quad (\text{비엄격도 가능})$$
로 완화 가능합니다.

**강쌍대성 정리:**

원 문제가 볼록이고, Slater 조건이 성립하면
$$d^* = p^*$$

## 🔬 정리와 증명

**정리 1: 강쌍대성 (분리 초평면을 이용한 증명)**

*가정:* 
- $f_0, f_1,...,f_m$은 볼록함수
- $h_1,...,h_p$는 아핀함수
- Slater 조건: ∃$\tilde{x}$: $f_i(\tilde{x}) < 0, h_j(\tilde{x}) = 0$

*증명:*

**Step 1: 확장 집합 정의**

$$A = \{(u,v,t) \mid \exists x: f_i(x) \leq u_i \text{ for } i=1,...,m,$$
$$h_j(x) = v_j \text{ for } j=1,...,p, f_0(x) \leq t\}$$

$A$는 에피그래프 개념을 다변수로 확장한 것입니다. 볼록함수의 정의에 의해 **$A$는 볼록 집합**입니다.

**Step 2: 쌍대 집합 정의**

$$B = \{(u,v,t) \mid u \leq 0, v = 0, t < p^*\}$$

$B$도 볼록 집합입니다. (절반공간의 교집합)

**Step 3: 교집합 확인**

Slater 조건에서 $\tilde{x}$가 존재하여 $f_i(\tilde{x}) < 0, h_j(\tilde{x}) = 0$이므로,
$(u,v,t) = (0,0,f_0(\tilde{x}))$에 대해:
- $u \leq 0$, $v = 0$, $f_0(\tilde{x}) < p^*$ (약쌍대성)이면 $(u,v,t) \in B$
- $(u,v,t) = (0,0,f_0(\tilde{x})) \in A$

하지만 $f_0(\tilde{x}) \geq d^*$이고, 일반적으로 $(f_0(\tilde{x}) < p^*, 0, 0) \not\in A$인 점들이 $B$에 있습니다.

실제로 $A \cap B = \emptyset$입니다. (귀류법: 만약 $(u_0,v_0,t_0) \in A \cap B$이면, $u_0 \leq 0, v_0 = 0, t_0 < p^*$인데, $(u_0,v_0,t_0) \in A$는 $t_0 \geq p^*$를 의미하여 모순)

**Step 4: 분리 초평면 정리 적용**

두 볼록집합 $A$와 $B$가 서로 분리되므로, 분리 초평면이 존재합니다:
$$\lambda^T u + \nu^T v + \mu t \geq 0 \quad \text{for all } (u,v,t) \in A$$
$$\lambda^T u + \nu^T v + \mu t \leq 0 \quad \text{for all } (u,v,t) \in B$$

에서 $\mu > 0$을 보일 수 있습니다. (정규화)

**Step 5: 정규화 및 쌍대 변수**

$\mu > 0$이므로 $\mu = 1$로 정규화하면:
$$\lambda^T u + \nu^T v + t \geq 0 \quad \text{for all } (u,v,t) \in A$$

여기서 $(u,v,t) = (f(x), h(x), f_0(x)) \in A$를 대입하면:
$$\lambda^T f(x) + \nu^T h(x) + f_0(x) \geq 0$$
$$L(x,\lambda,\nu) \geq 0$$

따라서 $g(\lambda,\nu) = \inf_x L(x,\lambda,\nu) \geq 0$ (무한이 아닌 경우)

$\lambda \geq 0$도 보일 수 있으므로, $d^* = \sup_{\lambda \geq 0, \nu} g(\lambda,\nu) \geq 0 = f_0(x^*)$

약쌍대성과 합치면 $d^* = p^*$ ∎

**정리 2: LP에서의 강쌍대성**

LP는 항상 Slater 조건을 만족합니다 (존재한다면). 따라서 LP의 원 문제와 쌍대 문제의 최적값은 같습니다.

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import linprog

print("=" * 70)
print("강쌍대성 검증: Slater 조건의 역할")
print("=" * 70)

# 예제 1: Slater 조건 만족하는 QP (강쌍대성)
print("\n[예제 1] Slater 조건 만족 (강쌍대성 성립)")
print("-" * 70)

x = cp.Variable(2)
f0 = cp.sum_squares(x)
constraints = [
    x[0] + x[1] <= 0.5,  # f₁(x) = x₁ + x₂ - 0.5 ≤ 0
    x[0] - x[1] <= 0.3,  # f₂(x) = x₁ - x₂ - 0.3 ≤ 0
]

problem = cp.Problem(cp.Minimize(f0), constraints)
problem.solve()

p_opt = problem.value
x_opt = x.value

print(f"원 문제 최적값: p* = {p_opt:.6f}")
print(f"최적해: x* = {x_opt}")

# Slater 점 확인: x̃ = (-1, -1)
x_slater = np.array([-1.0, -1.0])
f1_slater = x_slater[0] + x_slater[1] - 0.5
f2_slater = x_slater[0] - x_slater[1] - 0.3

print(f"\nSlater 점 검증: x̃ = {x_slater}")
print(f"  f₁(x̃) = {f1_slater:.3f} < 0 ? {f1_slater < 0}")
print(f"  f₂(x̃) = {f2_slater:.3f} < 0 ? {f2_slater < 0}")
print(f"  → Slater 조건 만족: {f1_slater < 0 and f2_slater < 0}")

# 쌍대 변수 추출
lambda1 = constraints[0].dual_value
lambda2 = constraints[1].dual_value
print(f"\n최적 쌍대 변수: λ₁* = {lambda1:.6f}, λ₂* = {lambda2:.6f}")

# 쌍대 함수 값 계산 (수학적으로)
# L(x,λ) = ‖x‖² + λ₁(x₁ + x₂ - 0.5) + λ₂(x₁ - x₂ - 0.3)
# ∂L/∂x₁ = 2x₁ + λ₁ + λ₂ = 0 → x₁ = -(λ₁ + λ₂)/2
# ∂L/∂x₂ = 2x₂ + λ₁ - λ₂ = 0 → x₂ = -(λ₁ - λ₂)/2

if lambda1 is not None and lambda2 is not None:
    x1_dual = -(lambda1 + lambda2) / 2
    x2_dual = -(lambda1 - lambda2) / 2
    g_dual = x1_dual**2 + x2_dual**2 - lambda1 * 0.5 - lambda2 * 0.3
    print(f"쌍대 함수 값: g(λ*) = {g_dual:.6f}")
    print(f"Duality gap: p* - d* = {p_opt - g_dual:.2e}")

# 예제 2: Slater 조건 위반 (경계 사례)
print("\n" + "=" * 70)
print("[예제 2] Slater 조건 위반 (경계 사례)")
print("-" * 70)

# min x² s.t. x ≤ 0 (Slater 만족)
# vs
# min (x-1)² s.t. x ≤ 0, x ≥ 0 (x = 0만 가능, Slater 위반)

x2 = cp.Variable()
f0_2 = (x2 - 1)**2
constraints_2 = [x2 <= 0, x2 >= 0]

problem_2 = cp.Problem(cp.Minimize(f0_2), constraints_2)
problem_2.solve()

p_opt_2 = problem_2.value
x_opt_2 = x2.value

print(f"제약 조건: x ≤ 0 AND x ≥ 0")
print(f"원 문제 최적값: p* = {p_opt_2:.6f}")
print(f"최적해: x* = {x_opt_2}")

# Slater 점 존재 여부: x ≤ 0, x ≥ 0을 엄격히 만족하는 점은 없음
print(f"Slater 조건: 부등식을 엄격히 만족하는 점이 없음 (경계)")
print(f"→ Slater 조건 위반")

# 쌍대 변수
lambda_ub = constraints_2[0].dual_value  # x ≤ 0
lambda_lb = constraints_2[1].dual_value  # x ≥ 0

print(f"쌍대 변수: λ₁(≤0) = {lambda_ub:.6f}, λ₂(≥0) = {lambda_lb:.6f}")

# 예제 3: LP에서의 강쌍대성
print("\n" + "=" * 70)
print("[예제 3] LP: 항상 강쌍대성 (Slater 만족 가정)")
print("-" * 70)

# min c^T x s.t. Ax ≤ b
# LP의 쌍대: max b^T y s.t. A^T y = c, y ≥ 0

c = np.array([1, 2, 1])
A = np.array([
    [1, 1, 1],
    [2, 1, 0],
])
b = np.array([4, 3])

# CVXPY로 풀기
x_lp = cp.Variable(3)
problem_lp_primal = cp.Problem(
    cp.Minimize(c @ x_lp),
    [A @ x_lp <= b, x_lp >= 0]
)
problem_lp_primal.solve()

p_lp = problem_lp_primal.value
x_lp_opt = x_lp.value

print(f"Primal LP:")
print(f"  min c^T x = {p_lp:.6f}")
print(f"  x* = {x_lp_opt}")

# Dual LP: max b^T y s.t. A^T y ≤ c, y ≥ 0
y = cp.Variable(2)
problem_lp_dual = cp.Problem(
    cp.Maximize(b @ y),
    [A.T @ y <= c, y >= 0]
)
problem_lp_dual.solve()

d_lp = problem_lp_dual.value
y_opt = y.value

print(f"\nDual LP:")
print(f"  max b^T y = {d_lp:.6f}")
print(f"  y* = {y_opt}")

print(f"\n강쌍대성 검증:")
print(f"  p* = {p_lp:.6f}")
print(f"  d* = {d_lp:.6f}")
print(f"  p* - d* = {abs(p_lp - d_lp):.2e}")
print(f"  강쌍대성 성립 ? {abs(p_lp - d_lp) < 1e-5}")

# 예제 4: 가시화 - Feasible region과 Slater 점
print("\n" + "=" * 70)
print("[예제 4] 기하학적 해석")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽: Slater 조건 만족
ax1 = axes[0]
x1_range = np.linspace(-2, 1, 100)

# 제약: x₁ + x₂ ≤ 0.5, x₁ - x₂ ≤ 0.3
x2_constraint1 = 0.5 - x1_range  # x₂ ≤ 0.5 - x₁
x2_constraint2 = x1_range - 0.3  # x₂ ≥ x₁ - 0.3

ax1.fill_between(x1_range, x2_constraint2, x2_constraint1, 
                 where=(x2_constraint2 <= x2_constraint1),
                 alpha=0.3, color='blue', label='Feasible region')
ax1.plot(x1_range, x2_constraint1, 'b-', label='$x_1 + x_2 = 0.5$')
ax1.plot(x1_range, x2_constraint2, 'b-', label='$x_1 - x_2 = 0.3$')

# Slater 점
ax1.plot(x_slater[0], x_slater[1], 'ro', markersize=10, label=f'Slater point: $\\tilde{{x}}={x_slater}$')

# 최적해
ax1.plot(x_opt[0], x_opt[1], 'g*', markersize=15, label=f'Optimal: $x^*={np.round(x_opt, 2)}$')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Slater 조건 만족: 내부점 존재')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-2, 1])
ax1.set_ylim([-2, 2])

# 오른쪽: 강쌍대성 비교
ax2 = axes[1]
problems = ['QP\n(Slater만족)', 'LP\n(Slater만족)', 'Boundary\ncase']
p_values = [p_opt, p_lp, p_opt_2]
d_values = [p_opt, d_lp, p_opt_2]  # 강쌍대성 만족 가정
gaps = [abs(p - d) for p, d in zip(p_values, d_values)]

x_pos = np.arange(len(problems))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, p_values, width, label="p* (Primal)", alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, d_values, width, label="d* (Dual)", alpha=0.8)

ax2.set_ylabel('최적값')
ax2.set_title('강쌍대성: $p^* = d^*$')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(problems)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/slater_strong_duality.png', dpi=150, bbox_inches='tight')
print("✓ 기하학적 해석 그래프 저장됨")

plt.show()
```

## 🔗 AI/ML 연결
- **SVM의 기초**: 선형 분류기는 Slater 조건을 만족하므로, 쌍대 문제(커널 최적화)의 최적값 = 원 문제의 최적값
- **내부점 방법(IPM)**: 각 반복에서 Slater 조건의 약화 버전을 유지하며 최적해로 접근
- **분산 최적화**: 각 노드에서 부분 문제가 Slater 조건을 만족하면, 전체 시스템의 수렴성 보증

## ⚖️ 가정과 한계
- Slater 조건은 **충분조건**이지만 필요조건은 아님 (약한 제약 유효 조건도 충분)
- 부등식 제약이 선형이면 비엄격 조건 가능
- 실제 문제에서 Slater 점을 명시적으로 찾기는 어려울 수 있음

## 📌 핵심 정리
1. **Slater 조건**: 부등식을 엄격히 만족하는 내부점 존재
2. **강쌍대성 정리**: 볼록 + Slater 조건 → $d^* = p^*$
3. **분리 초평면**: 강쌍대성 증명의 핵심 기하학적 도구
4. **LP의 특수성**: 항상 강쌍대성 성립 (안정성 가정 하)
5. **다음 단계**: KKT 조건으로 최적성 특성화

## 🤔 생각해볼 문제

**문제 1:** Slater 조건을 만족하지 않는 볼록 문제의 예시를 들고, duality gap이 0이 아님을 보이세요.

<details>
<summary>힌트 및 해설</summary>

예: min x₁² s.t. x₁ ≥ 0, x₂² ≤ 0

$x_2^2 \leq 0$은 $x_2 = 0$을 강제하므로, 가능 집합은 $\{x : x_1 \geq 0, x_2 = 0\}$입니다.

이 집합의 상대적 내부(relative interior)에서 $x_1 > 0$을 엄격히 만족하는 점이 없습니다 (경계에만 있음).

따라서 Slater 조건 위반, 그리고 일반적으로 duality gap이 발생할 수 있습니다.

</details>

**문제 2:** 다음 QP에서 Slater 조건을 확인하고, 강쌍대성 여부를 판정하세요:
```
min (1/2)x^T Q x
s.t. Ax ≤ b
     x ≥ 0
```
여기서 Q는 양정부호, A는 m×n 행렬, b ∈ ℝᵐ입니다.

<details>
<summary>힌트 및 해설</summary>

Slater 점을 찾기: x̃ = 0이 가능한지 확인
- f₀(0) = 0 ✓
- A·0 = 0 ≤ b ? (b의 모든 성분 ≥ 0이면 가능)
- 0 ≥ 0 ✓

만약 b > 0 (모든 성분)이면, x̃ = 0이 엄격히 가능합니다.

따라서 Slater 조건 만족 → 강쌍대성 성립 ($d^* = p^*$)

</details>

**문제 3:** 분리 초평면 정리가 강쌍대성 증명에서 어떤 역할을 하는지 설명하세요.

<details>
<summary>힌트 및 해설</summary>

확장 집합 $A$와 집합 $B$가 서로 분리된다는 것은, 원 문제의 가능 집합과 이상적 목적함수 값 범위가 "충분히 떨어져 있다"는 의미입니다.

분리 초평면 $(\\lambda, \\nu, \\mu)$는 정확히 최적 쌍대 변수이며, 이 초평면이 두 집합을 분리한다는 것은:

- 한쪽(A)에선 $L(x,\lambda,\nu) \geq 0$ (정규화 후)
- 다른 쪽(B)에선 $-\mu t \leq 0$, 즉 $t \geq d^*$

이를 통해 $d^* = p^*$를 도출합니다.

</details>

<div align="center">

| [◀ 02. 약쌍대성(Weak Duality)](./02-weak-duality.md) | [📚 README](../README.md) | [04. KKT 조건 ▶](./04-kkt-conditions.md) |

</div>
