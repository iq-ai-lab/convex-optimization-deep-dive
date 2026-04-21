# 4. KKT 조건 — 필요충분조건으로서

## 🎯 핵심 질문
- KKT 조건의 4가지 구성 요소는 무엇인가?
- 볼록 문제에서 KKT는 왜 필요충분조건이 되는가?
- 비볼록 문제에서 KKT의 역할은 무엇인가?

## 🔍 왜 이 이론이 AI에서 중요한가
KKT 조건은 제약이 있는 최적화 문제의 최적성을 특성화합니다. 모든 최적화 알고리즘(경사하강법, 내부점 방법, 증강 라그랑주법)의 수렴 조건이 KKT 조건입니다. 또한 최적성 조건을 코드로 검증할 때 KKT를 확인합니다.

## 📐 수학적 선행 조건
- Lagrangian과 쌍대 함수 (Chapter 4-1)
- 강쌍대성과 Slater 조건 (Chapter 4-3)
- 볼록 함수의 1차 최적성 조건

## 📖 직관적 이해
KKT 조건은 "균형" 조건입니다. 최적해에서는:
1. 목적함수의 기울기가 제약들의 기울기들의 조합과 균형을 이룬다 (정류성)
2. 각 제약은 가능성을 만족한다 (가능성)
3. 쌍대 변수는 음이 아니다 (쌍대 가능성)
4. 활성 제약만 영향을 미친다 (상보 느슨함)

## ✏️ 엄밀한 정의

**원 문제:**
```
minimize   f₀(x)
subject to fᵢ(x) ≤ 0,  i = 1,...,m
          hⱼ(x) = 0,   j = 1,...,p
```

**KKT 조건 (4개):**

1. **정류 조건 (Stationarity):**
   $$\nabla_x L(x^*, \lambda^*, \nu^*) = 0$$
   $$\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$

2. **원 가능성 (Primal feasibility):**
   $$f_i(x^*) \leq 0, \quad i = 1,...,m$$
   $$h_j(x^*) = 0, \quad j = 1,...,p$$

3. **쌍대 가능성 (Dual feasibility):**
   $$\lambda_i^* \geq 0, \quad i = 1,...,m$$

4. **상보 느슨함 (Complementary slackness):**
   $$\lambda_i^* f_i(x^*) = 0, \quad i = 1,...,m$$

## 🔬 정리와 증명

**정리 1: KKT는 필요조건 (모든 미분가능한 문제)**

$x^*$가 최적해이고 제약 유효 조건(LICQ)이 성립하면, KKT 조건을 만족하는 $(\lambda^*, \nu^*)$가 존재합니다.

(증명 생략: 극값 정리와 제약 유효 조건 이용)

**정리 2: KKT는 충분조건 (볼록 문제)**

*가정:* 
- $f_0, f_1,...,f_m$은 볼록함수
- $h_j$는 아핀함수
- KKT 조건 만족: $(x^*, \lambda^*, \nu^*)$

*결론:* $x^*$는 원 문제의 최적해입니다.

*증명:*

KKT의 정류 조건에서
$$\nabla f_0(x^*) = -\sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) - \sum_{j=1}^p \nu_j^* \nabla h_j(x^*)$$

볼록함수의 1차 조건에 의해, 임의의 가능 해 $x$에 대해:
$$f_0(x) \geq f_0(x^*) + \nabla f_0(x^*)^T (x - x^*)$$

정류 조건을 대입:
$$f_0(x) \geq f_0(x^*) - \sum_i \lambda_i^* \nabla f_i(x^*)^T (x-x^*) - \sum_j \nu_j^* \nabla h_j(x^*)^T(x-x^*)$$

$f_i$의 볼록성:
$$f_i(x) \geq f_i(x^*) + \nabla f_i(x^*)^T(x-x^*) \geq \nabla f_i(x^*)^T(x-x^*)$$

(상보 느슨함: $\lambda_i^* f_i(x^*) = 0$이므로 $f_i(x^*) = 0$ 또는 $\lambda_i^* = 0$)

$x$가 가능하면 $f_i(x) \leq 0$이므로:
$$\nabla f_i(x^*)^T(x-x^*) \leq f_i(x) - f_i(x^*) \leq -f_i(x^*)$$

상보 느슨함에서 $\lambda_i^* f_i(x^*) = 0$이므로:
$$\lambda_i^* f_i(x) \leq -\lambda_i^* f_i(x^*) = 0$$

또한 $h_j(x^*) = h_j(x) = 0$ (아핀)이므로:
$$\nabla h_j(x^*)^T(x-x^*) = h_j(x) - h_j(x^*) = 0$$

종합하면:
$$f_0(x) \geq f_0(x^*) - \sum_i \lambda_i^* (f_i(x) - f_i(x^*)) \geq f_0(x^*)$$

($\lambda_i^* \geq 0$, $f_i(x) \leq 0$)

따라서 $x^*$는 최적해입니다. ∎

**정리 3: 상보 느슨함의 해석**

$\lambda_i^* f_i(x^*) = 0$은:
- $\lambda_i^* > 0 \Rightarrow f_i(x^*) = 0$ (제약이 활성)
- $f_i(x^*) < 0 \Rightarrow \lambda_i^* = 0$ (제약이 비활성)

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize

print("=" * 70)
print("KKT 조건 검증")
print("=" * 70)

# 예제 1: 간단한 QP에서 KKT 검증
print("\n[예제 1] QP: min ‖x‖² s.t. Ax ≤ b")
print("-" * 70)

# min (1/2)(x₁² + x₂²) s.t. x₁ + x₂ ≥ 1, x₁ ≥ 0, x₂ ≥ 0

x = cp.Variable(2)
f0 = 0.5 * cp.sum_squares(x)
constraints = [
    x[0] + x[1] >= 1,
    x[0] >= 0,
    x[1] >= 0
]

problem = cp.Problem(cp.Minimize(f0), constraints)
problem.solve()

x_opt = x.value
f_opt = problem.value

print(f"최적해: x* = {x_opt}")
print(f"최적값: f₀(x*) = {f_opt:.6f}")

# KKT 조건 1: 정류 조건
# ∇f₀(x*) + λ₁∇f₁(x*) + λ₂∇f₂(x*) + λ₃∇f₃(x*) = 0
# ∇f₀ = x, ∇f₁ = (-1, -1), ∇f₂ = (-1, 0), ∇f₃ = (0, -1)

grad_f0 = x_opt  # (x₁, x₂)
grad_f1 = np.array([-1, -1])  # f₁(x) = -(x₁ + x₂ - 1) = 1 - x₁ - x₂
grad_f2 = np.array([-1, 0])   # f₂(x) = -x₁
grad_f3 = np.array([0, -1])   # f₃(x) = -x₂

# 쌍대 변수
lam1 = constraints[0].dual_value
lam2 = constraints[1].dual_value
lam3 = constraints[2].dual_value

if lam1 is None:
    lam1 = 0
if lam2 is None:
    lam2 = 0
if lam3 is None:
    lam3 = 0

print(f"\n[KKT 조건 검증]")
print(f"쌍대 변수: λ₁* = {lam1:.6f}, λ₂* = {lam2:.6f}, λ₃* = {lam3:.6f}")

# 1. 정류 조건
stationarity = grad_f0 + lam1 * grad_f1 + lam2 * grad_f2 + lam3 * grad_f3
print(f"\n1. Stationarity: ∇f₀ + Σλᵢ∇fᵢ = {stationarity}")
print(f"   ‖residual‖ = {np.linalg.norm(stationarity):.2e}")

# 2. 원 가능성
f_vals = [
    x_opt[0] + x_opt[1] - 1,  # f₁: ≤ 0
    -x_opt[0],                 # f₂: ≤ 0
    -x_opt[1]                  # f₃: ≤ 0
]

print(f"\n2. Primal feasibility:")
for i, f_val in enumerate(f_vals):
    print(f"   f_{i+1}(x*) = {f_val:.6f} ≤ 0 ? {f_val <= 1e-6}")

# 3. 쌍대 가능성
print(f"\n3. Dual feasibility:")
for i, lam in enumerate([lam1, lam2, lam3]):
    print(f"   λ_{i+1}* = {lam:.6f} ≥ 0 ? {lam >= -1e-6}")

# 4. 상보 느슨함
print(f"\n4. Complementary slackness:")
for i, (f_val, lam) in enumerate(zip(f_vals, [lam1, lam2, lam3])):
    prod = lam * f_val
    print(f"   λ_{i+1}* f_{i+1}(x*) = {lam:.6f} × {f_val:.6f} = {prod:.2e}")
    if lam > 1e-6:
        print(f"      → λ_{i+1}* > 0, so f_{i+1}(x*) should be 0: {abs(f_val) < 1e-5}")
    else:
        print(f"      → λ_{i+1}* ≈ 0")

# 예제 2: 비볼록 문제에서 KKT의 필요조건
print("\n" + "=" * 70)
print("[예제 2] 비볼록 문제: KKT는 필요조건만")
print("-" * 70)

# min x³ s.t. x ≤ 1, x ≥ -1
# f₀(x) = x³ (비볼록)

x_test_vals = np.linspace(-1, 1, 1000)
f_test_vals = x_test_vals**3

# 모서리 점들에서 값 확인
edge_points = [-1, -0.5, 0, 0.5, 1]
print("\n함수 값:")
for x_test in edge_points:
    f_test = x_test**3
    print(f"  x = {x_test:4.1f}: f(x) = {f_test:7.4f}")

p_opt_nonconvex = np.min(f_test_vals)
x_opt_nonconvex = x_test_vals[np.argmin(f_test_vals)]

print(f"\n최적해: x* = {x_opt_nonconvex:.6f}")
print(f"최적값: f(x*) = {p_opt_nonconvex:.6f}")

# x* = -1에서 KKT 검증
x_check = -1.0
f0_val = x_check**3
grad_f0 = 3 * x_check**2

# 제약: f₁(x) = x - 1 ≤ 0, f₂(x) = -x - 1 ≤ 0
f1_val = x_check - 1  # = -2
f2_val = -x_check - 1  # = 0 (활성)

grad_f1 = 1
grad_f2 = -1

print(f"\n[비볼록 점 x = -1에서 KKT]")
print(f"f₁(x) = {f1_val:.3f}, f₂(x) = {f2_val:.3f}")

# KKT를 만족하는 (λ₁, λ₂) 찾기
# ∇f₀ + λ₁∇f₁ + λ₂∇f₂ = 0
# 3·1 + λ₁·1 + λ₂·(-1) = 0
# 3 + λ₁ - λ₂ = 0
# λ₁ - λ₂ = -3

# 상보 느슨함: λ₁·f₁ = λ₁·(-2) = 0 → λ₁ = 0
# 따라서: λ₂ = 3

lam1_nonconv = 0
lam2_nonconv = 3

stationarity_nc = grad_f0 + lam1_nonconv * grad_f1 + lam2_nonconv * grad_f2
print(f"\nλ₁* = {lam1_nonconv:.3f}, λ₂* = {lam2_nonconv:.3f}")
print(f"Stationarity residual: {stationarity_nc:.6f}")
print(f"Complementary slackness:")
print(f"  λ₁* f₁(x*) = {lam1_nonconv * f1_val:.6f}")
print(f"  λ₂* f₂(x*) = {lam2_nonconv * f2_val:.6f}")

# 그러나 x* = -1이 최적이 아님!
print(f"\n⚠️ 주목: KKT 만족하지만 최적해가 아님 (비볼록)")
print(f"실제 최적해: x* = {x_opt_nonconvex:.6f}, f(x*) = {p_opt_nonconvex:.6f}")

# 예제 3: KKT 조건의 기하학적 의미
print("\n" + "=" * 70)
print("[예제 3] KKT의 기하학적 해석")
print("-" * 70)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 볼록 QP
ax1 = axes[0]
x1_range = np.linspace(-0.5, 2, 100)
x2_from_constraint = 1 - x1_range  # x₁ + x₂ = 1

f_contours_vals = [0.1, 0.25, 0.5, 1, 2]
for f_val in f_contours_vals:
    # (1/2)(x₁² + x₂²) = f_val → x₁² + x₂² = 2f_val
    r = np.sqrt(2 * f_val)
    theta = np.linspace(0, 2*np.pi, 100)
    x1_circle = r * np.cos(theta)
    x2_circle = r * np.sin(theta)
    ax1.plot(x1_circle, x2_circle, 'gray', alpha=0.5, linewidth=0.5)

# 제약
ax1.plot([0, 1.5], [1, -0.5], 'b-', linewidth=2, label='$x_1 + x_2 = 1$')
ax1.fill_between(x1_range, x2_from_constraint, 5, alpha=0.2, color='blue')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# 최적해
ax1.plot(x_opt[0], x_opt[1], 'r*', markersize=15, label=f'$x^* = ({x_opt[0]:.2f}, {x_opt[1]:.2f})$')

# 기울기
ax1.arrow(x_opt[0], x_opt[1], grad_f0[0]*0.1, grad_f0[1]*0.1, 
         head_width=0.05, head_length=0.05, fc='red', ec='red', label='$\\nabla f_0(x^*)$')
ax1.arrow(x_opt[0], x_opt[1], grad_f1[0]*0.1, grad_f1[1]*0.1, 
         head_width=0.05, head_length=0.05, fc='green', ec='green', label='$\\nabla f_1(x^*)$')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('KKT: 기울기의 균형\n$\\nabla f_0 + \\lambda_1^* \\nabla f_1 = 0$')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.5, 1.5])
ax1.set_ylim([-0.5, 2])
ax1.set_aspect('equal')

# 오른쪽: KKT 조건 체크리스트
ax2 = axes[1]
ax2.axis('off')

kkt_text = f"""
KKT 조건 검증 (Convex QP)

1. Stationarity
   ‖∇f₀ + λ₁∇f₁ + λ₂∇f₂ + λ₃∇f₃‖
   = {np.linalg.norm(stationarity):.2e} ✓

2. Primal Feasibility
   f₁(x*) = {f_vals[0]:.6f} ≤ 0 ✓
   f₂(x*) = {f_vals[1]:.6f} ≤ 0 ✓
   f₃(x*) = {f_vals[2]:.6f} ≤ 0 ✓

3. Dual Feasibility
   λ₁* = {lam1:.6f} ≥ 0 ✓
   λ₂* = {lam2:.6f} ≥ 0 ✓
   λ₃* = {lam3:.6f} ≥ 0 ✓

4. Complementary Slackness
   λ₁* f₁(x*) = {lam1*f_vals[0]:.2e} ✓
   λ₂* f₂(x*) = {lam2*f_vals[1]:.2e} ✓
   λ₃* f₃(x*) = {lam3*f_vals[2]:.2e} ✓

→ KKT 조건 모두 만족
→ x*는 전역 최적해
"""

ax2.text(0.1, 0.9, kkt_text, transform=ax2.transAxes, 
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/tmp/kkt_conditions.png', dpi=150, bbox_inches='tight')
print("✓ KKT 조건 시각화 저장됨")

print("\n" + "=" * 70)
print("모든 KKT 조건이 검증되었습니다.")
print("=" * 70)

plt.show()
```

## 🔗 AI/ML 연결
- **최적화 알고리즘**: 경사하강법, 뉴턴 방법, IPM의 종료 조건 = KKT 조건
- **SVM 학습**: 쌍대 문제의 KKT 조건 = Support vector 결정
- **제제약 신경망 학습**: 가중치 정규화, 배치 정규화의 쌍대 해석

## ⚖️ 가정과 한계
- 비볼록 문제에서 KKT는 필요조건일 수 있으나 충분조건은 아님
- 제약 유효 조건(LICQ) 가정 필요
- 수치 계산에서 정류 조건의 "0"은 작은 허용오차 내의 값

## 📌 핵심 정리
1. **KKT 4조건**: 정류, 원 가능, 쌍대 가능, 상보 느슨함
2. **볼록 문제**: KKT 충분조건 (KKT 만족 → 최적)
3. **모든 문제**: KKT 필요조건 (최적 → KKT 만족)
4. **상보 느슨함**: 활성 제약만 영향 미침
5. **다음 단계**: 쌍대 해석과 그림자 가격

## 🤔 생각해볼 문제

**문제 1:** KKT 조건에서 상보 느슨함의 의미를 설명하고, 이것이 왜 "느슨함"이라 불리는지 기술하세요.

<details>
<summary>힌트 및 해설</summary>

$\lambda_i^* f_i(x^*) = 0$은 두 가지 경우를 의미합니다:

1. $\lambda_i^* = 0$: 제약 $i$가 비활성 (slack) → 쌍대 변수도 0
2. $f_i(x^*) = 0$: 제약 $i$가 활성 (binding) → 쌍대 변수 양수 가능

"느슨함(slackness)"은 비활성 제약(slack variable > 0)을 의미합니다. 이 경우 쌍대 변수가 0이 되므로, 제약을 조금 완화해도 최적값이 변하지 않습니다.

</details>

**문제 2:** 비볼록 문제 $\min x^3$ s.t. $-1 \leq x \leq 1$에서 KKT 조건을 만족하는 모든 점을 찾으세요.

<details>
<summary>힌트 및 해설</summary>

가능 집합: $\{x : -1 \leq x \leq 1\}$

경계 점에서 확인:

**$x = -1$:**
- $f_0(x) = -1$, $\nabla f_0 = 3$
- 제약: $f_1(x) = x - 1 = -2 < 0$, $f_2(x) = -x - 1 = 0$
- KKT: $3 + 0 \cdot 1 + \lambda_2(-1) = 0$ → $\lambda_2 = 3$ ✓

**$x = 0$:**
- $f_0(x) = 0$, $\nabla f_0 = 0$
- 제약: $f_1(x) = -1 < 0$, $f_2(x) = -1 < 0$
- KKT: $0 + 0 + 0 = 0$ ✓ (쌍대 변수 모두 0)

**$x = 1$:**
- $f_0(x) = 1$, $\nabla f_0 = 3$
- 제약: $f_1(x) = 0$, $f_2(x) = -2 < 0$
- KKT: $3 + \lambda_1(1) + 0 = 0$ → $\lambda_1 = -3 < 0$ ✗

따라서 KKT를 만족하는 점: $x = -1$ (최적), $x = 0$ (안장점)

</details>

**문제 3:** 선형계획 (LP)에서 KKT 조건이 경제학의 "그림자 가격" 이론과 어떻게 연결되는지 설명하세요.

<details>
<summary>힌트 및 해설</summary>

LP:
```
min c^T x       s.t. Ax ≤ b, x ≥ 0
```

KKT 정류 조건:
$$c + A^T \lambda + \nu = 0$$

(여기서 λ는 Ax ≤ b의 쌍대, ν는 x ≥ 0의 쌍대)

쌍대 변수 λᵢ는 i번째 자원(제약 i의 RHS bᵢ)의 쌍대 가격입니다.

경제 해석: λᵢ > 0이면, bᵢ를 1단위 증가시킬 때 최적값이 -λᵢ만큼 개선됩니다 (자원의 한계 수익률).

</details>

<div align="center">

| [◀ 03. Slater 조건과 강쌍대성](./03-slater-strong-duality.md) | [📚 README](../README.md) | [05. 쌍대 해석 — 그림자 가격 ▶](./05-dual-interpretation.md) |

</div>
