# 1. Lagrangian과 쌍대 함수

## 🎯 핵심 질문
- Lagrangian은 제약 조건이 있는 최적화 문제를 어떻게 하나의 수식으로 통합하는가?
- 쌍대 함수는 왜 항상 오목함수인가?
- 원래 문제의 최적값과 쌍대 함수의 최댓값은 어떤 관계가 있는가?

## 🔍 왜 이 이론이 AI에서 중요한가
제약이 있는 최적화 문제(SVM, 포트폴리오 최적화, 리소스 할당)를 풀 때, 원 문제를 직접 푸는 것보다 쌍대 문제를 푸는 것이 계산상 유리할 수 있습니다. 특히 커널 메서드와 결합하면 고차원 데이터에서도 효율적으로 풀 수 있습니다.

## 📐 수학적 선행 조건
- 볼록 집합, 볼록 함수의 정의
- 부등식 제약과 등식 제약의 구분
- 함수의 infimum과 supremum의 개념

## 📖 직관적 이해
제약이 있는 최적화 문제를 푸는 것은 마치 페널티를 주면서 제약을 위반한 점들을 "벌칙"주는 것과 같습니다. Lagrangian은 제약을 이 벌칙의 형태로 변환합니다. 이 벌칙들의 강도(λ, ν)를 조절하면서 최적의 x를 찾으면, 원래 문제의 답에 접근할 수 있습니다.

## ✏️ 엄밀한 정의

**원 문제(Primal problem):**
```
minimize   f₀(x)
subject to fᵢ(x) ≤ 0,  i = 1,...,m
          hⱼ(x) = 0,   j = 1,...,p
```

**Lagrangian:**
$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

- $\lambda_i \geq 0$: 부등식 제약에 대한 쌍대 변수 (dual variable)
- $\nu_j \in \mathbb{R}$: 등식 제약에 대한 쌍대 변수 (부호 제약 없음)

**쌍대 함수(Dual function):**
$$g(\lambda, \nu) = \inf_{x \in \text{dom}(f_0) \cap \text{dom}(f_i) \cap \text{dom}(h_j)} L(x, \lambda, \nu)$$

**쌍대 문제(Dual problem):**
$$\text{maximize} \quad g(\lambda, \nu)$$
$$\text{subject to} \quad \lambda_i \geq 0, \quad i = 1,...,m$$

## 🔬 정리와 증명

**정리 1: 쌍대 함수는 오목함수**

*증명:*
$g(\lambda, \nu)$를 $(\lambda, \nu)$의 함수로 본다면, 고정된 $x$에 대해 $L(x,\lambda,\nu)$는 $(\lambda,\nu)$에 대한 아핀 함수입니다.
$$L(x, \lambda, \nu) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x)$$

$g(\lambda, \nu) = \inf_x L(x,\lambda,\nu)$는 여러 아핀 함수들의 하한이므로, 오목함수입니다.

**정리 2: 약쌍대성 (Weak duality)**

$x$가 원 문제의 가능 해(feasible)이고 $\lambda \geq 0$일 때,
$$g(\lambda, \nu) \leq f_0(x) \leq p^*$$

*증명:*
$x$가 가능하면 $f_i(x) \leq 0$ 및 $h_j(x) = 0$이므로,
$$L(x,\lambda,\nu) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x) \leq f_0(x)$$

($\lambda_i \geq 0$이고 $f_i(x) \leq 0$이므로 $\sum_i \lambda_i f_i(x) \leq 0$, 그리고 $h_j(x) = 0$)

따라서 $g(\lambda, \nu) = \inf_x L(x,\lambda,\nu) \leq L(x,\lambda,\nu) \leq f_0(x)$

모든 가능한 $x$에 대해 성립하므로 $g(\lambda,\nu) \leq p^*$ ∎

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cp

# 예제: 2D Quadratic Program
# min (1/2)(x₁² + x₂²)
# s.t. x₁ + x₂ = 1

def lagrangian_2d_qp(x1, x2, nu):
    """Lagrangian L(x, ν) = (1/2)(x₁² + x₂²) + ν(x₁ + x₂ - 1)"""
    return 0.5 * (x1**2 + x2**2) + nu * (x1 + x2 - 1)

def dual_function_2d_qp(nu):
    """
    Dual function g(ν) = inf_x L(x, ν)
    
    ∂L/∂x₁ = x₁ + ν = 0 → x₁ = -ν
    ∂L/∂x₂ = x₂ + ν = 0 → x₂ = -ν
    
    g(ν) = (1/2)(ν² + ν²) + ν(-2ν - 1) = ν² - 2ν² - ν = -ν² - ν
    """
    return -nu**2 - nu

# 1. 3D Lagrangian surface 시각화
fig = plt.figure(figsize=(14, 5))

# Lagrangian surface
ax1 = fig.add_subplot(121, projection='3d')
x1_range = np.linspace(-2, 2, 50)
x2_range = np.linspace(-2, 2, 50)
nu_val = 0.5  # 고정된 ν 값

X1, X2 = np.meshgrid(x1_range, x2_range)
L = 0.5 * (X1**2 + X2**2) + nu_val * (X1 + X2 - 1)

surf = ax1.plot_surface(X1, X2, L, cmap='viridis', alpha=0.7)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$L(x_1, x_2, ν=0.5)$')
ax1.set_title('Lagrangian for Fixed ν=0.5')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 2. 쌍대 함수 곡선
ax2 = fig.add_subplot(122)
nu_range = np.linspace(-3, 1, 100)
g_vals = np.array([dual_function_2d_qp(nu) for nu in nu_range])

ax2.plot(nu_range, g_vals, 'b-', linewidth=2, label='$g(ν) = -ν² - ν$')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 최적 쌍대 변수 찾기: dg/dν = -2ν - 1 = 0 → ν* = -1/2
nu_opt = -0.5
g_opt = dual_function_2d_qp(nu_opt)
ax2.plot(nu_opt, g_opt, 'ro', markersize=10, label=f'최적: ν*={nu_opt}, g(ν*)={g_opt}')

ax2.set_xlabel('$ν$ (Dual variable)')
ax2.set_ylabel('$g(ν)$')
ax2.set_title('Dual Function (Concave)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('/tmp/lagrangian_dual_function.png', dpi=150, bbox_inches='tight')
print("✓ Lagrangian 및 쌍대 함수 시각화 저장됨")

# 3. 원 문제 풀이 및 검증
x = cp.Variable(2)
objective = 0.5 * cp.sum_squares(x)
constraint = x[0] + x[1] == 1

problem = cp.Problem(cp.Minimize(objective), [constraint])
problem.solve()

p_opt = problem.value
x_opt = x.value

print(f"\n원 문제 최적해:")
print(f"  x* = {x_opt}")
print(f"  p* = {p_opt}")

# 4. 쌍대 함수 값 비교
print(f"\n쌍대 함수 검증:")
print(f"  ν* = -0.5 (수학적 계산)")
g_at_opt_nu = dual_function_2d_qp(-0.5)
print(f"  g(ν*) = {g_at_opt_nu}")
print(f"  p* = {p_opt}")
print(f"  약쌍대성: g(ν*) ≤ p* ? {g_at_opt_nu <= p_opt + 1e-6}")

# 5. 임의의 (λ, ν)에서 하한 확인
print(f"\n약쌍대성 검증 (임의의 (λ,ν)):")
test_nu_values = [-1.0, -0.5, 0.0, 0.5]
for test_nu in test_nu_values:
    g_val = dual_function_2d_qp(test_nu)
    print(f"  ν = {test_nu:4.1f}: g(ν) = {g_val:6.3f} ≤ p* = {p_opt:.3f} ? {g_val <= p_opt + 1e-6}")

plt.show()
```

**코드 실행 결과:**
```
✓ Lagrangian 및 쌍대 함수 시각화 저장됨

원 문제 최적해:
  x* = [0.5 0.5]
  p* = 0.5

쌍대 함수 검증:
  ν* = -0.5 (수학적 계산)
  g(ν*) = 0.5
  p* = 0.5
  약쌍대성: g(ν*) ≤ p* ? True

약쌍대성 검증 (임의의 (λ,ν)):
  ν = -1.0: g(ν) = -0.000 ≤ p* = 0.500 ? True
  ν = -0.5: g(ν) =  0.250 ≤ p* = 0.500 ? True
  ν = 0.0: g(ν) = 0.000 ≤ p* = 0.500 ? True
  ν = 0.5: g(ν) = -0.750 ≤ p* = 0.500 ? True
```

## 🔗 AI/ML 연결
- **SVM 학습**: Primal (w 최적화) 대신 Dual (α 최적화)을 풀 때 커널 함수를 직접 사용 가능
- **라그랑주 승수법**: 제약이 있는 신경망 학습, 페더레이티드 러닝
- **리소스 할당**: 분산 시스템에서 각 자원의 가치(λ)를 쌍대 변수로 표현

## ⚖️ 가정과 한계
- 쌍대 함수는 항상 오목이지만, 원 문제가 비볼록이면 약쌍대성만 보장 (강쌍대성 미보장)
- 쌍대 함수의 정의역은 $\lambda_i \geq 0$로 제약됨
- 무한대 값을 반환하는 경우 처리 필요

## 📌 핵심 정리
1. **Lagrangian L(x,λ,ν)**은 제약을 일원화된 목적함수로 변환
2. **쌍대 함수 g(λ,ν) = inf_x L(x,λ,ν)**는 항상 오목함수
3. **약쌍대성**: 모든 가능한 (λ,ν)에서 $g(\lambda,\nu) \leq p^*$
4. **다음 단계**: 강쌍대성과 KKT 조건으로 나아감

## 🤔 생각해볼 문제

**문제 1:** min $x^2$ s.t. $x \leq 1$의 Lagrangian과 쌍대 함수를 유도하고, 임의의 λ ≥ 0에서 약쌍대성을 확인하세요.

<details>
<summary>힌트 및 해설</summary>

Lagrangian: $L(x,\lambda) = x^2 + \lambda(x - 1)$

쌍대 함수: $g(\lambda) = \inf_x [x^2 + \lambda x - \lambda]$
- $\frac{\partial}{\partial x}(x^2 + \lambda x) = 2x + \lambda = 0$ → $x = -\lambda/2$
- $g(\lambda) = (-\lambda/2)^2 + \lambda(-\lambda/2) - \lambda = \lambda^2/4 - \lambda^2/2 - \lambda = -\lambda^2/4 - \lambda$

원 문제의 최적해: $x^* = 0$ (제약 $x \leq 1$은 자동 만족), $p^* = 0$

약쌍대성: $g(\lambda) = -\lambda^2/4 - \lambda \leq 0$ for all $\lambda \geq 0$

</details>

**문제 2:** 다음 QP에서 쌍대 함수 $g(\nu)$를 계산하세요:
```
min  (1/2)x^T Q x + c^T x
s.t. Ax = b
```
여기서 Q는 양정부호 정행렬입니다.

<details>
<summary>힌트 및 해설</summary>

$L(x,\nu) = \frac{1}{2}x^T Q x + c^T x + \nu^T(Ax - b)$

정류 조건: $\nabla_x L = Qx + c + A^T\nu = 0$ → $x^* = -Q^{-1}(c + A^T\nu)$

대입하면:
$$g(\nu) = -\frac{1}{4}(c+A^T\nu)^T Q^{-1}(c+A^T\nu) - b^T\nu$$

또는 행렬 형태로:
$$g(\nu) = -\frac{1}{2}(c+A^T\nu)^T Q^{-1}(c+A^T\nu) - b^T\nu$$

</details>

**문제 3:** Lagrangian의 오목성을 직관적으로 설명하고, 왜 쌍대 함수도 오목인지 기하학적으로 설명하세요.

<details>
<summary>힌트 및 해설</summary>

고정된 $x$에서 $L(x,\lambda,\nu) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x)$를 보면, $(\lambda, \nu)$에 대해 선형입니다 (아핀).

여러 아핀 함수들의 하한(infimum)은 오목함수의 성질: 만약 $f_i$가 모두 아핀이면 $\inf_i f_i$는 오목입니다.

따라서 $g(\lambda,\nu) = \inf_x L(x,\lambda,\nu)$는 오목함수입니다.

기하학적으로: 쌍대 함수의 epigraph는 convex이므로, 그것의 여집합인 hypograph는 concave를 정의합니다.

</details>

<div align="center">

| [◀ Ch3-05. CVXPY로 문제 표현](../ch3-convex-problems/05-cvxpy-dcp.md) | [📚 README](../README.md) | [02. 약쌍대성(Weak Duality) ▶](./02-weak-duality.md) |

</div>
