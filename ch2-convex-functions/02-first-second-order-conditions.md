# 2. 일계·이계 조건

## 🎯 핵심 질문

- 1차 미분으로 볼록성을 어떻게 판단할 수 있는가?
- 헤시안(Hessian)이 양반정치(PSD)라는 것이 정말 함수의 볼록성을 의미하는가?
- 미분 불가능한 점에서의 "subgradient"는 무엇인가?

---

## 🔍 왜 이 이론이 AI에서 중요한가

1. **경사하강법(GD)의 수렴 분석**: $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ 조건이 GD가 감소하는 이유를 설명하고, 수렴 속도 분석의 기반을 제공.

2. **2차 방법(Newton)의 설계**: Newton 방법은 헤시안이 PSD일 때만 안정적; $\nabla^2 f \succ 0$ (positive definite)이면 스텝 크기를 조정할 수 있음.

3. **정규화의 효과**: Ridge 회귀 $\min_w \|Xw-y\|^2 + \lambda\|w\|^2$에서 $\lambda > 0$이 헤시안을 PSD로 강화하여 수치 안정성 개선.

4. **Subgradient의 활용**: ReLU, L1 정규화 같은 미분 불가능 함수를 최적화할 때 subgradient를 사용.

---

## 📐 수학적 선행 조건

- 행렬의 양반정치성(PSD: $A \succeq 0$), 고유값 판정법
- Hessian 행렬의 계산 (2차 편미분)
- [이전 문서: 01. 볼록 함수의 3개 동치 정의](./01-convex-function-definitions.md)
- Taylor 전개 (2차까지)
- 선형대수: 행렬의 고유분해(eigenvalue decomposition)

---

## 📖 직관적 이해

### 1차 조건의 기하학적 의미

```
        f
        ↑
        │     f(y)
        │    /│
        │   / │  실제 함수
        │  /  │
        │ /   │ 접선: f(x) + ∇f(x)ᵀ(y-x)
        │*────*─────────────────
        0  x  y       (y-x)
        
접선(tangent line)이 항상 함수 위에 있음 = 1차 조건
```

**의미**: 한 점에서의 그래디언트(기울기)만으로도 다른 모든 점에서의 함수값의 하한을 구할 수 있다.

### 2차 조건의 기하학적 의미

헤시안 $H = \nabla^2 f(x)$를 고유분해하면:
$$H = \sum_i \lambda_i v_i v_i^T, \quad \lambda_i \geq 0 \text{ (PSD 조건)}$$

각 방향 $v_i$에서 함수가 "위로 굽음"(curvature > 0):
$$f(x + tv_i) \approx f(x) + \nabla f(x)^T(tv_i) + \frac{t^2}{2}\lambda_i \geq f(x)$$

### Subgradient: 미분불가능점에서의 확장

```
   f(x)
      ↑
      │     /|
      │    / |  함수
      │   /  |
      │  /   |
      │ *────*────  모든 subgradient들
      |/     |
      └──────────→ x
      
x에서의 subgradient: f(y) ≥ f(x) + g·(y-x) for all y를 만족하는 g
```

**의미**: 미분 불가능한 점에서도 "접선의 기울기 후보"들의 집합(convex set)이 존재.

---

## ✏️ 엄밀한 정의

**정의 2.5** (1차 조건과 볼록성)
미분가능 함수 $f: \mathbb{R}^n \to \mathbb{R}$가 볼록 $\Leftrightarrow$ 모든 $x, y \in \text{dom}(f)$에 대해:
$$f(y) \geq f(x) + \nabla f(x)^T(y-x)$$

**정의 2.6** (2차 조건과 볼록성)
두 번 미분가능 함수 $f: \mathbb{R}^n \to \mathbb{R}$가 볼록 $\Leftrightarrow$ 모든 $x \in \text{dom}(f)$에 대해:
$$\nabla^2 f(x) \succeq 0 \quad \text{(양반정치)}$$

**정의 2.7** (Subgradient)
벡터 $g \in \mathbb{R}^n$이 함수 $f$의 점 $x$에서의 **subgradient** $\Leftrightarrow$
$$f(y) \geq f(x) + g^T(y-x) \quad \text{for all } y$$

점 $x$에서의 모든 subgradient의 집합을 $\partial f(x)$ (subdifferential)이라 함.

**정의 2.8** (강볼록성의 1차 형태, 미리)
$f$가 $\mu > 0$에 대해 **$\mu$-strongly convex** $\Leftrightarrow$
$$f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$$

---

## 🔬 정리와 증명

**정리 2.4** (1차 조건의 필요충분 조건, 미분가능 경우)
미분가능 함수 $f$가 볼록 $\Leftrightarrow$ 모든 $x, y$에 대해 $f(y) \geq f(x) + \nabla f(x)^T(y-x)$

**증명:**
($\Rightarrow$) $f$가 볼록이라고 가정. 임의의 $x, y$와 $\lambda \in (0,1]$에 대해:
$$f(x + \lambda(y-x)) \leq (1-\lambda)f(x) + \lambda f(y)$$

정리하면:
$$\frac{f(x + \lambda(y-x)) - f(x)}{\lambda} \leq f(y) - f(x)$$

$\lambda \to 0^+$로 극한:
$$\nabla f(x)^T(y-x) = \lim_{\lambda \to 0^+} \frac{f(x + \lambda(y-x)) - f(x)}{\lambda} \leq f(y) - f(x)$$

따라서 $f(y) \geq f(x) + \nabla f(x)^T(y-x)$. □

($\Leftarrow$) 1차 조건이 성립한다고 가정. $\lambda \in (0,1)$이고 $z = \lambda x + (1-\lambda)y$라 하자.
$$f(x) \geq f(z) + \nabla f(z)^T(x-z)$$
$$f(y) \geq f(z) + \nabla f(z)^T(y-z)$$

$\lambda$배, $(1-\lambda)$배해서 더하면:
$$\lambda f(x) + (1-\lambda)f(y) \geq f(z) + \nabla f(z)^T[\lambda(x-z) + (1-\lambda)(y-z)]$$

$\lambda(x-z) + (1-\lambda)(y-z) = \lambda(x - \lambda x - (1-\lambda)y) + (1-\lambda)(y - \lambda x - (1-\lambda)y)$
$= \lambda(1-\lambda)(x-y) + (1-\lambda)(-\lambda)(x-y) = 0$

따라서 $f(z) \leq \lambda f(x) + (1-\lambda)f(y)$, 즉 $f$는 Jensen을 만족하므로 볼록. □

---

**정리 2.5** (2차 조건 필요충분 조건)
두 번 미분가능 함수 $f$가 볼록 $\Leftrightarrow$ 모든 $x \in \text{dom}(f)$에 대해 $\nabla^2 f(x) \succeq 0$

**증명 스케치:**
Taylor 전개 (2차):
$$f(y) = f(x) + \nabla f(x)^T(y-x) + \frac{1}{2}(y-x)^T\nabla^2 f(\xi)(y-x)$$
여기서 $\xi \in [x, y]$ (선분 위의 어떤 점).

$\nabla^2 f(\xi) \succeq 0$이면 마지막 항 $\geq 0$이므로:
$$f(y) \geq f(x) + \nabla f(x)^T(y-x)$$

이는 1차 조건이므로 $f$는 볼록.

역방향: $f$가 볼록이면 1차 조건 성립 → 1차 조건을 $x$에서 미분 → $\nabla^2 f(x) \succeq 0$. □

---

**정리 2.6** (Subgradient의 존재와 연결)
$f$가 미분가능하면 $\partial f(x) = \{\nabla f(x)\}$ (한 점 집합).

**증명:** 미분가능하면 1차 조건에서 $f(y) \geq f(x) + \nabla f(x)^T(y-x)$이므로 $\nabla f(x) \in \partial f(x)$.

역으로, $g \in \partial f(x)$이면 $f(y) \geq f(x) + g^T(y-x)$ for all $y$.

$y = x + t(y-x)$ (작은 $t$)로 놓으면:
$$f(x+th) \geq f(x) + g^T(th) \implies \frac{f(x+th)-f(x)}{t} \geq g^Th$$

$t \to 0$이면 $\nabla f(x)^Th \geq g^Th$ for all $h$, 따라서 $g = \nabla f(x)$. □

---

**예제 2.3**: $f(x) = x^TAx$ (A는 $n \times n$ 행렬)의 볼록성 판정

헤시안: $\nabla^2 f(x) = A + A^T$

$f$가 볼록 $\Leftrightarrow A + A^T \succeq 0$ $\Leftrightarrow$ $A$가 대칭이고 양반정치.

만약 $A$가 대칭이고 $A \succeq 0$ (모든 고유값 $\geq 0$)이면, $\nabla^2 f(x) = 2A \succeq 0$이므로 $f$는 볼록.

**검증 (NumPy)**:
```python
import numpy as np

# A가 PSD인 경우
A = np.array([[2, 1], [1, 2]])  # 고유값: 1, 3 (모두 > 0)
eigenvalues = np.linalg.eigvalsh(A)
print(f"A의 고유값: {eigenvalues}, PSD: {np.all(eigenvalues >= 0)}")

# Hessian = 2A도 PSD
hessian = 2*A
eigenvalues_h = np.linalg.eigvalsh(hessian)
print(f"Hessian의 고유값: {eigenvalues_h}, PSD: {np.all(eigenvalues_h >= 0)}")
```

---

**예제 2.4**: Log-Sum-Exp의 헤시안이 PSD임을 증명

$f(x) = \log(\sum_{i=1}^n e^{x_i})$

**1계 도함수 (softmax)**:
$$\frac{\partial f}{\partial x_i} = \frac{e^{x_i}}{\sum_j e^{x_j}} =: p_i \quad (p_i \geq 0, \sum_i p_i = 1)$$

**2계 도함수**:
$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial p_i}{\partial x_j} = \begin{cases}
p_i(1-p_i) & \text{if } i=j \\
-p_i p_j & \text{if } i \neq j
\end{cases}$$

행렬 형태: $\nabla^2 f(x) = \text{diag}(p) - pp^T$ (여기서 $p = [p_1, \ldots, p_n]^T$)

**PSD 증명**: $v \in \mathbb{R}^n$에 대해:
$$v^T\nabla^2 f v = v^T(\text{diag}(p) - pp^T)v = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2$$

Cauchy-Schwarz: $(\sum_i p_i v_i)^2 \leq (\sum_i p_i)(sum_i p_i v_i^2) = \sum_i p_i v_i^2$

따라서 $v^T\nabla^2 f v \geq 0$. □

---

**예제 2.5**: ReLU의 Subgradient ($\partial f(0) = [0, 1]$)

$f(x) = \max(0, x)$

- $x > 0$에서: $\nabla f(x) = 1$, $\partial f(x) = \{1\}$
- $x < 0$에서: $\nabla f(x) = 0$, $\partial f(x) = \{0\}$
- $x = 0$에서: 미분불가능

$x = 0$에서의 subgradient: $f(y) \geq f(0) + g(y-0) = g \cdot y$를 만족하는 $g$?

$y > 0$: $y \geq g \cdot y \implies g \leq 1$
$y < 0$: $0 \geq g \cdot y \implies g \geq 0$

따라서 $\partial f(0) = [0, 1]$.

---

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime, minimize
from sympy import symbols, diff, Matrix, simplify, hessian as sp_hessian

# ============================================
# 1. 1차 조건 검증: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
# ============================================

def verify_first_order_condition(f, grad_f, x, y_vals):
    """
    1차 조건 검증: f(y) - f(x) - ∇f(x)ᵀ(y-x) ≥ 0?
    """
    f_x = f(x)
    grad_x = grad_f(x)
    
    violations = []
    for y in y_vals:
        f_y = f(y)
        taylor_approx = f_x + np.dot(grad_x, y - x)
        gap = f_y - taylor_approx
        
        if gap < -1e-10:  # 수치 오차 허용
            violations.append((y, gap))
    
    return violations

# 테스트 함수 1: f(x) = x²
f1 = lambda x: x**2
grad_f1 = lambda x: 2*x

x0 = 1.0
y_vals = np.linspace(-2, 3, 100)
violations = verify_first_order_condition(f1, grad_f1, x0, y_vals)

print("=" * 60)
print("1차 조건 검증: f(x) = x²")
print("=" * 60)
print(f"x = {x0}, 1차 조건 위반 수: {len(violations)}")
if len(violations) == 0:
    print("✓ 모든 y에서 1차 조건 만족!")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

y_range = np.linspace(-2, 3, 200)
f_vals = f1(y_range)
grad_taylor = f1(x0) + grad_f1(x0) * (y_range - x0)

ax = axes[0]
ax.plot(y_range, f_vals, 'b-', linewidth=2.5, label='$f(x) = x^2$')
ax.plot(y_range, grad_taylor, 'r--', linewidth=2, label='Taylor 근사')
ax.fill_between(y_range, f_vals, grad_taylor, where=(f_vals >= grad_taylor), 
                 alpha=0.3, color='green', label='$f(y) ≥ 근사$ (볼록)')
ax.scatter([x0], [f1(x0)], color='red', s=100, zorder=5)
ax.axvline(x0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('y')
ax.set_ylabel('함수값')
ax.set_title('1차 조건: 접선이 항상 함수 아래')
ax.legend()
ax.grid(True, alpha=0.3)

# 테스트 함수 2: f(x) = log-sum-exp
def lse(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

def lse_grad(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

x0_lse = np.array([1.0, 0.5])
y_lse_vals = [np.array([1.0 + t, 0.5 + t]) for t in np.linspace(-1, 1, 100)]
violations_lse = verify_first_order_condition(lse, lse_grad, x0_lse, y_lse_vals)

print("\n1차 조건 검증: log-sum-exp")
print(f"x = {x0_lse}, 1차 조건 위반 수: {len(violations_lse)}")
print("✓ 모든 y에서 1차 조건 만족!" if len(violations_lse) == 0 else "✗ 위반 발견!")

# 1D projection: x + t·d 방향
d = np.array([1.0, -0.5])
t_vals = np.linspace(-1, 1, 100)
f_proj = [lse(x0_lse + t*d) for t in t_vals]
grad_proj = lse_grad(x0_lse)
taylor_proj = lse(x0_lse) + np.dot(grad_proj, t_vals[:, np.newaxis] * d)

ax = axes[1]
ax.plot(t_vals, f_proj, 'b-', linewidth=2.5, label='$f(x_0 + td)$')
ax.plot(t_vals, taylor_proj, 'r--', linewidth=2, label='Taylor 근사')
ax.fill_between(t_vals, f_proj, taylor_proj, where=(np.array(f_proj) >= taylor_proj), 
                 alpha=0.3, color='green')
ax.scatter([0], [lse(x0_lse)], color='red', s=100, zorder=5, label='$x_0$')
ax.set_xlabel('$t$ (step size)')
ax.set_ylabel('함수값')
ax.set_title('log-sum-exp의 1차 조건')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('first_order_condition.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 2. 2차 조건 검증: 헤시안 고유값 계산
# ============================================

print("\n" + "=" * 60)
print("2차 조건 검증: 헤시안의 양반정치성")
print("=" * 60)

# 2.1 이차형식: f(x) = x^T A x
A = np.array([[3, 1], [1, 2]])  # 대칭, PSD (고유값: 1, 4)
eigenvals_A, eigenvecs_A = np.linalg.eigh(A)

def f_quadratic(x, A):
    return x @ A @ x

def hessian_quadratic(A):
    return 2*A

H = hessian_quadratic(A)
eigenvals_H = np.linalg.eigvalsh(H)

print(f"\nf(x) = x^T A x, A = {A.tolist()}")
print(f"  Hessian = 2A의 고유값: {eigenvals_H}")
print(f"  모두 ≥ 0 (PSD): {np.all(eigenvals_H >= 0)} → 볼록 ✓")

# 2.2 Log-Sum-Exp: 기호 계산 + 수치 검증
x1, x2 = symbols('x1 x2', real=True)
lse_expr = sp_hessian(
    x1 + x2 + 1,  # 더 간단한 함수로
    symbols('x1 x2')
)
# 실제로는 복잡하니 수치로

x_test = np.array([1.0, -0.5, 0.3])
h_numerical = np.zeros((3, 3))
eps = 1e-5

grad_at_x = lse_grad(x_test)

for i in range(3):
    for j in range(3):
        x_ij = x_test.copy()
        x_ij[i] += eps
        grad_i_plus = lse_grad(x_ij)
        
        h_numerical[i, j] = (grad_i_plus[j] - grad_at_x[j]) / eps

eigenvals_lse = np.linalg.eigvalsh(h_numerical)
print(f"\nlog-sum-exp at x = {x_test}")
print(f"  Hessian의 고유값 (수치): {eigenvals_lse}")
print(f"  모두 ≥ 0 (PSD): {np.all(eigenvals_lse >= -1e-6)} → 볼록 ✓")

# ============================================
# 3. Subgradient 검증: ReLU
# ============================================

print("\n" + "=" * 60)
print("Subgradient 검증: ReLU 함수")
print("=" * 60)

def relu(x):
    return np.maximum(0, x)

def subdiff_relu(x):
    """
    ReLU의 Subgradient 집합 반환
    """
    if x > 0:
        return [1.0]
    elif x < 0:
        return [0.0]
    else:  # x == 0
        return np.linspace(0, 1, 11)  # [0, 1]의 부분 샘플

# ReLU at x=0에서의 subgradient 검증
x_test = 0.0
subdiff = subdiff_relu(x_test)

print(f"\nReLU at x = {x_test}")
print(f"  ∂f(0) = [0, 1] (subgradient set)")

# 각 g ∈ [0, 1]에 대해 f(y) ≥ f(0) + g(y-0)인지 확인
y_test_vals = np.linspace(-1, 1, 50)
all_valid = True

for g in subdiff:
    for y in y_test_vals:
        f_y = relu(y)
        lower_bound = relu(x_test) + g * (y - x_test)
        if f_y < lower_bound - 1e-10:
            all_valid = False
            print(f"  위반: g={g}, y={y}")

print(f"  모든 subgradient에서 부등식 만족: {all_valid} ✓")

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))

x_range = np.linspace(-1.5, 1.5, 200)
y_range = relu(x_range)

ax.plot(x_range, y_range, 'b-', linewidth=3, label='ReLU: $f(x) = \\max(0,x)$')

# x=0에서의 subgradient들
x0 = 0
for g in np.linspace(0, 1, 5):
    y_line = g * (x_range - x0)
    ax.plot(x_range, y_line, '--', alpha=0.6, linewidth=1.5, label=f'g={g:.1f}')

ax.axvline(0, color='red', linestyle=':', alpha=0.5, linewidth=2)
ax.scatter([0], [0], color='red', s=150, zorder=5, marker='o', 
           edgecolors='black', linewidths=2, label='$x=0$ (미분불가)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title('ReLU의 Subgradient: $\\partial f(0) = [0,1]$')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('subgradient_relu.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Subgradient 시각화 완료")

# ============================================
# 4. 강볼록성 검증
# ============================================

print("\n" + "=" * 60)
print("강볼록성(Strong Convexity) 검증")
print("=" * 60)

def verify_strong_convexity(f, x, y, mu):
    """
    μ-강볼록성 검증: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)‖y-x‖²
    """
    f_x = f(x)
    grad_x = approx_fprime(x, f, epsilon=1e-5)
    f_y = f(y)
    
    lhs = f_y
    rhs = f_x + np.dot(grad_x, y - x) + (mu/2) * np.linalg.norm(y - x)**2
    
    return lhs, rhs, lhs - rhs

# 테스트: f(x) = (1/2)x^T(A)x + bᵀx, A PSD
A_strong = np.array([[3, 0], [0, 2]])  # 고유값: 2, 3
b = np.array([1, -1])

def f_strongly_convex(x, A, b):
    return 0.5 * (x @ A @ x) + b @ x

# 최소 고유값이 μ (강볼록성 상수)
mu_min = np.min(np.linalg.eigvalsh(A_strong))
print(f"\nf(x) = (1/2)xᵀAx + bᵀx")
print(f"  A의 최소 고유값: {mu_min}")
print(f"  → μ={mu_min} 강볼록성 기대")

x_test = np.array([0.5, -0.3])
y_test = np.array([1.2, 0.8])

f = lambda x: f_strongly_convex(x, A_strong, b)
lhs, rhs, gap = verify_strong_convexity(f, x_test, y_test, mu_min)

print(f"  검증 (x={x_test}, y={y_test}):")
print(f"    f(y) = {lhs:.6f}")
print(f"    RHS (1차+강볼록항) = {rhs:.6f}")
print(f"    Gap = {gap:.6e} (≥ 0이어야 함) {'✓' if gap >= -1e-6 else '✗'}")

print("\n" + "=" * 60)
print("모든 검증 완료!")
print("=" * 60)
```

---

## 🔗 AI/ML 연결

1. **경사하강법의 안정성**: 1차 조건 $f(y) \geq f(x) + \nabla f(x)^T(y-x)$에서, $y = x - \alpha \nabla f(x)$로 선택하면 $f(y) - f(x) \approx -\alpha \|\nabla f(x)\|^2 < 0$ (감소 보장)

2. **Newton 방법의 수렴성**: 헤시안이 일정 범위 내에서 PSD이고 bounded이면, Newton 방법은 2차 수렴(quadratic convergence)

3. **정규화(Regularization)의 필요성**: $\lambda\|w\|^2$ 정규화항을 추가하면 전체 손실함수의 헤시안에 $2\lambda I$를 더해, 고유값의 최솟값을 $2\lambda$로 강제 → 수치 안정성 개선

---

## ⚖️ 가정과 한계

| 개념 | 가정 | 한계 |
|-----|------|------|
| **1차 조건** | 미분가능성 | ReLU 같은 미분불가능 함수는 subgradient로 확장 필요 |
| **2차 조건** | 2회 미분가능 | 일부 ML 함수(cross-entropy)는 Hessian이 명시적 형태 복잡 |
| **PSD 판정** | $\nabla^2 f \succeq 0$ globally | 국소적 PSD만 확인 가능 (수치적으로) |
| **강볼록성** | $\nabla^2 f \succeq \mu I$ | $\mu$를 과대평가하면 비현실적 수렴 속도 예측 |

---

## 📌 핵심 정리

| 개념 | 수식 | 의미 |
|-----|------|------|
| **1차 조건** | $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ | 접선이 함수 아래 → 미분가능 함수의 기본 |
| **2차 조건** | $\nabla^2 f(x) \succeq 0$ | 모든 방향에서 곡률 ≥ 0 |
| **Subgradient** | $\partial f(x) = \{g : f(y) \geq f(x)+g^T(y-x)\}$ | 미분불가능점에서의 기울기 집합 |
| **강볼록성** | $f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$ | 유일해 존재 + 빠른 수렴 |

---

## 🤔 생각해볼 문제

**문제 2.4**: 다음 함수가 어느 범위에서 1차 조건을 만족하는지 확인하시오.
$$f(x) = \sqrt{1 + x^2}$$

<details>
<summary>힌트 및 해설</summary>

**힌트**: 1계 도함수를 계산하고, 임의의 두 점에서 부등식을 확인하세요.

**해설**:
$$\nabla f(x) = \frac{x}{\sqrt{1+x^2}}$$

함수 $f$는 엄격히 볼록하므로 1차 조건을 만족합니다.
$$\sqrt{1+y^2} \geq \sqrt{1+x^2} + \frac{x}{\sqrt{1+x^2}}(y-x)$$

이는 Taylor 전개와 2차 항의 양성으로 증명할 수 있습니다.

</details>

---

**문제 2.5**: 행렬 $A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$에 대해, $f(x) = x^TAx$가 볼록한지 판정하고 그 이유를 설명하시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: $A$의 고유값을 계산하세요.

**해설**:
$$\det(A - \lambda I) = \det\begin{pmatrix} 1-\lambda & 2 \\ 2 & 1-\lambda \end{pmatrix} = (1-\lambda)^2 - 4 = \lambda^2 - 2\lambda - 3$$
$$= (\lambda - 3)(\lambda + 1)$$

고유값: $\lambda_1 = 3 > 0$, $\lambda_2 = -1 < 0$

$A$가 indefinite (mixed sign)이므로 **$f$는 볼록하지 않음**.

Hessian = $2A$도 고유값이 $\{6, -2\}$이므로 PSD가 아님.

</details>

---

**문제 2.6**: ReLU 함수의 Subgradient $\partial \text{ReLU}(x)$를 모든 $x \in \mathbb{R}$에 대해 구하고, 이것이 항상 부등식 $f(y) \geq f(x) + g(y-x)$를 만족하는지 확인하시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: 세 경우 $x > 0$, $x < 0$, $x = 0$을 나누어 생각하세요.

**해설**:
$$\partial \text{ReLU}(x) = \begin{cases}
\{1\} & \text{if } x > 0 \\
\{0\} & \text{if } x < 0 \\
[0, 1] & \text{if } x = 0
\end{cases}$$

각 경우에서:
- $x > 0, g=1$: $\max(0,y) \geq \max(0,x) + 1 \cdot (y-x) = x + (y-x) = y$
  - $y \geq 0$이면 $\max(0,y) = y$ ✓
  - $y < 0$이면 $0 > y$ ✗ 인데, $\max(0,y) = 0 \geq y$이므로 실제로는 성립
  
실제로는 모든 경우에 부등식이 성립합니다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 볼록 함수의 3개 동치 정의](./01-convex-function-definitions.md) | [📚 README](../README.md) | [03. 강볼록성과 매끄러움 ▶](./03-strong-convexity-smoothness.md) |

</div>
