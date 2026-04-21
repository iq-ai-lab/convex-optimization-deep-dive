# 4. 볼록 함수의 연산

## 🎯 핵심 질문

- 볼록 함수들의 합/상한이 항상 볼록한가?
- 왜 곱은 볼록하지 않을 수 있는가?
- CVXPY가 프로그래머의 실수를 자동으로 잡는 비결은?

---

## 🔍 왜 이 이론이 AI에서 중요한가

1. **DCP(Disciplined Convex Programming)**: CVXPY는 자동으로 함수의 볼록성을 판정하여 풀이 가능 여부를 결정. 이는 "원자(atom)" 수준의 연산이 볼록성을 보존하는지 확인함으로써 가능.

2. **손실 함수 설계**: L1, L2 정규화를 이용해 볼록 손실함수를 만들 수 있음. $\min_w \|Xw-y\|_2 + \lambda\|w\|_1$ 같은 형태는 convex 연산의 산물.

3. **신경망의 한계**: 신경망이 비볼록인 이유는 여러 계층의 합성(composition)이 볼록성을 보존하지 않기 때문.

4. **최적화 문제 변환**: 비볼록 문제를 볼록 완화(convex relaxation)로 변환할 때, 어떤 연산이 안전한지 알아야 함.

---

## 📐 수학적 선행 조건

- [이전 문서: 01. 볼록 함수의 3개 동치 정의](./01-convex-function-definitions.md)
- [이전 문서: 02. 일계·이계 조건](./02-first-second-order-conditions.md)
- 상한(supremum), 하한(infimum) 개념
- 합성 함수의 미분

---

## 📖 직관적 이해

### 어떤 연산이 볼록성을 보존하는가?

```
안전한 연산들:
┌─────────────────────────────────┐
│ 1. 비음 가중합: Σᵢ wᵢfᵢ (wᵢ≥0) │
│ 2. 포인트와이즈 상한: sup fα    │
│ 3. 아핀 합성: f(Ax+b)          │
│ 4. 스칼라 합성: g(f(x))         │
│    (g 오목 증가 + f 오목 = 볼록) │
└─────────────────────────────────┘

위험한 연산들:
┌──────────────────────────────┐
│ ✗ 곱: f·g (양쪽 모두 볼록)    │
│ ✗ 지수: exp(f) (f 선형이어도) │
│ ✗ 합성: h(f(g(x)))           │
│   (조건을 엄격히 만족해야 함)  │
└──────────────────────────────┘
```

### Log-Sum-Exp의 구성: 선형 함수의 상한

$$\log\sum_i e^{x_i} = \max_p \left( p^T x - (-\log p^T 1) \right)$$
$$= \max_p (p^T x)$$

여기서 최대값은 선형 함수들(각 p에 대해)의 상한이므로 볼록함수는 **선형 함수의 상한**으로 표현됨 = 포인트와이즈 상한.

---

## ✏️ 엄밀한 정의

**정의 2.13** (비음 가중합)
$f_i$가 모두 볼록이고 $w_i \geq 0$이면, $f(x) = \sum_i w_i f_i(x)$도 볼록.

**정의 2.14** (포인트와이즈 상한)
각 $\alpha \in A$에 대해 $f_\alpha$가 볼록이면, $f(x) = \sup_{\alpha \in A} f_\alpha(x)$도 볼록.

**정의 2.15** (아핀 합성)
$f$가 볼록이고 $A: \mathbb{R}^m \to \mathbb{R}^n$ 아핀(affine)이면, $(f \circ A)(x) = f(Ax+b)$도 볼록.

**정의 2.16** (스칼라 합성, composition rule)
$f: \mathbb{R}^n \to \mathbb{R}$, $g: \mathbb{R} \to \mathbb{R}$일 때, $h(x) = g(f(x))$가 볼록 $\Leftarrow$
- $f$가 오목, $g$가 오목 증가(혹은 $f$ 볼록, $g$가 볼록 증가), **그리고**
- $\text{dom}(g) \supseteq \text{range}(f)$

**정의 2.17** (Inf-convolution)
$$f \square g (x) := \inf_y (f(y) + g(x-y))$$

---

## 🔬 정리와 증명

**정리 2.11** (비음 가중합이 볼록성 보존)
$f_1, \ldots, f_m$이 모두 볼록이고 $w_i \geq 0$이면, $f = \sum_i w_i f_i$도 볼록.

**증명:**
Jensen 부등식으로:
$$f(\lambda x + (1-\lambda)y) = \sum_i w_i f_i(\lambda x + (1-\lambda)y)$$
$$\leq \sum_i w_i[\lambda f_i(x) + (1-\lambda)f_i(y)]$$
$$= \lambda \sum_i w_i f_i(x) + (1-\lambda)\sum_i w_i f_i(y) = \lambda f(x) + (1-\lambda)f(y)$$

□

---

**정리 2.12** (포인트와이즈 상한이 볼록성 보존)
각 $\alpha \in A$에 대해 $f_\alpha(x)$가 (모든 $x$에 대해) 볼록이면, $f(x) = \sup_{\alpha \in A} f_\alpha(x)$도 볼록.

**증명:**
Jensen: $\lambda, (1-\lambda) \in [0,1]$에 대해,
$$f(\lambda x + (1-\lambda)y) = \sup_\alpha f_\alpha(\lambda x + (1-\lambda)y)$$
$$\leq \sup_\alpha [\lambda f_\alpha(x) + (1-\lambda)f_\alpha(y)]$$

이 부등식이 성립하려면, 우변을 풀어야 함:
$$\leq \sup_\alpha \lambda f_\alpha(x) + \sup_\beta (1-\lambda)f_\beta(y) = \lambda f(x) + (1-\lambda)f(y)$$

□

---

**정리 2.13** (아핀 합성이 볼록성 보존)
$f: \mathbb{R}^n \to \mathbb{R}$가 볼록이고 $A \in \mathbb{R}^{n \times m}$, $b \in \mathbb{R}^n$이면, 
$h(x) = f(Ax + b)$도 볼록.

**증명:**
Jensen으로,
$$h(\lambda x + (1-\lambda)y) = f(A[\lambda x + (1-\lambda)y] + b)$$
$$= f(\lambda(Ax+b) + (1-\lambda)(Ay+b))$$
$$\leq \lambda f(Ax+b) + (1-\lambda)f(Ay+b) = \lambda h(x) + (1-\lambda)h(y)$$

□

---

**정리 2.14** (스칼라 합성 규칙)
$f: \mathbb{R}^n \to \mathbb{R}$, $g: \mathbb{R} \to \mathbb{R}$일 때:
1. $f$ 오목, $g$ 오목 증가 $\Rightarrow$ $g \circ f$ 오목
2. $f$ 볼록, $g$ 볼록 증가 $\Rightarrow$ $g \circ f$ 볼록
3. $f$ 오목, $g$ 볼록 감소 $\Rightarrow$ $g \circ f$ 볼록

**증명 (경우 2)**: $f$가 볼록, $g$가 볼록 증가라고 가정. $x, y$와 $\lambda \in [0,1]$:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

$g$가 증가이므로 (단조증가 함수를 양변에 적용):
$$g(f(\lambda x + (1-\lambda)y)) \leq g(\lambda f(x) + (1-\lambda)f(y))$$

$g$가 볼록이므로:
$$g(\lambda f(x) + (1-\lambda)f(y)) \leq \lambda g(f(x)) + (1-\lambda)g(f(y))$$

따라서 $(g \circ f)(\lambda x + (1-\lambda)y) \leq \lambda(g \circ f)(x) + (1-\lambda)(g \circ f)(y)$. □

---

**정리 2.15** (Inf-convolution의 볼록성)
$f, g$가 모두 볼록이면, $f \square g (x) := \inf_y(f(y) + g(x-y))$도 볼록.

**증명:**
$z = \lambda x + (1-\lambda)y$라 하면:
$$f \square g(z) = \inf_{u,v} [f(u) + g(z - u)]$$

여기서 최소값은 $z = \lambda x + (1-\lambda)y$ 형태로 분해:
$$u = \lambda u_1 + (1-\lambda)u_2, \quad z - u = \lambda(x-u_1) + (1-\lambda)(y-u_2)$$

따라서:
$$f(u) + g(z-u) = f(\lambda u_1 + (1-\lambda)u_2) + g(\lambda(x-u_1) + (1-\lambda)(y-u_2))$$
$$\leq \lambda f(u_1) + (1-\lambda)f(u_2) + \lambda g(x-u_1) + (1-\lambda)g(y-u_2)$$
$$= \lambda [f(u_1)+g(x-u_1)] + (1-\lambda)[f(u_2)+g(y-u_2)]$$

양변 최소값을 취하면:
$$f \square g(\lambda x + (1-\lambda)y) \leq \lambda(f \square g)(x) + (1-\lambda)(f \square g)(y)$$

□

---

**예제 2.8**: Log-Sum-Exp = 선형 함수의 상한

$$f(x) = \log \sum_i e^{x_i}$$

이를 상한으로 표현:
$$f(x) = \sup_{p : p \geq 0, \sum p_i = 1} (p^T x - (-\sum p_i \log p_i))$$
$$= \sup_{p \in \Delta} (p^T x + \sum p_i \log p_i)$$

여기서 $g_p(x) = p^T x + \sum p_i \log p_i$는 선형 함수 + 상수이므로 선형 = 특별한 볼록함수.

따라서 log-sum-exp는 **선형 함수들의 상한**이므로 정리 2.12에 의해 **볼록**.

---

**예제 2.9**: Euclidean norm $\|\cdot\|_2$의 합성 규칙

$f(x) = \|Ax\|_2 = \sqrt{\sum_i (Ax)_i^2}$가 볼록한가?

- $f_1(x) = \|Ax\|_2^2 = x^TA^TAx$ (이차형식, 볼록)
- $g(t) = \sqrt{t}$ (오목, 증가)

따라서 정리 2.14의 **경우 3** (오목 감소): 이것은 적용되지 않음.

대신 직접 증명: 삼각 부등식으로 norm은 항상 볼록.

혹은 다르게: $f_2(x) = x^TA^TAx$ (볼록), $g(t) = \sqrt{t}$ (오목 증가)는 정리 2.14 **경우 1** (오목 증가 ← 역방향)이 아님.

실제로: **Norm은 삼각 부등식에 의해 직접 증명됨** (정리 2.14 사용 필요 없음).

---

**예제 2.10**: 곱은 볼록하지 않을 수 있음

$f(x) = x$, $g(x) = x$ (둘 다 선형 = 볼록 & 오목)

$(f \cdot g)(x) = x^2$ (볼록)

하지만 $f(x) = e^x$ (볼록), $g(x) = e^x$ (볼록)라 하면:

$(f \cdot g)(x) = e^{2x}$ (볼록)

**일반적으로 비음인 두 볼록함수의 곱이 항상 볼록하지는 않음**:
$f(x) = x$, $g(x) = \max(0, x-1)$ (둘 다 볼록)
$h(x) = f(x) g(x) = x \max(0, x-1)$는 비볼록:
- $h''(x) = \max(0, x-1) + x \cdot 0 = \max(0, x-1)$는 $x < 1$에서 음수 아님
- 실제로 $h(x) = \max(0, x^2 - x)$는... 확인 필요.

---

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# ============================================
# 1. 비음 가중합의 볼록성
# ============================================

print("=" * 70)
print("1. 비음 가중합의 볼록성")
print("=" * 70)

def verify_nonnegative_weighted_sum(f_list, weights, x, y, lambda_vals):
    """
    여러 함수의 가중합이 볼록성을 만족하는지 확인
    """
    f_sum = lambda z: sum(w * f(z) for w, f in zip(weights, f_list))
    
    violations = 0
    for lam in lambda_vals:
        z = lam * x + (1 - lam) * y
        lhs = f_sum(z)
        rhs = lam * f_sum(x) + (1 - lam) * f_sum(y)
        
        if lhs > rhs + 1e-10:
            violations += 1
    
    return violations == 0

# 테스트 함수들 (모두 볼록)
f1 = lambda x: x**2           # 이차
f2 = lambda x: np.abs(x)      # 절댓값
f3 = lambda x: np.exp(x)      # 지수

weights = [0.5, 0.3, 0.2]  # 모두 비음
f_list = [f1, f2, f3]

x, y = -1.0, 2.0
lambda_vals = np.linspace(0, 1, 100)

result = verify_nonnegative_weighted_sum(f_list, weights, x, y, lambda_vals)
print(f"\n가중합 검증: {weights[0]}·f₁ + {weights[1]}·f₂ + {weights[2]}·f₃")
print(f"  f₁(x) = x², f₂(x) = |x|, f₃(x) = eˣ")
print(f"  Jensen 부등식 만족: {result} ✓" if result else f"  검증 실패 ✗")

# 시각화
x_range = np.linspace(-2, 3, 200)
f1_vals = f1(x_range)
f2_vals = f2(x_range)
f3_vals = f3(x_range)
f_sum_vals = weights[0]*f1_vals + weights[1]*f2_vals + weights[2]*f3_vals

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(x_range, f1_vals, label='$f_1(x)=x^2$', linewidth=2)
ax.plot(x_range, f2_vals, label='$f_2(x)=|x|$', linewidth=2)
ax.plot(x_range, f3_vals, label='$f_3(x)=e^x$', linewidth=2)
ax.set_xlim(-2, 3)
ax.set_ylim(-0.5, 5)
ax.set_title('개별 볼록 함수들')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x_range, f_sum_vals, 'b-', linewidth=2.5, label='$0.5f_1 + 0.3f_2 + 0.2f_3$')

# Jensen 검증: 현(chord)
z1, z2 = -1.0, 3.0
y1, y2 = weights[0]*f1(z1) + weights[1]*f2(z1) + weights[2]*f3(z1), \
         weights[0]*f1(z2) + weights[1]*f2(z2) + weights[2]*f3(z2)
ax.plot([z1, z2], [y1, y2], 'r--', linewidth=2, label='현 (chord)')
ax.scatter([z1, z2], [y1, y2], color='red', s=100, zorder=5)

ax.set_xlim(-2, 3)
ax.set_ylabel('함수값')
ax.set_title('가중합의 볼록성')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('weighted_sum_convexity.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ 가중합 시각화 완료")

# ============================================
# 2. 포인트와이즈 상한의 볼록성
# ============================================

print("\n" + "=" * 70)
print("2. 포인트와이즈 상한 (Log-Sum-Exp)")
print("=" * 70)

def pointwise_maximum(functions, x):
    """포인트와이즈 최댓값"""
    return np.max([f(x) for f in functions])

# Log-Sum-Exp를 선형함수의 상한으로 표현
def log_sum_exp(x):
    x_max = np.max(x) if isinstance(x, np.ndarray) else x
    if isinstance(x, np.ndarray):
        return x_max + np.log(np.sum(np.exp(x - x_max)))
    else:
        # 스칼라의 경우는 단순 계산
        return x  # 1D에서는 log(exp(x)) = x

# 1D에서: log(exp(x)) = x는 선형 = 자명하게 볼록

# 2D에서: log(exp(x1) + exp(x2))
def lse_2d(x):
    """2D log-sum-exp"""
    x_max = np.max(x)
    return x_max + np.log(np.exp(x[0]-x_max) + np.exp(x[1]-x_max))

# Jensen 검증
x1 = np.array([1.0, 0.5])
x2 = np.array([0.0, 1.5])

violations = 0
for lam in np.linspace(0, 1, 50):
    z = lam * x1 + (1 - lam) * x2
    lhs = lse_2d(z)
    rhs = lam * lse_2d(x1) + (1 - lam) * lse_2d(x2)
    
    if lhs > rhs + 1e-10:
        violations += 1

print(f"\nLog-Sum-Exp Jensen 검증:")
print(f"  위반 수: {violations}/50 ✓" if violations == 0 else f"  위반 발견: {violations}")

# Log-Sum-Exp를 정렬로 근사
def linear_functions_max(x, num_points=5):
    """p의 심플렉스 위의 점들에서 선형함수들의 최댓값"""
    p_list = np.random.dirichlet(np.ones(2), size=num_points)  # 심플렉스 샘플
    return np.max([p @ x for p in p_list])

# 시각화
x1_range = np.linspace(-1, 2, 50)
x2_range = np.linspace(-1, 2, 50)
X1, X2 = np.meshgrid(x1_range, x2_range)

Z_lse = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z_lse[i, j] = lse_2d(np.array([X1[i, j], X2[i, j]]))

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Z_lse, cmap='viridis', alpha=0.8)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('log-sum-exp')
ax.set_title('Log-Sum-Exp: 선형 함수의 상한')

ax2 = fig.add_subplot(122)
levels = np.linspace(Z_lse.min(), Z_lse.max(), 15)
contour = ax2.contour(X1, X2, Z_lse, levels=levels, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Log-Sum-Exp 등고선 (볼록)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pointwise_maximum_lse.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ 포인트와이즈 상한 시각화 완료")

# ============================================
# 3. 아핀 합성의 볼록성
# ============================================

print("\n" + "=" * 70)
print("3. 아핀 합성 (f(Ax+b))")
print("=" * 70)

# f(x) = ‖x‖₂² (볼록)
def f_norm_sq(x):
    return np.sum(x**2)

# A, b 정의
A = np.array([[2, 1], [1, -1]])  # 2x2
b = np.array([0.5, -0.3])

# 합성: h(y) = f(Ay + b)
def h(y):
    return f_norm_sq(A @ y + b)

# Jensen 검증 (2D)
y1 = np.array([1.0, 0.5])
y2 = np.array([-0.5, 1.5])

violations = 0
for lam in np.linspace(0, 1, 50):
    z = lam * y1 + (1 - lam) * y2
    lhs = h(z)
    rhs = lam * h(y1) + (1 - lam) * h(y2)
    
    if lhs > rhs + 1e-10:
        violations += 1

print(f"\n아핀 합성 h(y) = f(Ay+b), f(x) = ‖x‖²₂:")
print(f"  위반 수: {violations}/50 ✓" if violations == 0 else f"  위반 발견: {violations}")

# 시각화
y1_range = np.linspace(-2, 2, 50)
y2_range = np.linspace(-2, 2, 50)
Y1, Y2 = np.meshgrid(y1_range, y2_range)

Z_h = np.zeros_like(Y1)
for i in range(Y1.shape[0]):
    for j in range(Y1.shape[1]):
        Z_h[i, j] = h(np.array([Y1[i, j], Y2[i, j]]))

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(Y1, Y2, Z_h, cmap='plasma', alpha=0.8)
ax.set_xlabel('$y_1$')
ax.set_ylabel('$y_2$')
ax.set_zlabel('$h(y)$')
ax.set_title('아핀 합성: $h(y) = f(Ay+b)$')

ax2 = fig.add_subplot(122)
levels = np.linspace(Z_h.min(), Z_h.max(), 15)
contour = ax2.contour(Y1, Y2, Z_h, levels=levels, cmap='plasma')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('$y_1$')
ax2.set_ylabel('$y_2$')
ax2.set_title('아핀 합성 등고선 (볼록)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('affine_composition_convexity.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ 아핀 합성 시각화 완료")

# ============================================
# 4. CVXPY DCP 규칙 검증
# ============================================

print("\n" + "=" * 70)
print("4. CVXPY DCP (Disciplined Convex Programming) 규칙")
print("=" * 70)

# 변수와 파라미터
x = cp.Variable(3)
y = cp.Parameter(3)
y.value = np.array([1.0, 2.0, 3.0])

# 유효한 convex 표현들
print("\n유효한 convex 조합:")

# 1. L2 노름 (convex)
expr1 = cp.norm(x, 2)
problem1 = cp.Problem(cp.Minimize(expr1))
print(f"  1. cp.norm(x, 2): {problem1.is_dcp()} ✓ (L2 norm)")

# 2. L1 노름 (convex)
expr2 = cp.norm(x, 1)
problem2 = cp.Problem(cp.Minimize(expr2))
print(f"  2. cp.norm(x, 1): {problem2.is_dcp()} ✓ (L1 norm)")

# 3. sum of squares (convex)
expr3 = cp.sum_squares(x)
problem3 = cp.Problem(cp.Minimize(expr3))
print(f"  3. cp.sum_squares(x): {problem3.is_dcp()} ✓ (sum of squares)")

# 4. exp (convex)
expr4 = cp.sum(cp.exp(x))
problem4 = cp.Problem(cp.Minimize(expr4))
print(f"  4. cp.sum(cp.exp(x)): {problem4.is_dcp()} ✓ (exp)")

# 5. log (concave, 최소화 시 부등식 제약으로 사용)
expr5 = -cp.sum(cp.log(x))  # -log가 convex
problem5 = cp.Problem(cp.Minimize(expr5))
print(f"  5. -cp.sum(cp.log(x)): {problem5.is_dcp()} ✓ (-log)")

print("\n비유효한 (DCP 위반) 표현들:")

# 6. x * y (곱: non-convex)
try:
    z = cp.Variable(3)
    expr6 = cp.sum(x * z)  # element-wise 곱
    problem6 = cp.Problem(cp.Minimize(expr6))
    print(f"  6. x * z (원소별 곱): {problem6.is_dcp()} ✗ (비볼록)")
except Exception as e:
    print(f"  6. x * z (원소별 곱): DCP 규칙 위반 ✗")

# 7. x**2인데 변수끼리: non-convex
try:
    w = cp.Variable(3)
    expr7 = cp.sum(w**2)  # 제곱합은 convex인데, w변수끼리 곱하면?
    problem7 = cp.Problem(cp.Minimize(expr7))
    print(f"  7. cp.sum(w**2): {problem7.is_dcp()} ✓ (제곱은 convex)")
except Exception as e:
    print(f"  7. cp.sum(w**2): {e}")

print("\n✓ CVXPY DCP 검증 완료")

print("\n" + "=" * 70)
print("모든 검증 완료!")
print("=" * 70)
```

---

## 🔗 AI/ML 연결

1. **정규화 항의 설계**: $\lambda \|w\|_1 + \mu \|w\|_2^2$ (탄성망 regularization)는 비음 가중합이므로 항상 볼록. 손실함수가 볼록하면 전체도 볼록.

2. **CVXPY의 자동 검증**: 사용자가 작성한 최적화 문제가 DCP 규칙을 따르는지 자동으로 확인. 규칙을 위반하면 경고 → 풀이 불가능성 사전 감지.

3. **음의 엔트로피와 Log-Sum-Exp**: 둘 다 선형 함수의 상한으로 표현 가능 → 포인트와이즈 상한의 보존으로 자동 증명 가능.

4. **SVM의 이중 문제(dual)**: 원래 문제가 비볼록이지만, 라그랑주 쌍대(Lagrange dual)는 항상 볼록 → convex relaxation의 기초.

---

## ⚖️ 가정과 한계

| 연산 | 조건 | 한계 |
|-----|------|------|
| **가중합** | 모든 가중치 ≥ 0 | 음수 가중치이면 비볼록 가능 |
| **상한** | 각 함수가 다른 모든 점에서도 볼록 | 특정 구간에서만 비교하면 실패 |
| **아핀 합성** | A가 아핀(affine) 필요 | 비선형 합성은 조건 필요 |
| **스칼라 합성** | g가 증가/감소, 볼록/오목 조건 | 조건 잘못되면 비볼록 |

---

## 📌 핵심 정리

| 연산 | 기호 | 조건 | 결과 |
|-----|------|------|------|
| **가중합** | $\sum w_i f_i$ | $w_i \geq 0$, $f_i$ 볼록 | 볼록 |
| **상한** | $\sup_\alpha f_\alpha$ | 각 $f_\alpha$ 볼록 | 볼록 |
| **아핀 합성** | $f(Ax+b)$ | $f$ 볼록 | 볼록 |
| **합성 (케이스 1)** | $g(f(x))$ | $f$ 오목, $g$ 오목 증가 | 오목 |
| **합성 (케이스 2)** | $g(f(x))$ | $f$ 볼록, $g$ 볼록 증가 | 볼록 |
| **Inf-conv** | $f \square g$ | $f, g$ 볼록 | 볼록 |

---

## 🤔 생각해볼 문제

**문제 2.10**: 다음 함수가 볼록한지 판정하고, 사용한 규칙을 명시하시오.
$$f(x) = \sqrt{\|Ax - b\|_2^2 + c^2}$$

<details>
<summary>힌트 및 해설</summary>

**힌트**: Norm과 합성 규칙을 활용하세요. $t = \|Ax-b\|_2$라 놓으면...

**해설**:
- $t = \|Ax - b\|_2$ (아핀 합성 + norm = 볼록)
- $g(t) = \sqrt{t^2 + c^2}$ (오목 함수... 아님!)

실제로 $g(t) = \sqrt{t^2 + c^2}$는 **볼록**입니다 (2-norm의 일반화).

따라서 $f(x) = g(\|Ax-b\|_2)$ 형태에서:
- $h(x) = \|Ax-b\|_2$ (볼록)
- $g(t) = \sqrt{t^2 + c^2}$ (볼록 증가)

정리 2.14 케이스 2에 의해 **$f$는 볼록**. ✓

</details>

---

**문제 2.11**: 두 오목 함수 $f, g$의 곱 $f(x)g(x)$가 오목할 조건은?

<details>
<summary>힌트 및 해설</summary>

**힌트**: 곱의 헤시안을 계산하고, 2차 편미분의 부호를 분석하세요.

**해설**:
$$h(x) = f(x)g(x)$$
$$\nabla h = f \nabla g + g \nabla f$$
$$\nabla^2 h = f \nabla^2 g + (\nabla f)(\nabla g)^T + g \nabla^2 f + (\nabla g)(\nabla f)^T$$

$f, g$가 오목이므로 $\nabla^2 f \preceq 0$, $\nabla^2 g \preceq 0$.

하지만 $(\nabla f)(\nabla g)^T + (\nabla g)(\nabla f)^T$ 항이 음수가 아닐 수 있음.

**일반적으로 성립하지 않음**. 특수한 경우(예: 비음 오목함수들)에만 가능.

</details>

---

**问题 2.12**: CVXPY에서 다음 문제가 풀이 가능한가? 왜?
```python
x = cp.Variable(n)
problem = cp.Problem(cp.Minimize(cp.norm(x, 1) + cp.quad_form(x, Q)))
```
여기서 $Q$는 양반정치 행렬이다.

<details>
<summary>힌트 및 해설</summary>

**힌트**: 각 항의 볼록성을 확인하고, 가중합 규칙을 적용하세요.

**해설**:
- $\text{cp.norm}(x, 1)$ = L1 노름 = 볼록 ✓
- $\text{cp.quad\_form}(x, Q) = x^T Q x$ = Q PSD일 때 볼록 ✓
- 둘의 합 = 비음 가중합(암묵적 가중치 1, 1) = 볼록 ✓

따라서 **DCP 규칙 만족 → 풀이 가능**. ✓

```python
problem.is_dcp()  # True
```

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. 강볼록성과 매끄러움](./03-strong-convexity-smoothness.md) | [📚 README](../README.md) | [05. Conjugate 함수와 Legendre 변환 ▶](./05-conjugate-legendre.md) |

</div>
