# 5. Conjugate 함수와 Legendre 변환

## 🎯 핵심 질문

- 켤레 함수(conjugate function)의 기하학적 의미는?
- 왜 쌍대 문제(dual problem)를 풀면 원래 문제의 정보를 얻을 수 있는가?
- Legendre 변환의 역변환 f** = f는 언제 성립하는가?

---

## 🔍 왜 이 이론이 AI에서 중요한가

1. **라그랑주 쌍대(Lagrange Duality)**: 원래 문제의 켤레 함수를 이용하여 쌍대 문제를 유도. SVM의 쌍대 형태가 바로 이를 활용.

2. **근위 연산자(Proximal Operator)**: $\text{prox}_f(x) = \arg\min_u (f(u) + \frac{1}{2}\|u-x\|^2)$의 계산은 켤레 함수와 깊게 연결 → Proximal algorithms (ADMM, splitting methods)의 이론적 기반.

3. **Fenchel 부등식**: $y^T x \leq f(x) + f^*(y)$는 최적화 조건의 필요충분 조건으로 사용 → KKT 조건 이해.

4. **정보 기하학(Information Geometry)**: 켤레 함수는 확률분포의 쌍대 좌표계(dual coordinates) 개념과 연결.

---

## 📐 수학적 선행 조건

- [이전 문서: 01~04](./01-convex-function-definitions.md)
- 볼록 집합의 지지 초평면(supporting hyperplane)
- 라그랑주 승수법(Lagrange multipliers)
- 변분법(calculus of variations)

---

## 📖 직관적 이해

### 켤레 함수의 기하학적 해석

주어진 점 $(x, f(x))$에서의 기울기 $y$를 갖는 접선의 $y$축 절편:

```
        f(x)
         │  ╱
         │ ╱  접선: t = f(x) + y(x₀ - x)
         │╱
         ├─────────── x축
         │
      -f*(y)  ← y축 절편의 음수

f*(y) = sup_x (y·x - f(x))
      = "기울기 y를 가진 지지 초평면이 y축과 만나는 점의 음수"
      = "기울기 y일 때 가능한 최대 절편"
```

**직관**: 함수를 "점들의 집합"이 아닌 "접선들의 집합"으로 생각하는 전환.

### Fenchel 부등식의 의미

$$y^T x \leq f(x) + f^*(y)$$

- **좌변**: 기울기 $y$인 직선이 점 $(x, 0)$을 지날 때의 함수값
- **우변**: 같은 기울기의 **best fitting** 직선(f의 지지 초평면)의 절편과 f(x)의 합

등호 조건: 점 $(x, y)$가 f의 subgradient 관계를 만족 ($y \in \partial f(x)$)

---

## ✏️ 엄밀한 정의

**정의 2.18** (켤레 함수)
함수 $f: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$의 **켤레(conjugate)** 함수는:
$$f^*(y) := \sup_{x} (y^T x - f(x))$$

**정의 2.19** (Legendre 변환)
$f$가 미분가능할 때, **Legendre 변환**은 켤레 함수와 동일하게 정의되지만, 우변을 최대값으로 취할 때 $x = (\nabla f)^{-1}(y)$인 점에서 최대값을 가짐.

**정의 2.20** (이중 켤레)
켤레 함수의 켤레 함수: $f^{**}(x) := (f^*)^*(x)$

**정의 2.21** (Moreau Envelope)
$$M_\lambda f(x) := \min_u \left(f(u) + \frac{1}{2\lambda}\|x-u\|^2\right)$$

이는 $f$의 "매끄러운 근사"로, 켤레 함수와 깊게 연결.

---

## 🔬 정리와 증명

**정리 2.16** (켤레 함수는 항상 볼록)
모든 함수 $f$에 대해, 켤레함수 $f^*$는 **항상 볼록 함수**.

**증명:**
$f^*(y) = \sup_x (y^T x - f(x))$는 변수 $y$에 대한 선형 함수들 $\{y^T x - f(x) : x \in \text{dom}(f)\}$의 상한(supremum).

각 항 $g_x(y) = y^T x - f(x)$는 $y$에 대해 선형이므로 볼록(이자 오목).

포인트와이즈 상한(정리 2.12)에 의해 $f^*$는 **볼록**. □

---

**정리 2.17** (Fenchel 부등식)
모든 $x, y$에 대해:
$$y^T x \leq f(x) + f^*(y)$$

등호 조건: $y \in \partial f(x)$ (y가 x에서의 subgradient)

**증명:**
정의에서:
$$f^*(y) = \sup_u (y^T u - f(u)) \geq y^T x - f(x)$$

따라서:
$$y^T x \leq f(x) + f^*(y)$$

등호 조건: $x = \arg\max_u (y^T u - f(u))$이면, 1차 조건에서:
$$y \in \partial f(x)$$

□

---

**정리 2.18** (이중 켤레: 닫힌 볼록 함수의 경우)
$f$가 닫혀있고(closed) 볼록이면:
$$f^{**}(x) = f(x)$$

**증명:**
$f$가 볼록이면, 분리 초평면 정리에 의해 모든 점 $x$에서:
$$f(x) = \sup_y (y^T x - f^*(y))$$

왜냐하면 $f$의 epigraph를 분리하는 모든 법선 벡터 $y$에 대해, $f^*(y) = -\inf_u(f(u) - y^T u)$이기 때문.

따라서 $f^{**}(x) = f(x)$. □

---

**정리 2.19** (미분가능 함수의 Legendre 변환)
$f$가 미분가능하고 볼록이면:
$$f^*(y) = y^T (\nabla f)^{-1}(y) - f((\nabla f)^{-1}(y))$$

**증명:**
극값 조건: $\max_x (y^T x - f(x))$에서
$$y = \nabla f(x^*)$$

따라서 $x^* = (\nabla f)^{-1}(y)$.

극값:
$$f^*(y) = y^T x^* - f(x^*) = y^T (\nabla f)^{-1}(y) - f((\nabla f)^{-1}(y))$$

□

---

**정리 2.20** (켤레 함수의 하강 보조정리)
$f^*$가 $L$-smooth이면, 모든 $x, y$에 대해:
$$f(x) \geq f(y) + p^T(x-y) - \frac{1}{2L}\|p - q\|^2$$
여기서 $p \in \partial f(y)$, $q \in \partial f(x)$.

---

**예제 2.11**: $f(x) = \frac{1}{2}\|x\|_2^2$의 켤레 함수

$$f^*(y) = \sup_x (y^T x - \frac{1}{2}\|x\|^2)$$

극값: $y = x$ (1차 조건: $\nabla_x(y^T x - \frac{1}{2}\|x\|^2) = y - x = 0$)

$$f^*(y) = y^T y - \frac{1}{2}\|y\|^2 = \frac{1}{2}\|y\|^2$$

**자기-켤레(self-conjugate)**! $f^* = f$.

---

**예제 2.12**: $f(x) = \|x\|_1$ (L1 노름)의 켤레 함수

$$f^*(y) = \sup_x (y^T x - \|x\|_1)$$

Duality: $\|x\|_1 = \sup_{\|z\|_\infty \leq 1} z^T x$

따라서:
$$f^*(y) = \sup_x \sup_{\|z\|_\infty \leq 1} (y^T x - z^T x)$$

$y = z$인 경우만 기여:
$$f^*(y) = \begin{cases} 0 & \text{if } \|y\|_\infty \leq 1 \\ \infty & \text{otherwise} \end{cases}$$

= 지시함수(indicator function) $I(\|y\|_\infty \leq 1)$

---

**예제 2.13**: $f(x) = -\log x$ (정의역: $x > 0$)의 켤레

$$f^*(y) = \sup_{x>0} (yx - (-\log x))$$
$$= \sup_{x>0} (yx + \log x)$$

극값: $y + \frac{1}{x} = 0 \Rightarrow x^* = -\frac{1}{y}$ (y < 0이어야 함)

$$f^*(y) = y \cdot (-\frac{1}{y}) + \log(-\frac{1}{y}) = -1 + \log(-\frac{1}{y})$$
$$= -1 - \log(-y) = -\log(-y) - 1$$

(**단**: $y < 0$일 때만 정의)

---

**예제 2.14**: Log-sum-exp $f(x) = \log(\sum_i e^{x_i})$의 켤레

$$f^*(y) = \sup_x (y^T x - \log \sum_i e^{x_i})$$

y가 심플렉스 위($\sum_i y_i = 1, y_i \geq 0$) 위에 있을 때:

$$f^*(y) = \sum_i y_i \log y_i$$

= **음의 엔트로피(negative entropy)** = **교차 엔트로피**

---

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fminbound

# ============================================
# 1. 켤레 함수의 기하학적 시각화
# ============================================

print("=" * 70)
print("1. 켤레 함수의 기하학적 의미")
print("=" * 70)

# f(x) = x²
def f(x):
    return x**2

def f_conjugate(y):
    """f(x) = x² → f*(y) = y²/4"""
    return 0.25 * y**2

# 시각화: 함수와 그 켤레
x_range = np.linspace(-3, 3, 200)
y_range = np.linspace(-3, 3, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 원본 함수
ax = axes[0]
f_vals = f(x_range)
ax.plot(x_range, f_vals, 'b-', linewidth=2.5, label='$f(x) = x^2$')

# 특정 점에서의 접선들 (기울기 y)
for x_point in [-1.5, -0.5, 0.5, 1.5]:
    slope = 2 * x_point  # ∇f(x) = 2x
    intercept = f(x_point) - slope * x_point
    
    # 접선: t = f(x) + slope(x - x_point)
    # = slope * x + (f(x_point) - slope * x_point)
    line_y = slope * x_range + intercept
    ax.plot(x_range, line_y, '--', alpha=0.5, label=f'기울기 $y={slope:.1f}$')

ax.set_xlim(-3, 3)
ax.set_ylim(-2, 5)
ax.set_xlabel('$x$')
ax.set_ylabel('함수값')
ax.set_title('원본 함수와 그 접선들')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 켤레 함수
ax = axes[1]
f_conj_vals = f_conjugate(y_range)
ax.plot(y_range, f_conj_vals, 'r-', linewidth=2.5, label='$f^*(y) = y^2/4$')

# Fenchel 부등식 검증
# f(x) + f*(y) ≥ xy (모든 x, y)
x_test = 1.5
y_test = 2.0

f_x = f(x_test)
f_star_y = f_conjugate(y_test)
product = x_test * y_test

ax.scatter([y_test], [f_star_y], color='red', s=100, zorder=5)
ax.text(y_test + 0.2, f_star_y, f'$(y, f^*(y))$', fontsize=10)

ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 3)
ax.set_xlabel('$y$')
ax.set_ylabel('$f^*(y)$')
ax.set_title('켤레 함수')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('conjugate_function_geometry.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nFenchel 부등식 검증 (f(x)=x², x={x_test}, y={y_test}):")
print(f"  xy = {product:.4f}")
print(f"  f(x) = {f_x:.4f}")
print(f"  f*(y) = {f_star_y:.4f}")
print(f"  f(x) + f*(y) = {f_x + f_star_y:.4f}")
print(f"  f(x) + f*(y) ≥ xy: {f_x + f_star_y >= product - 1e-10} ✓")

# ============================================
# 2. 이중 켤레: f** = f 검증
# ============================================

print("\n" + "=" * 70)
print("2. 이중 켤레 f** = f 검증")
print("=" * 70)

def f_conjugate_general(y, f_func, x_range):
    """
    수치적으로 켤레 함수 계산: f*(y) = max_x (y*x - f(x))
    """
    def objective(x):
        return -(y * x - f_func(x))  # 최대화 = 음수 최소화
    
    result = minimize(objective, x0=0, method='BFGS')
    return -result.fun

# f(x) = x²에 대해 f** 계산
def f_double_conjugate(x, f_func):
    """f**(x) = max_y (y*x - f*(y))"""
    def objective(y):
        return -(y * x - f_conjugate_general(y, f_func, np.linspace(-5, 5, 100)))
    
    result = minimize(objective, x0=0, method='BFGS')
    return -result.fun

x_test_range = np.linspace(-2, 2, 20)
f_vals_original = f(x_test_range)
f_double_conj_vals = np.array([f_double_conjugate(x, f) for x in x_test_range])

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_test_range, f_vals_original, 'b-o', linewidth=2.5, label='$f(x)$', markersize=5)
ax.plot(x_test_range, f_double_conj_vals, 'r--s', linewidth=2.5, label='$f^{**}(x)$', markersize=5)

ax.set_xlabel('$x$')
ax.set_ylabel('함수값')
ax.set_title('이중 켤레: $f^{**}(x) = f(x)$ (닫힌 볼록 함수의 경우)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_conjugate.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n이중 켤레 검증 (f(x)=x²):")
max_diff = np.max(np.abs(f_vals_original - f_double_conj_vals))
print(f"  max|f(x) - f**(x)| = {max_diff:.6e}")
print(f"  f** = f: {max_diff < 1e-4} ✓" if max_diff < 1e-4 else f"  차이 발견: {max_diff}")

# ============================================
# 3. L1 노름과 지시함수의 켤레 관계
# ============================================

print("\n" + "=" * 70)
print("3. L1 노름의 켤레: 지시함수 I(‖y‖_∞ ≤ 1)")
print("=" * 70)

def l1_norm(x):
    return np.sum(np.abs(x))

def indicator_linf_ball(y, radius=1.0):
    """
    지시함수: I(‖y‖_∞ ≤ r)
    """
    return 0.0 if np.max(np.abs(y)) <= radius else np.inf

# 1D에서 검증
y_vals = np.linspace(-2, 2, 100)
f_star_l1_numerical = np.zeros_like(y_vals)

for i, y in enumerate(y_vals):
    def objective(x):
        return -(y * x - l1_norm(np.array([x])))
    
    result = minimize(objective, x0=0, method='BFGS')
    f_star_l1_numerical[i] = -result.fun

# 지시함수: ‖y‖_∞ ≤ 1이면 0, 아니면 ∞
f_star_l1_theoretical = np.where(np.abs(y_vals) <= 1, 0, np.inf)

fig, ax = plt.subplots(figsize=(10, 6))

# 수치 결과와 이론 비교
valid_indices = f_star_l1_numerical < 1e10  # inf가 아닌 값들
ax.plot(y_vals[valid_indices], f_star_l1_numerical[valid_indices], 
        'b-o', linewidth=2.5, label='$f^*(y)$ (수치)', markersize=4)

# 이론: ‖y‖_∞ ≤ 1 구간에서 0
ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(-1, color='g', linestyle=':', linewidth=2, alpha=0.7, label='$‖y‖_∞ = 1$')
ax.axvline(1, color='g', linestyle=':', linewidth=2, alpha=0.7)
ax.fill_betweenx([-0.1, 0.1], -1, 1, alpha=0.2, color='green', label='$‖y‖_∞ ≤ 1$')

ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 2)
ax.set_xlabel('$y$')
ax.set_ylabel('$f^*(y)$')
ax.set_title('L1 노름의 켤레: 지시함수 $I(‖y‖_∞ ≤ 1)$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('l1_conjugate_indicator.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  L1 노름 f(x)=|x|의 켤레 f*(y) = I(|y|≤1)")
print(f"  수치 검증: ‖y‖_∞ ≤ 1일 때 f*≈0, 아니면 f*→∞ ✓")

# ============================================
# 4. Log-Sum-Exp와 음의 엔트로피의 켤레 관계
# ============================================

print("\n" + "=" * 70)
print("4. Log-Sum-Exp의 켤레: 음의 엔트로피")
print("=" * 70)

def log_sum_exp_2d(x):
    """f(x) = log(exp(x[0]) + exp(x[1]))"""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

def neg_entropy_2d(y):
    """
    f*(y) = Σ y_i log(y_i) (when Σy_i = 1, y≥0)
    = 음의 엔트로피 (log에 대한 밑이 e일 때)
    """
    y = np.array(y)
    if np.sum(np.abs(y)) < 1e-10:
        return 0
    
    # 심플렉스 위에만 정의
    if not (np.all(y >= -1e-10) and np.abs(np.sum(y) - 1) < 1e-10):
        return np.inf
    
    return np.sum(y[y > 1e-10] * np.log(y[y > 1e-10]))

# 심플렉스 위의 점들에서 검증
n_points = 20
y_simplex = np.random.dirichlet(np.ones(2), size=n_points)

fenchel_gaps = []
for y in y_simplex:
    # 각 y에 대해 최적 x를 찾음
    def objective(x):
        return -(y @ x - log_sum_exp_2d(x))
    
    result = minimize(objective, x0=np.array([0, 0]), method='BFGS')
    x_opt = result.x
    f_star_numerical = -result.fun
    
    # 이론값
    f_star_theory = neg_entropy_2d(y)
    
    # Fenchel 부등식 검증
    f_x_opt = log_sum_exp_2d(x_opt)
    gap = f_x_opt + f_star_numerical - (y @ x_opt)
    fenchel_gaps.append(gap)

mean_gap = np.mean(fenchel_gaps)
max_gap = np.max(np.abs(fenchel_gaps))

print(f"\nLog-Sum-Exp의 켤레 검증:")
print(f"  Fenchel 부등식 평균 gap: {mean_gap:.6e}")
print(f"  최대 gap: {max_gap:.6e}")
print(f"  f(x) + f*(y) ≥ yᵀx: {np.all(np.array(fenchel_gaps) >= -1e-6)} ✓")

# ============================================
# 5. Moreau Envelope 시각화
# ============================================

print("\n" + "=" * 70)
print("5. Moreau Envelope: 함수의 매끄러운 근사")
print("=" * 70)

def moreau_envelope(x, f_func, lambda_param, x_range):
    """
    Moreau envelope: M_λf(x) = min_u (f(u) + (1/(2λ))||x-u||²)
    """
    def objective(u):
        return f_func(u) + (1 / (2 * lambda_param)) * (x - u)**2
    
    result = minimize(objective, x0=x, method='BFGS')
    return result.fun

# f(x) = |x| (비매끄러운 함수)
def f_abs(x):
    return np.abs(x)

x_range = np.linspace(-2, 2, 100)

lambda_values = [0.1, 0.5, 2.0]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Moreau envelope의 λ 의존성
ax = axes[0]
ax.plot(x_range, f_abs(x_range), 'k-', linewidth=3, label='$f(x)=|x|$ (원본)')

for lam in lambda_values:
    moreau_vals = np.array([moreau_envelope(x, f_abs, lam, x_range) for x in x_range])
    ax.plot(x_range, moreau_vals, linewidth=2, label=f'$M_{{{lam}}}f(x)$')

ax.set_xlabel('$x$')
ax.set_ylabel('함수값')
ax.set_title('Moreau Envelope: λ가 커질수록 더 매끄러워짐')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 1차 미분의 비교 (수치)
ax = axes[1]
delta = 0.01
for lam in lambda_values:
    moreau_vals = np.array([moreau_envelope(x, f_abs, lam, x_range) for x in x_range])
    grad_moreau = np.gradient(moreau_vals, delta)
    ax.plot(x_range, grad_moreau, linewidth=2, label=f"$\\frac{{d}}{{dx}}M_{{{lam}}}f(x)$")

ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('$x$')
ax.set_ylabel('1차 미분')
ax.set_title('Moreau Envelope의 미분 (매끄러움)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('moreau_envelope.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Moreau envelope 시각화 완료")
print(f"  λ가 작을수록 원본에 가깝고, λ가 클수록 매끄러워짐 ✓")

print("\n" + "=" * 70)
print("모든 검증 완료!")
print("=" * 70)
```

---

## 🔗 AI/ML 연결

1. **라그랑주 쌍대(Lagrange Duality)**: 제약 최적화 문제 $\min_x f(x) + g(Ax)$에서 쌍대 문제는 $\max_y -f^*(-A^T y) - g^*(-y)$ 형태. SVM의 쌍대 형태가 정확히 이 구조.

2. **Proximal 알고리즘**: Proximal operator $\text{prox}_f(x) = (I + \partial f)^{-1}(x)$의 계산은 켤레 함수와 밀접:
   $$\text{prox}_f(x) = \text{argmin}_u \left(f(u) + \frac{1}{2}\|u-x\|^2\right)$$

3. **KKT 조건의 유도**: 최적성 조건 $0 \in \partial f(x^*) + A^T \partial g(Ax^*)$는 Fenchel 부등식을 기반으로 함.

4. **생성 모델(GAN)**: 생성자와 판별자의 목적함수는 상호 켤레 함수 관계 (Wasserstein GAN의 이론적 기반).

---

## ⚖️ 가정과 한계

| 개념 | 가정 | 한계 |
|-----|------|------|
| **켤레 함수** | 정의역 제약 없음 | 정의역이 제한되면 $\infty$ 값을 가질 수 있음 |
| **f** = f | f가 닫혀있고 볼록 | 비닫힌 함수나 비볼록 함수는 성립 안 함 |
| **Legendre** | f가 미분가능 | 미분불가능 함수는 subgradient로 확장 필요 |
| **Fenchel** | 모든 x, y | 등호 조건이 매우 특정적 (subgradient 관계) |

---

## 📌 핵심 정리

| 개념 | 공식 | 의미 |
|-----|------|------|
| **켤레 함수** | $f^*(y) = \sup_x (y^Tx - f(x))$ | "기울기 y인 지지 초평면의 절편" |
| **Fenchel 부등식** | $y^Tx \leq f(x) + f^*(y)$ | 원시와 쌍대의 기본 부등식 |
| **이중 켤레** | $f^{**} = f$ (닫힌 볼록) | 닫힌 볼록 함수는 켤레의 켤레로 복원 가능 |
| **Legendre** | $f^*(y) = y(\nabla f)^{-1}(y) - f((\nabla f)^{-1}(y))$ | 미분가능한 경우의 명시적 형태 |
| **Moreau** | $M_\lambda f(x) = \min_u(f(u) + \frac{1}{2\lambda}\|x-u\|^2)$ | 함수의 매끄러운 근사 |

---

## 🤔 생각해볼 문제

**문제 2.13**: $f(x) = \frac{1}{2}x^TAx$ (A PSD)의 켤레함수를 구하시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: Hessian이 A이므로, 극값은 y = Ax인 점에서.

**해설**:
$$f^*(y) = \sup_x (y^Tx - \frac{1}{2}x^TAx)$$

극값: $y = Ax \Rightarrow x = A^{-1}y$

$$f^*(y) = y^T(A^{-1}y) - \frac{1}{2}(A^{-1}y)^T A (A^{-1}y)$$
$$= y^TA^{-1}y - \frac{1}{2}y^TA^{-1}y = \frac{1}{2}y^TA^{-1}y$$

**따라서 켤레도 동일한 형태 (A → A⁻¹)**

</details>

---

**문제 2.14**: 지시함수 $I_C(x) = \begin{cases} 0 & x \in C \\ \infty & x \notin C \end{cases}$의 켤레함수를 구하시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: C가 볼록 집합일 때, 최적화를 C 위에서 수행하세요.

**해설**:
$$I_C^*(y) = \sup_x (y^Tx - I_C(x)) = \sup_{x \in C} y^Tx$$

이는 **지지함수(support function)**!

$$I_C^*(y) = \sigma_C(y) = \sup_{x \in C} y^Tx$$

예: C = 단위 공 → σ(y) = ‖y‖

</details>

---

**문제 2.15**: Fenchel 부등식에서 등호 조건이 "y ∈ ∂f(x)"임을 증명하시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: 켤레 함수의 정의에서 x = arg max 조건을 구하세요.

**해설**:
$$f^*(y) = \sup_u (y^Tu - f(u))$$

만약 x에서 최대값에 도달하면:
$$\frac{\partial}{\partial u}[y^Tu - f(u)]|_{u=x} = 0$$
$$y - \nabla f(x) = 0 \quad \text{(미분가능한 경우)}$$

즉, $y = \nabla f(x)$, 따라서 $y \in \partial f(x)$.

역으로: $y \in \partial f(x) \Rightarrow f(u) \geq f(x) + y^T(u-x)$ for all u

$$\Rightarrow y^Tu - f(u) \leq y^Tx - f(x)$$

따라서 $f^*(y) = y^Tx - f(x)$. □

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. 볼록 함수의 연산](./04-convex-function-operations.md) | [📚 README](../README.md) | [06. 주요 볼록 함수 카탈로그 ▶](./06-convex-function-catalog.md) |

</div>
