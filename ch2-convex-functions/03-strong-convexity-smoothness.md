# 3. 강볼록성(Strong Convexity)과 매끄러움(Smoothness)

## 🎯 핵심 질문

- 강볼록성이란 "더 빨리 곡하는 볼록 함수"인가?
- L-smooth는 헤시안의 상한이 L·I라는 뜻인가?
- 조건수(condition number) κ = L/μ가 수렴 속도를 왜 결정하는가?

---

## 🔍 왜 이 이론이 AI에서 중요한가

1. **수렴 속도 분석**: GD의 수렴은 $(1 - \mu/L)^k$에 따르고, κ가 클수록 느림. 이는 머신러닝 최적화에서 가장 중요한 성능 지표.

2. **전처리(Preconditioning)의 필요성**: 특성(feature)의 스케일이 다르면 Hessian의 고유값 범위가 커져 κ가 악화됨. 정규화/표준화가 학습을 가속하는 이유.

3. **Adam, RMSprop 같은 적응형 방법의 동기**: 각 매개변수의 "국소 곡률"을 추정하여 스텝 크기를 조정 → 실질적으로 조건수를 개선.

4. **신경망 학습 곡선**: 초기에는 비강볼록 (많은 국소 최솟값), 최적 근처에서 강볼록 근사 → Hessian eigenvalue 분석으로 설명 가능.

---

## 📐 수학적 선행 조건

- 벡터 노름, 행렬 노름(spectral norm)
- [이전 문서: 02. 일계·이계 조건](./02-first-second-order-conditions.md)
- Hessian의 고유값(eigenvalue)과 조건수
- Taylor 전개 (2차)

---

## 📖 직관적 이해

### 강볼록성의 기하학적 의미

```
    f(x)
       ↑
       │     실제 함수
       │    /│
       │   / │  ← 이차 하한(quadratic lower bound)
       │  /  │    f(x) + ∇f(x)ᵀ(y-x) + (μ/2)‖y-x‖²
       │ *────*
       └──────────→ x
       
일반 볼록:  f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
강볼록:     f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)‖y-x‖²

= "함수가 포물선처럼 빠르게 곡한다"
```

### L-smooth의 기하학적 의미

```
    f(x)
       ↑
       │     ← 상한(quadratic upper bound)
       │      f(x) + ∇f(x)ᵀ(y-x) + (L/2)‖y-x‖²
       │     /│
       │    / │  실제 함수
       │   /  │
       │  /   │
       │ *────*
       └──────────→ x
       
= "함수의 그래디언트가 L-Lipschitz"
= "곡률이 L 이상으로 빠르지 않다"
```

### 조건수의 역할

등고선의 "타원 모양"을 결정:
- κ = 1: 원형 (최적 → GD가 한 스텝에 수렴)
- κ = 10: 약간 타원형 (수렴이 느림)
- κ = 1000: 극도로 납작한 타원 (GD가 지그재그로 진행, 매우 느림)

---

## ✏️ 엄밀한 정의

**정의 2.9** (μ-강볼록성)
함수 $f$가 $\mu > 0$에 대해 **μ-strongly convex** $\Leftrightarrow$ 모든 $x, y \in \text{dom}(f)$에 대해:
$$f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$$

**정의 2.10** (L-smoothness, L-smooth)
함수 $f$가 **L-smooth** $\Leftrightarrow$ 모든 $x, y$에 대해:
$$f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2$$

동치 조건: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$ (Lipschitz gradient)

**정의 2.11** (조건수)
함수 $f$가 μ-강볼록이고 L-smooth일 때, **조건수(condition number)**는:
$$\kappa = \frac{L}{\mu}$$

**정의 2.12** (2차 조건으로의 특성화)
- μ-강볼록 ↔ $\nabla^2 f(x) \succeq \mu I$ (모든 $x$에서)
- L-smooth ↔ $\nabla^2 f(x) \preceq L I$ (모든 $x$에서)

---

## 🔬 정리와 증명

**정리 2.7** (강볼록성 ↔ 2차 조건)
두 번 미분가능 함수 $f$가 μ-강볼록 $\Leftrightarrow$ 모든 $x$에서 $\nabla^2 f(x) \succeq \mu I$

**증명:**
($\Rightarrow$) 강볼록성 정의에서:
$$f(x+h) \geq f(x) + \nabla f(x)^T h + \frac{\mu}{2}\|h\|^2$$

Taylor: $f(x+h) = f(x) + \nabla f(x)^T h + \frac{1}{2}h^T\nabla^2 f(\xi)h$ ($\xi \in [x, x+h]$)

$h \to 0$일 때:
$$\frac{1}{2}h^T\nabla^2 f(x)h \geq \frac{\mu}{2}\|h\|^2 = \frac{\mu}{2}h^Th$$

따라서 $h^T[\nabla^2 f(x) - \mu I]h \geq 0$ for all $h$, 즉 $\nabla^2 f(x) \succeq \mu I$. □

($\Leftarrow$) $\nabla^2 f(x) \succeq \mu I$라고 가정. Taylor 중간값 정리:
$$f(y) = f(x) + \nabla f(x)^T(y-x) + \frac{1}{2}(y-x)^T\nabla^2 f(\xi)(y-x)$$

$$\geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$$ □

---

**정리 2.8** (L-smoothness ↔ Descent Lemma)
$f$가 L-smooth $\Leftrightarrow$ 모든 $x, y$에 대해:
$$f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2$$

**증명 스케치:**
($\Rightarrow$) Hessian이 $\preceq LI$이면, Taylor:
$$f(y) = f(x) + \nabla f(x)^T(y-x) + \int_0^1 (1-t)(y-x)^T\nabla^2(x+t(y-x))(y-x)dt$$
$$\leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2$$

($\Leftarrow$) 부등식이 성립 → Lipschitz gradient → Hessian $\preceq LI$ □

---

**정리 2.9** (강볼록 함수의 유일 최솟값)
$f$가 μ-강볼록이면, 유일한 최솟값 $x^*$가 존재하고:
$$\frac{\mu}{2}\|x - x^*\|^2 \leq f(x) - f(x^*)$$

**증명:**
$x^*$에서 $\nabla f(x^*) = 0$ (최솟값의 필요조건).

강볼록성:
$$f(x^*) \geq f(x) + \nabla f(x)^T(x^* - x) + \frac{\mu}{2}\|x^* - x\|^2$$

$\nabla f(x^*) = 0$이므로:
$$f(x^*) \geq f(x) - \nabla f(x)^T(x - x^*) + \frac{\mu}{2}\|x - x^*\|^2$$

한편, 임의의 $x$에 대해 강볼록성:
$$f(x) \geq f(x^*) + \nabla f(x^*)^T(x - x^*) + \frac{\mu}{2}\|x - x^*\|^2 = f(x^*) + \frac{\mu}{2}\|x-x^*\|^2$$

□

---

**정리 2.10** (GD의 수렴률, μ-강볼록 + L-smooth)
$f$가 μ-강볼록이고 L-smooth일 때, 스텝 크기 $\alpha = 1/L$인 GD는:
$$f(x^{(k)}) - f(x^*) \leq \left(1 - \frac{\mu}{L}\right)^k [f(x^{(0)}) - f(x^*)]$$

**증명:**
Descent lemma ($\alpha = 1/L$):
$$f(x^+ ) = f(x - \frac{1}{L}\nabla f(x)) \leq f(x) - \frac{1}{2L}\|\nabla f(x)\|^2$$

강볼록성: $\|\nabla f(x)\|^2 \geq 2\mu(f(x) - f(x^*))$

따라서:
$$f(x^+) \leq f(x) - \frac{\mu}{L}(f(x) - f(x^*)) = \left(1 - \frac{\mu}{L}\right)(f(x) - f(x^*))$$

귀납법으로 $f(x^{(k)}) - f(x^*) \leq (1 - \kappa^{-1})^k[f(x^{(0)}) - f(x^*)]$. □

---

**예제 2.6**: Ridge 회귀의 조건수 분석

Ridge: $f(w) = \|Xw - y\|^2 + \lambda\|w\|^2$

$$\nabla^2 f(w) = 2X^TX + 2\lambda I$$

고유값: $\lambda_i(2X^TX + 2\lambda I) = 2\lambda_i(X^TX) + 2\lambda$

- 최소: $2\lambda$ (λ = 0인 경우)
- 최대: $2\lambda_{\max}(X^TX) + 2\lambda$

$$\mu = 2\lambda, \quad L = 2\lambda_{\max}(X^TX) + 2\lambda$$

$$\kappa = \frac{2\lambda_{\max}(X^TX) + 2\lambda}{2\lambda} = \frac{\lambda_{\max}(X^TX)}{\lambda} + 1$$

결론: **λ를 크게 하면 κ가 작아져 수렴이 빨라짐** (대신 정확도 손상)

---

**예제 2.7**: Quadratic function $f(x) = \frac{1}{2}x^TAx - b^Tx$ ($A \succ 0$)

$$\nabla^2 f(x) = A$$

- μ = λ_min(A), L = λ_max(A)
- κ = λ_max(A) / λ_min(A) = condition number of A

κ가 크면 A의 고유벡터 방향에 따라 곡률이 극도로 다름
→ GD가 "일부 방향은 빠르게, 일부 방향은 매우 느리게" 수렴

---

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Ellipse

# ============================================
# 1. 강볼록성과 L-smoothness 검증
# ============================================

def verify_strong_convexity_and_smoothness(f, grad_f, x, y, mu, L):
    """
    강볼록성: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)‖y-x‖²
    L-smoothness: f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)‖y-x‖²
    """
    f_x = f(x)
    f_y = f(y)
    grad_x = grad_f(x)
    diff = y - x
    
    taylor = f_x + np.dot(grad_x, diff)
    quadratic = (0.5) * np.linalg.norm(diff)**2
    
    # 강볼록 검증
    lower_bound = taylor + mu * quadratic
    strong_convex_gap = f_y - lower_bound
    
    # L-smoothness 검증
    upper_bound = taylor + L * quadratic
    smoothness_gap = upper_bound - f_y
    
    return strong_convex_gap, smoothness_gap, (lower_bound, upper_bound)

print("=" * 70)
print("1. 강볼록성과 L-smoothness 검증")
print("=" * 70)

# 테스트 함수: Ridge 회귀 스타일
A = np.array([[3.0, 0.5], [0.5, 1.0]])  # PSD 행렬
b = np.array([1.0, -0.5])

def f_ridge(x, A, b, lam):
    return 0.5 * (x @ A @ x) + lam * (x @ x) - b @ x

def grad_ridge(x, A, b, lam):
    return A @ x + 2 * lam * x - b

# 강볼록성 상수: μ = λ_min(A + 2λI)
# L-smoothness 상수: L = λ_max(A + 2λI)
lam_reg = 0.5

A_ridge = A + 2 * lam_reg * np.eye(2)
eigenvalues = np.linalg.eigvalsh(A_ridge)
mu = eigenvalues[0]
L = eigenvalues[1]
kappa = L / mu

print(f"\nRidge 회귀: f(x) = (1/2)xᵀAx + λ‖x‖² - bᵀx, λ={lam_reg}")
print(f"  A의 고유값: {np.linalg.eigvalsh(A)}")
print(f"  μ (강볼록성): {mu:.4f}")
print(f"  L (smoothness): {L:.4f}")
print(f"  κ = L/μ: {kappa:.4f}")

# 임의의 점 쌍에서 검증
test_pairs = [
    (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
    (np.array([0.5, -0.3]), np.array([1.2, 0.8])),
    (np.array([-1.0, 0.0]), np.array([2.0, -1.5])),
]

f = lambda x: f_ridge(x, A, b, lam_reg)
grad = lambda x: grad_ridge(x, A, b, lam_reg)

all_passed = True
for i, (x, y) in enumerate(test_pairs):
    sc_gap, sm_gap, bounds = verify_strong_convexity_and_smoothness(
        f, grad, x, y, mu, L
    )
    
    passed = (sc_gap >= -1e-10) and (sm_gap >= -1e-10)
    all_passed = all_passed and passed
    
    status = "✓" if passed else "✗"
    print(f"\n  쌍 {i+1}: {status}")
    print(f"    강볼록성 gap: {sc_gap:.6e} (≥ 0)")
    print(f"    smoothness gap: {sm_gap:.6e} (≥ 0)")

print(f"\n  모든 검증 통과: {all_passed}")

# ============================================
# 2. 조건수에 따른 등고선 형태 비교
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

def quadratic_2d(x, y, A):
    """f(x,y) = [x, y]ᵀ A [x, y] / 2"""
    return 0.5 * (A[0,0]*x**2 + 2*A[0,1]*x*y + A[1,1]*y**2)

# 서로 다른 조건수의 행렬들
test_cases = [
    ("κ=1 (원형)", np.eye(2)),
    ("κ=10 (타원형)", np.array([[10, 0], [0, 1]])),
    ("κ=100 (납작한 타원)", np.array([[100, 0], [0, 1]])),
]

x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_range, y_range)

for idx, (ax, (title, A_test)) in enumerate(zip(axes, test_cases)):
    kappa_test = A_test[0, 0] / A_test[1, 1]
    Z = quadratic_2d(X, Y, A_test)
    
    # 등고선
    contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 최솟값 표시
    ax.scatter([0], [0], color='red', s=100, zorder=5, marker='*',
               edgecolors='black', linewidths=2)
    
    # 고유벡터 그리기
    eigenvalues, eigenvectors = np.linalg.eigh(A_test)
    for i, (eig_val, eig_vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        scale = np.sqrt(eig_val)
        ax.arrow(0, 0, eig_vec[0], eig_vec[1], 
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n$\\kappa$={kappa_test:.0f}")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

plt.tight_layout()
plt.savefig('condition_number_ellipses.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ 등고선 시각화 완료")

# ============================================
# 3. 조건수에 따른 GD 수렴 속도
# ============================================

print("\n" + "=" * 70)
print("2. 조건수에 따른 경사하강법(GD) 수렴 속도")
print("=" * 70)

def gradient_descent(f, grad_f, x0, alpha, num_iterations=100):
    """경사하강법 구현"""
    x = x0.copy()
    trajectory = [x.copy()]
    
    for _ in range(num_iterations):
        g = grad_f(x)
        x = x - alpha * g
        trajectory.append(x.copy())
    
    return np.array(trajectory)

# 여러 조건수에서 GD 실행
kappa_values = [1, 10, 100]
trajectories = {}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for kappa in kappa_values:
    # A를 조건수에 맞춰 구성: λ_min=1, λ_max=κ
    A_test = np.array([[kappa, 0], [0, 1.0]])
    
    f_test = lambda x: 0.5 * (x @ A_test @ x)
    grad_test = lambda x: A_test @ x
    
    # L, μ 계산
    L_test = kappa
    mu_test = 1.0
    
    # 최적 스텝 크기
    alpha_opt = 2 / (L_test + mu_test)
    
    # GD 실행
    x0 = np.array([1.0, 1.0])
    traj = gradient_descent(f_test, grad_test, x0, alpha_opt, num_iterations=50)
    trajectories[kappa] = traj
    
    # 함수값 추적
    f_vals = np.array([f_test(x) for x in traj])
    
    # 이론적 수렴률
    rho = 1 - mu_test / L_test  # (1 - μ/L)
    theoretical = f_test(x0) * (rho ** np.arange(len(f_vals)))
    
    # 시각화 1: 함수값
    axes[0].semilogy(f_vals, 'o-', label=f'$\\kappa$={kappa} (실제)', markersize=4)
    axes[0].semilogy(theoretical, '--', label=f'$\\kappa$={kappa} (이론)', alpha=0.7)
    
    # 시각화 2: 2D 궤적
    x_range = np.linspace(-1.2, 1.2, 100)
    y_range = np.linspace(-1.2, 1.2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = quadratic_2d(X, Y, A_test)
    
    # 첫 번째 등고선
    if kappa == 1:
        ax = axes[1]
        levels = np.logspace(-2, 1, 20)
        ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        
        for k, (kappa_inner, traj_inner) in enumerate(trajectories.items()):
            ax.plot(traj_inner[:20, 0], traj_inner[:20, 1], 'o-', 
                   label=f'$\\kappa$={kappa_inner}', markersize=3, alpha=0.8)
    
axes[0].set_xlabel('반복 횟수')
axes[0].set_ylabel('$f(x^{(k)}) - f(x^*)$')
axes[0].set_title('수렴 속도: $\\rho = 1 - \\mu/L = (\\kappa-1)/(\\kappa+1)$')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlim(-1.2, 1.2)
axes[1].set_ylim(-1.2, 1.2)
axes[1].set_aspect('equal')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title('GD 궤적: κ가 크면 지그재그')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gd_convergence_rates.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ 수렴 속도 시각화 완료")

# ============================================
# 3. 강볼록성 검증: 포물선 하한
# ============================================

print("\n" + "=" * 70)
print("3. 강볼록성의 포물선 하한 시각화")
print("=" * 70)

x_1d = np.linspace(-2, 2, 100)
x_opt = 0.0

# 여러 μ 값에서
mu_values = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (ax, mu_val) in enumerate(zip(axes, [mu_values[0], mu_values[2]])):
    # f(x) = (1/2)x² (표준 이차형식, 고유값=1 = L=μ)
    # 하지만 더 강볼록하게: f(x) = (1/2)(1+μ)x²
    
    f_vals = 0.5 * (1 + mu_val) * x_1d**2
    
    # x=1에서의 접선과 강볼록 하한
    x_ref = 1.0
    f_x_ref = 0.5 * (1 + mu_val) * x_ref**2
    grad_x_ref = (1 + mu_val) * x_ref
    
    # 부등식 우변들
    taylor_approx = f_x_ref + grad_x_ref * (x_1d - x_ref)
    strong_convex_bound = taylor_approx + mu_val * 0.5 * (x_1d - x_ref)**2
    
    ax.plot(x_1d, f_vals, 'b-', linewidth=3, label='$f(x)$')
    ax.plot(x_1d, taylor_approx, 'r--', linewidth=2, label='1차 Taylor')
    ax.plot(x_1d, strong_convex_bound, 'g:', linewidth=2.5, label=f'강볼록 하한 ($\\mu={mu_val}$)')
    
    ax.fill_between(x_1d, f_vals, strong_convex_bound, 
                     where=(f_vals >= strong_convex_bound), alpha=0.3, color='yellow')
    ax.scatter([x_ref], [f_x_ref], color='red', s=100, zorder=5, marker='o',
              edgecolors='black', linewidths=2)
    
    ax.set_xlim(-2, 2)
    ax.set_ylabel('함수값')
    ax.set_xlabel('$x$')
    ax.set_title(f'강볼록성 ($\\mu={mu_val}$): 포물선 하한')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strong_convexity_bounds.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ 강볼록성 시각화 완료")

# ============================================
# 4. 정규화의 효과: 조건수 개선
# ============================================

print("\n" + "=" * 70)
print("4. Ridge 정규화가 조건수를 개선하는 메커니즘")
print("=" * 70)

# 인공 데이터셋
np.random.seed(42)
n_features = 2
A_data = np.array([[10.0, 1.0], [1.0, 1.0]])  # 조건수: 11

print(f"\n원본 데이터 행렬 A:")
print(f"  고유값: {np.linalg.eigvalsh(A_data)}")
print(f"  조건수: {np.linalg.cond(A_data):.4f}")

# 여러 λ 값에서 조건수 계산
lambda_values = np.logspace(-2, 1, 50)
condition_numbers = []

for lam in lambda_values:
    A_ridge = A_data + 2 * lam * np.eye(n_features)
    kappa = np.linalg.cond(A_ridge)
    condition_numbers.append(kappa)

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(lambda_values, condition_numbers, 'b-', linewidth=2.5, label='$\\kappa(\\lambda)$')
ax.axhline(np.linalg.cond(A_data), color='r', linestyle='--', 
          label=f'원본 $\\kappa$ = {np.linalg.cond(A_data):.2f}', linewidth=2)
ax.axvline(0.1, color='g', linestyle=':', alpha=0.7, linewidth=1.5, label='추천 $\\lambda$')

ax.set_xlabel('정규화 계수 $\\lambda$')
ax.set_ylabel('조건수 $\\kappa$')
ax.set_title('Ridge 정규화: 조건수 개선')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('regularization_condition_number.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ 정규화 효과 시각화 완료")

print("\n" + "=" * 70)
print("모든 검증 완료!")
print("=" * 70)
```

---

## 🔗 AI/ML 연결

1. **배치 정규화(Batch Normalization)**: 각 계층의 입력을 정규화하여 Hessian의 조건수를 개선. 신경망을 μ-강볼록처럼 행동하게 만들어 학습 안정성 향상.

2. **적응형 학습률(Adam, RMSprop)**: 과거 기울기의 제곱 이동평균을 사용하여 각 매개변수의 "국소 곡률"을 추정. 실질적으로 스텝 크기를 조정하여 κ를 개선.

3. **전처리(Preprocessing)**: 특성 정규화/표준화는 설계 행렬 X의 조건수를 개선 → 회귀 모델의 수렴 속도 향상.

---

## ⚖️ 가정과 한계

| 개념 | 가정 | 한계 |
|-----|------|------|
| **μ-강볼록** | $\nabla^2 f(x) \succeq \mu I$ globally | 국소 강볼록만 가능한 경우 많음 (신경망) |
| **L-smoothness** | $\nabla^2 f(x) \preceq L I$ globally | L을 보수적으로 설정하면 수렴 속도 예측 부정확 |
| **조건수** | μ, L이 고정값 | 신경망처럼 목적함수가 변하면 κ도 변함 |
| **GD 수렴률** | $(1-\mu/L)^k$ | 실제 신경망은 훨씬 복잡한 수렴 패턴 |

---

## 📌 핵심 정리

| 개념 | 정의 | 영향 |
|-----|------|------|
| **μ-강볼록성** | $f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$ | 유일해 존재, 빠른 수렴 |
| **L-smoothness** | $f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2$ | 그래디언트 변화 제한 |
| **조건수** | $\kappa = L/\mu$ | 수렴 속도 = $(1-1/\kappa)^k$ |
| **Descent Lemma** | $f(x - \frac{1}{L}\nabla f) \leq f(x) - \frac{1}{2L}\|\nabla f\|^2$ | GD 분석의 핵심 도구 |

---

## 🤔 생각해볼 문제

**문제 2.7**: 함수 $f(x) = \frac{1}{2}x^TAx - b^Tx$에서 $A = \begin{pmatrix} 4 & 1 \\ 1 & 1 \end{pmatrix}$일 때, μ와 L을 구하고 조건수를 계산하시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: Hessian을 구하고, 고유값을 계산하세요.

**해설**:
$$\nabla^2 f(x) = A = \begin{pmatrix} 4 & 1 \\ 1 & 1 \end{pmatrix}$$

특성방정식: $\det(A - \lambda I) = (4-\lambda)(1-\lambda) - 1 = \lambda^2 - 5\lambda + 3 = 0$

$$\lambda = \frac{5 \pm \sqrt{25-12}}{2} = \frac{5 \pm \sqrt{13}}{2}$$

$$\lambda_1 \approx 0.697, \quad \lambda_2 \approx 4.303$$

$$\mu = 0.697, \quad L = 4.303, \quad \kappa = \frac{L}{\mu} \approx 6.17$$

</details>

---

**문제 2.8**: 강볼록성의 정의에서, μ-강볼록 함수의 최솟값 $x^*$이 유일함을 증명하시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: 두 개의 서로 다른 최솟값이 존재한다고 가정하고 모순을 도출하세요.

**해설**:
$x_1 \neq x_2$가 모두 최솟값이라고 가정하면, $\nabla f(x_1) = \nabla f(x_2) = 0$ 및 $f(x_1) = f(x_2) = f^*$.

강볼록성:
$$f(x_2) \geq f(x_1) + \nabla f(x_1)^T(x_2-x_1) + \frac{\mu}{2}\|x_2-x_1\|^2$$
$$f^* \geq f^* + 0 + \frac{\mu}{2}\|x_2-x_1\|^2$$

따라서 $\|x_2 - x_1\|^2 \leq 0$, 즉 $x_1 = x_2$. 모순! □

</details>

---

**문제 2.9**: 조건수 κ = 1000인 이차 함수에서 GD의 수렴을 $(1-1/\kappa)^k$로 예측할 때, 오차를 10^-6으로 줄이기 위해 몇 번의 반복이 필요한가?

<details>
<summary>힌트 및 해설</summary>

**힌트**: $(1 - 1/1000)^k \leq 10^{-6}$을 풀어서 k를 구하세요.

**해설**:
$$(0.999)^k \leq 10^{-6}$$
$$k \log(0.999) \leq -6 \log(10)$$
$$k \geq \frac{6 \log(10)}{-\log(0.999)} = \frac{6 \times 2.303}{0.001} \approx 13818$$

약 **14000번** 반복이 필요합니다. κ가 크면 수렴이 지수적으로 느려집니다!

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 일계·이계 조건](./02-first-second-order-conditions.md) | [📚 README](../README.md) | [04. 볼록 함수의 연산 ▶](./04-convex-function-operations.md) |

</div>
