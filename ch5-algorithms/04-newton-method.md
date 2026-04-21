# 04. 뉴턴 방법의 국소·전역 수렴

## 🎯 핵심 질문
- 뉴턴 방법은 왜 2차 수렴하는가?
- 국소 수렴과 전역 수렴의 차이는 무엇인가?
- Hessian 역행렬 계산 비용이 정당화되는가?

## 🔍 왜 이 이론이 AI에서 중요한가
뉴턴 방법은 기하급수적 2차 수렴을 제공하여 최솟값 근처에서 극도로 빠릅니다. 하지만 Hessian 계산 비용이 크므로 대규모 문제에는 L-BFGS 같은 근사가 필요합니다. 이 단원에서는 뉴턴 방법의 이론과 실제 응용의 트레이드오프를 다룹니다.

## 📐 수학적 선행 조건
- **Hessian matrix**: $H = \nabla^2 f$, 2차 편미분
- **Lipschitz Hessian**: $\|H(x) - H(y)\| \le M\|x-y\|$
- **Positive definiteness**: $\lambda_{\min}(H(x^*)) > 0$
- **미분기하**: Newton 방향의 기하학적 의미

## 📖 직관적 이해

**1차 vs 2차**: 경사하강법은 1차 근사(평면)로 함수를 모델링합니다. 뉴턴 방법은 2차 근사(포물면)로 더 정확한 근사를 만듭니다.

**뉴턴 방향**: "현재 위치에서 2차 근사의 최솟값" 방향입니다:
$$\Delta x_N = -[H(x)]^{-1} \nabla f(x)$$

이는 1-step으로 정확한 최솟값을 찾을 수 있는 이차형식에서는 바로 해입니다.

**이차 수렴**: 오차가 제곱되므로 (\$\|e_{k+1}\| \sim \|e_k\|^2\$), 수렴하기 시작하면 매우 빠릅니다.

**감폭 (Damping)**: 시작점이 최솟값에서 멀면 뉴턴 방향이 나쁠 수 있으므로, 학습률 $t$를 도입합니다: $x_{k+1} = x_k + t \Delta x_N$.

## ✏️ 엄밀한 정의

**정의 (뉴턴 방법 - 순수)**:
$$x_{k+1} = x_k - [H(x_k)]^{-1} \nabla f(x_k)$$

**정의 (감폭 뉴턴 방법)**:
$$x_{k+1} = x_k - t_k [H(x_k)]^{-1} \nabla f(x_k)$$
여기서 $t_k \in (0, 1]$은 적응적 학습률.

**정의 (Newton decrement)**:
$$\lambda(x) = \sqrt{\nabla f(x)^T [H(x)]^{-1} \nabla f(x)}$$
수렴 판단 기준: $\lambda(x)$ 작음 → 최솟값 근처.

## 🔬 정리와 증명

**정리 1 (국소 2차 수렴)**

$f$가 $C^2$이고 $x^*$가 강한 최솟값 (즉, $H(x^*) \succ 0$ with eigenvalues in $[m, M]$)이면, $\|H(x)\| - H(y)\| \le L\|x-y\|$ (Lipschitz Hessian)일 때, 충분히 $x_0 \approx x^*$이면:

$$\|x_{k+1} - x^*\| \le \frac{L}{2m^2}\|x_k - x^*\|^2$$

*증명*:

**Step 1**: Taylor 전개를 2차까지:
$$\nabla f(x_k) = \nabla f(x^*) + H(x^*)(x_k - x^*) + O(\|x_k - x^*\|^2)$$

최적성에서 $\nabla f(x^*) = 0$이므로:
$$\nabla f(x_k) = H(x^*)(x_k - x^*) + O(\|x_k - x^*\|^2)$$

**Step 2**: Newton 스텝:
$$x_{k+1} = x_k - [H(x_k)]^{-1} \nabla f(x_k)$$

Hessian을 $H(x_k) = H(x^*) + (H(x_k) - H(x^*))$로 나누면:
$$[H(x_k)]^{-1} = [H(x^*)]^{-1} \left[I + [H(x^*)]^{-1}(H(x_k) - H(x^*))\right]^{-1}$$

**Step 3**: Perturbation 분석:
$$\|[H(x_k)]^{-1} - [H(x^*)]^{-1}\| \le \frac{\|H(x_k) - H(x^*)\|}{m^2}$$

Lipschitz Hessian: $\|H(x_k) - H(x^*)\| \le L\|x_k - x^*\|$

따라서:
$$\|[H(x_k)]^{-1}\| \le \frac{1}{m}(1 + O(\|x_k - x^*\|))$$

**Step 4**: 오차 재귀식:
$$e_{k+1} = x_{k+1} - x^* = x_k - x^* - [H(x_k)]^{-1}[\nabla f(x_k)]$$

적절한 계산을 통해:
$$e_{k+1} = -[H(x_k)]^{-1}[H(x_k) - H(x^*) - \int_0^1 H(x^* + t(x_k-x^*))(1-t)dt](x_k - x^*)$$

2차 Taylor 오차를 bound하면:
$$\|e_{k+1}\| \le \frac{L}{2m^2}\|e_k\|^2$$

여기서 $L$은 Hessian의 Lipschitz 상수, $m$은 $H(x^*)$의 최소 고유값. □

**정리 2 (Newton Decrement와 수렴)**

정리 1의 조건에서:
$$\|x_k - x^*\| \le \frac{2m^2}{3L\lambda(x_k)}$$

즉, Newton decrement $\lambda(x_k)$가 작으면 최솟값에 가깝습니다.

*증명*:

Newton decrement는 국소 근처에서:
$$\lambda(x) = \sqrt{\nabla f(x)^T [H(x)]^{-1} \nabla f(x)} \approx \sqrt{(x-x^*)^T H(x)(x-x^*)}$$

최솟값 근처에서:
$$\lambda(x)^2 \ge m\|x - x^*\|^2$$

따라서:
$$\|x - x^*\|^2 \le \frac{\lambda(x)^2}{m}$$

이를 정리하면 원하는 부등식을 얻습니다. □

**정리 3 (감폭 뉴턴 - 전역 수렴)**

$f$가 $C^2$이고 L-smooth, 강볼록이면, 감폭 뉴턴:
$$x_{k+1} = x_k - t_k \cdot [H(x_k)]^{-1}\nabla f(x_k)$$

여기서 $t_k$는 백트래킹 라인 서치 (sufficient decrease)로 선택되면:
- 전역 수렴: $\|x_k - x^*\| \to 0$
- 국소에서 2차 수렴 (quadratic phase)

*증명 스케치*:

**Phase 1 (선형 감소)**: 초반에 $t_k = 1$이 조건을 만족하지 않으면, 감폭으로 함수값이 일정량 감소:
$$f(x_{k+1}) \le f(x_k) - c_1 t_k \nabla f(x_k)^T [H(x_k)]^{-1} \nabla f(x_k)$$

이는 충분한 감소를 보장합니다.

**Phase 2 (2차 수렴)**: 최솟값 근처에서 $t_k = 1$이 조건을 만족하게 되고, 정리 1이 적용되어 2차 수렴.

따라서 처음엔 천천히, 나중엔 빠르게 수렴합니다. □

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 테스트 함수 1: 강볼록 이차형식
def quadratic(x, A, b):
    return 0.5 * x @ A @ x + b @ x

def grad_quadratic(x, A, b):
    return A @ x + b

def hess_quadratic(x, A, b):
    return A

# 테스트 함수 2: Rosenbrock (비강볼록, 어려운 함수)
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosenbrock(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])

def hess_rosenbrock(x):
    return np.array([
        [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
        [-400*x[0], 200]
    ])

# 1. 이차형식 설정
np.random.seed(42)
n = 50
Q = np.random.randn(n, n)
A = Q @ Q.T + np.eye(n)
b = np.random.randn(n)

x_opt = -np.linalg.solve(A, b)
f_opt = quadratic(x_opt, A, b)

eigenvalues = np.linalg.eigvalsh(A)
m = np.min(eigenvalues)  # Strong convexity
M = np.max(eigenvalues)  # Smoothness

print(f"Strong convexity parameter m = {m:.4f}")
print(f"Smoothness parameter L = {M:.4f}")
print(f"Condition number κ = {M/m:.2f}")

# 2. 뉴턴 방법 (순수)
def newton_method(x0, grad_f, hess_f, num_iters=50):
    x = x0.copy()
    losses = []
    
    for k in range(num_iters):
        grad = grad_f(x)
        hess = hess_f(x)
        
        try:
            # Hessian이 수치적으로 불안정할 수 있음
            delta_x = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print(f"Hessian singular at iteration {k}")
            break
        
        x = x + delta_x
        
        # 함수값 기록
        f_val = quadratic(x, A, b) - f_opt if n == len(x) else rosenbrock(x)
        losses.append(f_val)
    
    return x, np.array(losses)

# 3. 감폭 뉴턴 방법 (백트래킹)
def damped_newton_method(x0, grad_f, hess_f, f_func, num_iters=50, c1=1e-4):
    x = x0.copy()
    losses = []
    
    for k in range(num_iters):
        grad = grad_f(x)
        hess = hess_f(x)
        
        try:
            delta_x = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print(f"Hessian singular at iteration {k}")
            break
        
        # 백트래킹 라인 서치
        f_x = f_func(x)
        t = 1.0
        alpha = 0.5  # 감폭 인수
        
        for _ in range(20):
            x_new = x + t * delta_x
            f_new = f_func(x_new)
            
            # Sufficient decrease 조건
            if f_new <= f_x + c1 * t * grad @ delta_x:
                x = x_new
                break
            
            t *= alpha
        
        losses.append(f_func(x))
    
    return x, np.array(losses)

# 4. 경사하강법 (비교용)
def gradient_descent(x0, grad_f, f_func, eta=0.01, num_iters=1000):
    x = x0.copy()
    losses = []
    
    for k in range(num_iters):
        grad = grad_f(x)
        x = x - eta * grad
        losses.append(f_func(x))
    
    return x, np.array(losses)

# 실행
x0_quad = np.random.randn(n)
x_newton, losses_newton = newton_method(
    x0_quad,
    lambda x: grad_quadratic(x, A, b),
    lambda x: hess_quadratic(x, A, b),
    num_iters=20
)
x_damped, losses_damped = damped_newton_method(
    x0_quad,
    lambda x: grad_quadratic(x, A, b),
    lambda x: hess_quadratic(x, A, b),
    lambda x: quadratic(x, A, b),
    num_iters=20
)
x_gd, losses_gd = gradient_descent(
    x0_quad,
    lambda x: grad_quadratic(x, A, b),
    lambda x: quadratic(x, A, b),
    eta=1/M,
    num_iters=100
)

# 시각화 1: 이차형식 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 로그-선형
ax = axes[0, 0]
k_newton = np.arange(len(losses_newton))
k_gd = np.arange(len(losses_gd))

# Relative losses
rel_newton = np.array(losses_newton) / np.max([np.abs(losses_newton[0]), 1e-10])
rel_gd = np.array(losses_gd) / np.max([np.abs(losses_gd[0]), 1e-10])

ax.semilogy(k_newton, rel_newton, 'r-o', label='Newton', linewidth=2, markersize=5)
ax.semilogy(k_gd, rel_gd, 'b-o', label='Gradient Descent', linewidth=2, markersize=5)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('Relative loss', fontsize=11)
ax.set_title('Newton vs GD: Quadratic Function', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (2) 처음 10회 iteration 상세
ax = axes[0, 1]
ax.semilogy(k_newton[:10], rel_newton[:10], 'r-o', label='Newton (2nd order)', linewidth=2, markersize=6)
ax.semilogy(k_gd[:20], rel_gd[:20], 'b-o', label='GD (1st order)', linewidth=2, markersize=6)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('Relative loss', fontsize=11)
ax.set_title('First Iterations: Quadratic Convergence', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (3) 오차의 제곱 관계 (e_k+1 vs e_k^2)
ax = axes[1, 0]
errors_newton = np.abs(rel_newton) + 1e-15
errors_gd = np.abs(rel_gd) + 1e-15

# Newton에서 e_k+1 vs e_k^2
if len(errors_newton) > 1:
    e_squared = errors_newton[:-1]**2
    e_next = errors_newton[1:]
    
    # 기울기 추정 (L/(2m^2))
    valid_idx = e_squared > 1e-15
    if np.sum(valid_idx) > 1:
        coeffs = e_next[valid_idx] / e_squared[valid_idx]
        ax.scatter(e_squared[valid_idx], e_next[valid_idx], s=50, alpha=0.6, label='Newton')
        
        # 이론적 직선
        e_range = np.logspace(-8, -1, 100)
        C_est = np.mean(coeffs)
        ax.plot(e_range, C_est * e_range, 'r--', label=f'Predicted: {C_est:.2e}·e²', linewidth=2)

ax.set_xlabel('||e_k||² (log)', fontsize=11)
ax.set_ylabel('||e_{k+1}|| (log)', fontsize=11)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Quadratic Convergence Verification', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# (4) Newton decrement 변화
ax = axes[1, 1]
lambda_values = []
for k in range(min(15, len(losses_newton))):
    x = x0_quad.copy()
    for _ in range(k):
        grad = grad_quadratic(x, A, b)
        hess = hess_quadratic(x, A, b)
        try:
            delta_x = -np.linalg.solve(hess, grad)
            x = x + delta_x
        except:
            break
    
    grad = grad_quadratic(x, A, b)
    hess = hess_quadratic(x, A, b)
    try:
        hess_inv = np.linalg.inv(hess)
        lambda_val = np.sqrt(grad @ hess_inv @ grad)
        lambda_values.append(lambda_val)
    except:
        pass

if lambda_values:
    ax.semilogy(lambda_values, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration k', fontsize=11)
    ax.set_ylabel('Newton Decrement λ(x_k)', fontsize=11)
    ax.set_title('Newton Decrement Over Iterations', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('newton_convergence.png', dpi=150)
plt.show()

# Rosenbrock 함수 테스트
print("\n" + "="*60)
print("Rosenbrock Function Test")
print("="*60)

x0_rosen = np.array([-1.2, 1.0])
x_newton_r, losses_newton_r = damped_newton_method(
    x0_rosen,
    grad_rosenbrock,
    hess_rosenbrock,
    rosenbrock,
    num_iters=30
)
x_gd_r, losses_gd_r = gradient_descent(
    x0_rosen,
    grad_rosenbrock,
    rosenbrock,
    eta=0.001,
    num_iters=5000
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(losses_newton_r, 'r-o', label='Damped Newton', linewidth=2, markersize=4)
ax.semilogy(losses_gd_r[:len(losses_newton_r)*10], 'b-o', label='GD (sampled)', linewidth=2, markersize=4)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('f(x_k) - f* (log)', fontsize=11)
ax.set_title('Newton vs GD on Rosenbrock Function', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('newton_rosenbrock.png', dpi=150)
plt.show()

print(f"Newton iterations: {len(losses_newton_r)}")
print(f"GD iterations (for reference): {len(losses_gd_r)}")
```

## 🔗 AI/ML 연결

**L-BFGS**: 완전한 Hessian 계산 대신 근사를 사용하여, Newton의 빠른 수렴과 GD의 낮은 비용을 결합합니다.

**Second-order methods**: 최근 연구에서 신경망 학습에 2차 방법 (K-FAC, Shampoo)이 재조명받고 있습니다.

**최적화 라이브러리**: scipy.optimize.minimize의 'BFGS', 'L-BFGS-B'는 모두 Newton 계열입니다.

## ⚖️ 가정과 한계

**가정 1**: Hessian이 양정치(positive definite)여야 합니다. 안장점에서는 실패합니다.

**가정 2**: $C^2$ 미분가능성이 필요합니다. 신경망 손실은 ReLU 때문에 미분 불가능합니다.

**계산 비용**: Hessian 역행렬 계산은 $O(n^3)$이므로 대규모 문제에는 부적합합니다.

**국소 수렴**: 시작점이 최솟값에서 멀면 수렴 보장이 없습니다 (감폭으로 부분 해결).

## 📌 핵심 정리

1. **2차 수렴**: $\|e_{k+1}\| \lesssim \|e_k\|^2$ — 매우 빠름.
2. **국소 vs 전역**: 순수 Newton은 국소만, 감폭은 전역 수렴.
3. **Newton decrement**: 수렴 판정 기준 $\lambda(x) = \sqrt{\nabla f^T H^{-1} \nabla f}$.
4. **계산 트레이드오프**: 2차 수렴 vs $O(n^3)$ Hessian 역행렬.
5. **L-BFGS 필요**: 대규모 문제는 근사 Hessian 사용.

## 🤔 생각해볼 문제

**문제 1**: 2차 수렴 $\|e_{k+1}\| \le C\|e_k\|^2$에서 상수 $C = L/(2m^2)$의 의미를 설명하시오. $L$과 $m$이 크면/작으면 어떻게 되는가?

<details>
<summary>힌트 및 해설</summary>

$L$은 Hessian의 Lipschitz 상수 (곡률 변화 속도), $m$은 강볼록성 파라미터입니다.

- $L$이 크면 (곡률이 빠르게 변함) → $C$가 크므로 수렴 느림
- $m$이 작으면 (약한 강볼록) → $C$가 크므로 수렴 느림

따라서 조건이 좋아야 2차 수렴 이점을 본다.

</details>

**문제 2**: 뉴턴 데크리먼트 $\lambda(x)$가 작으면 최솟값에 가까운 이유는? 정리 2에서 쓰인 Hessian의 성질을 설명하시오.

<details>
<summary>힌트 및 해설</summary>

$\lambda(x)^2 = \nabla f(x)^T H(x)^{-1} \nabla f(x)$

최솟값 근처에서:
$$\lambda(x)^2 \approx (x - x^*)^T H(x) (x - x^*) \ge m\|x - x^*\|^2$$

따라서 $\lambda$가 작으면 $\|x - x^*\|$도 작습니다.

</details>

**문제 3**: 감폭 뉴턴에서 백트래킹 라인 서치가 왜 필요한가? 순수 뉴턴이 발산할 수 있는 예를 들고, 코드로 검증하시오.

<details>
<summary>힌트 및 해설</summary>

시작점이 최솟값에서 멀면, Hessian이 부정확하거나 수치 오차로 $H^{-1}$이 잘못될 수 있습니다.

예: Rosenbrock 함수에서 초기점 $(-1.2, 1)$에서 순수 Newton은 발산할 수 있습니다.

라인 서치는 매 스텝 $f(x_{k+1}) < f(x_k)$를 강제하여 함수값을 항상 감소시킵니다.

</details>

<div align="center">
| [◀ 03. 하한 경계(Lower Bound)](./03-lower-bound.md) | [📚 README](../README.md) | [05. Interior Point Method ▶](./05-interior-point.md) |
</div>
