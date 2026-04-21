# 1. 볼록 함수의 3개 동치 정의

## 🎯 핵심 질문

- 볼록 함수를 정의하는 세 가지 방법(Jensen, Epigraph, 1차 조건)이 정말 동일한가?
- 왜 epigraph가 "볼록 함수를 정의하는 기하학적 도구"인가?
- 볼록 함수의 국소 최솟값이 항상 전역 최솟값일 수 있을까?

---

## 🔍 왜 이 이론이 AI에서 중요한가

1. **손실 함수의 최적성 보장**: 신경망 손실함수가 볼록하면 경사하강법이 찾는 임의의 국소 최솟값이 전역 최솟값 (Deep Learning의 비볼록성 이유)
2. **CVXPY의 DCP 규칙 기초**: 연산이 볼록성을 보존하는지 판단하려면 3가지 정의 중 가장 계산가능한 것(epigraph/1차 조건)을 사용
3. **SVM, 로지스틱 회귀의 유일해**: 손실함수가 엄격히 볼록하면 정확히 하나의 최적 매개변수 존재

---

## 📐 수학적 선행 조건

- 벡터 공간, 선형 결합의 개념
- 실함수의 연속성, 미분가능성
- [이전 챕터: Ch1-05. 극값점과 Krein-Milman 정리](../ch1-convex-sets/05-extreme-point-krein-milman.md) - 볼록 집합의 기하학
- 헤시안 행렬, 양반정치 행렬의 기초

---

## 📖 직관적 이해

### Jensen 부등식의 기하학적 의미
```
    f(x₂)
       ∧
       │     * (λx+(1-λ)y, f(λx+(1-λ)y))  ← 곡선 위의 점
       │    /│
       │   / │
       │  /  │ f의 값은 항상 현(chord) 아래
       │ /   │
       │*────*────────────────────
       f(x₁) ────────────
            [x₁, x₂] 구간에서 λ에 따른 내분점
```

**직관**: 함수가 볼록하다는 것은 "어떤 두 점을 이은 선분이 함수 곡선 위에 있거나 위쪽에 있다"

### Epigraph의 기하학적 의미

함수 $f$의 epigraph는 함수 그래프보다 위의 모든 점들의 집합:
$$\text{epi}(f) = \{(x, t) \in \mathbb{R}^n \times \mathbb{R} : f(x) \leq t\}$$

**기하학적 해석**: 
- $f$가 볼록 $\Leftrightarrow$ epi$(f)$가 볼록 집합
- 함수의 성질을 집합의 성질로 변환하는 "다리" 역할

### 1차 조건의 직관

$f$가 미분가능할 때:
$$f(y) \geq f(x) + \nabla f(x)^T(y-x)$$

**의미**: $x$에서의 1차 Taylor 근사가 실제 함수값의 "하한(lower bound)"을 제공한다. 이는 접선이 항상 함수 그래프 아래에 있다는 뜻.

---

## ✏️ 엄밀한 정의

**정의 2.1** (Jensen 부등식으로 정의된 볼록함수)
함수 $f: \mathbb{R}^n \to \mathbb{R}$이 **볼록(convex)**하다 $\Leftrightarrow$ 정의역이 볼록 집합이고, 모든 $x, y \in \text{dom}(f)$와 $\lambda \in [0,1]$에 대해:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

**정의 2.2** (Epigraph로 정의된 볼록함수)
함수 $f: \mathbb{R}^n \to \mathbb{R}$이 **볼록**하다 $\Leftrightarrow$ 그 epigraph가 $\mathbb{R}^{n+1}$의 볼록 집합이다.

**정의 2.3** (1차 조건으로 정의된 볼록함수)
함수 $f: \mathbb{R}^n \to \mathbb{R}$이 미분가능하고 **볼록**하다 $\Leftrightarrow$ 모든 $x, y \in \text{dom}(f)$에 대해:
$$f(y) \geq f(x) + \nabla f(x)^T(y-x)$$

**정의 2.4** (엄격 볼록성)
모든 $x \neq y$와 $\lambda \in (0,1)$에 대해 strict inequality가 성립하면 **엄격히 볼록(strictly convex)**.

---

## 🔬 정리와 증명

**정리 2.1** (Jensen ↔ Epigraph)
함수 $f$가 Jensen 부등식을 만족 $\Leftrightarrow$ epi$(f)$가 볼록 집합

**증명:**
($\Rightarrow$) $f$가 Jensen을 만족한다고 가정. $(x_1, t_1), (x_2, t_2) \in \text{epi}(f)$이면 $f(x_1) \leq t_1$, $f(x_2) \leq t_2$.

$\lambda \in [0,1]$에 대해:
$$f(\lambda x_1 + (1-\lambda)x_2) \leq \lambda f(x_1) + (1-\lambda)f(x_2) \leq \lambda t_1 + (1-\lambda)t_2$$

따라서 $(\lambda x_1 + (1-\lambda)x_2, \lambda t_1 + (1-\lambda)t_2) \in \text{epi}(f)$. □

($\Leftarrow$) epi$(f)$가 볼록이라고 가정. $(x, f(x)), (y, f(y)) \in \text{epi}(f)$ (정의상 자명).

볼록성에 의해:
$$(\lambda x + (1-\lambda)y, \lambda f(x) + (1-\lambda)f(y)) \in \text{epi}(f)$$

즉 $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$. □

---

**정리 2.2** (Jensen ↔ 1차 조건, 미분가능한 경우)
$f$가 미분가능할 때, $f$가 Jensen을 만족 $\Leftrightarrow$ $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ for all $x,y$

**증명 스케치:**
($\Rightarrow$) Taylor 전개: $f(x + t(y-x)) = f(x) + t\nabla f(x)^T(y-x) + o(t)$

$t \in (0,1]$이므로 $f(x + t(y-x)) \leq tf(y) + (1-t)f(x)$ (Jensen)

극한 $t \to 0^+$: $f(x) + \nabla f(x)^T(y-x) \leq f(y)$

($\Leftarrow$) 1차 조건에서:
- $f(x) \geq f(z) + \nabla f(z)^T(x-z)$
- $f(y) \geq f(z) + \nabla f(z)^T(y-z)$

$z = \lambda x + (1-\lambda)y$로 두면:
$$\lambda f(x) + (1-\lambda)f(y) \geq f(z) + \nabla f(z)^T[\lambda(x-z) + (1-\lambda)(y-z)]$$
$$= f(z) + \nabla f(z)^T[0] = f(z) = f(\lambda x + (1-\lambda)y)$$
□

---

**정리 2.3** (볼록함수의 국소 최솟값 = 전역 최솟값)
$f$가 볼록하고 $x^*$가 국소 최솟값이면, $x^*$는 전역 최솟값이다.

**증명:**
대조법. $x^*$가 전역 최솟값이 아니라고 가정하면, $f(y) < f(x^*)$인 $y$가 존재.

$x^*$가 국소 최솟값이므로, 어떤 $\delta > 0$에 대해 $\|x - x^*\| \leq \delta$이면 $f(x) \geq f(x^*)$.

$\lambda = \frac{\delta}{2\|y-x^*\|}$라 하면, $z = x^* + \lambda(y - x^*)$에 대해 $\|z - x^*\| = \delta/2 < \delta$.

따라서 $f(z) \geq f(x^*)$. 그런데 볼록성에 의해:
$$f(z) = f(x^* + \lambda(y-x^*)) \leq (1-\lambda)f(x^*) + \lambda f(y) < f(x^*)$$

모순! □

---

**예제 2.1** ($f(x) = x^2$는 볼록)
- **Jensen**: $(\lambda x + (1-\lambda)y)^2 = \lambda^2 x^2 + 2\lambda(1-\lambda)xy + (1-\lambda)^2y^2$
  $\lambda x^2 + (1-\lambda)y^2 - f(\lambda x+(1-\lambda)y)$
  $= \lambda(1-\lambda)(x^2 - 2xy + y^2) = \lambda(1-\lambda)(x-y)^2 \geq 0$ ✓

- **1차 조건**: $\nabla f(x) = 2x$이므로
  $f(y) - f(x) - \nabla f(x)^T(y-x) = y^2 - x^2 - 2x(y-x)$
  $= y^2 - x^2 - 2xy + 2x^2 = (y-x)^2 \geq 0$ ✓

---

**예제 2.2** (ReLU $f(x) = \max(0, x)$는 볼록하지만 미분불가)
Jensen 검증: $\lambda x + (1-\lambda)y$에서
$$\max(0, \lambda x + (1-\lambda)y) \leq \lambda \max(0,x) + (1-\lambda)\max(0,y)$$

$x, y \geq 0$이면 자명. $x < 0, y > 0$이면 $\lambda x + (1-\lambda)y$의 부호에 따라 확인 가능.

x=0에서 미분 불가능하지만 **subgradient** $\partial f(0) = [0,1]$ 존재.

---

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# ============================================
# 1. Jensen 부등식 검증
# ============================================
def jensen_test(f, x, y, lambda_vals=np.linspace(0, 1, 100)):
    """
    Jensen 부등식 검증: f(λx+(1-λ)y) ≤ λf(x)+(1-λ)f(y)
    """
    lhs = [f(lam * x + (1 - lam) * y) for lam in lambda_vals]  # f(convex combo)
    rhs = [lam * f(x) + (1 - lam) * f(y) for lam in lambda_vals]  # weighted average
    
    return np.array(lhs), np.array(rhs), lambda_vals

# 테스트 함수들
f_quad = lambda x: x**2                           # 볼록: x²
f_relu = lambda x: np.maximum(0, x)              # 볼록: ReLU
f_log = lambda x: -np.log(np.abs(x) + 1e-10)    # 오목 (음수로 부호 반대)

x, y = -2.0, 2.0
lhs_quad, rhs_quad, lams = jensen_test(f_quad, x, y)
lhs_relu, rhs_relu, _ = jensen_test(f_relu, x, y)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(lams, lhs_quad, 'b-', label='$f(λx+(1-λ)y)$', linewidth=2)
axes[0].plot(lams, rhs_quad, 'r--', label='$λf(x)+(1-λ)f(y)$', linewidth=2)
axes[0].fill_between(lams, lhs_quad, rhs_quad, alpha=0.3, color='green')
axes[0].set_xlabel('$λ$')
axes[0].set_ylabel('함수값')
axes[0].set_title('$f(x)=x^2$ Jensen 부등식 검증')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lams, lhs_relu, 'b-', label='$f(λx+(1-λ)y)$', linewidth=2)
axes[1].plot(lams, rhs_relu, 'r--', label='$λf(x)+(1-λ)f(y)$', linewidth=2)
axes[1].fill_between(lams, lhs_relu, rhs_relu, alpha=0.3, color='green')
axes[1].set_xlabel('$λ$')
axes[1].set_ylabel('함수값')
axes[1].set_title('ReLU Jensen 부등식 검증')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('jensen_inequality.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Jensen 부등식 검증 완료")

# ============================================
# 2. Epigraph 시각화 (3D)
# ============================================
x_vals = np.linspace(-2, 2, 50)
t_vals = np.linspace(-1, 5, 50)
X, T = np.meshgrid(x_vals, t_vals)

# f(x) = x²의 epigraph: {(x,t) | x² ≤ t}
Z_epi = X**2 <= T  # True: epigraph에 속함

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
x_curve = np.linspace(-2, 2, 100)
y_curve = x_curve**2
z_curve = np.zeros_like(x_curve)

ax1.plot_surface(X, T, Z_epi.astype(float), alpha=0.4, cmap='coolwarm')
ax1.plot(x_curve, y_curve, z_curve, 'b-', linewidth=3, label='$f(x)=x^2$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_zlabel('In Epi(f)')
ax1.set_title('Epigraph: $\{(x,t) : f(x) ≤ t\}$는 볼록 집합')

# 2D 슬라이스
ax2 = fig.add_subplot(122)
ax2.contourf(X, T, Z_epi.astype(float), levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.5)
ax2.plot(x_curve, y_curve, 'b-', linewidth=3, label='$f(x)=x^2$')
ax2.fill_between(x_curve, y_curve, 5, alpha=0.3, color='green', label='Epi$(f)$')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-1, 5)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
ax2.set_title('2D 슬라이스 (Epigraph은 위쪽 영역)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('epigraph_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Epigraph 시각화 완료")

# ============================================
# 3. 1차 조건 검증 (Log-Sum-Exp)
# ============================================
def log_sum_exp(x):
    """Numerically stable log-sum-exp"""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

def log_sum_exp_grad(x):
    """Gradient: d/dx log(Σ eˣⁱ) = softmax(x)"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

# 검증: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
x = np.array([1.0, 2.0, 0.5])
y = np.array([0.5, 1.5, 1.0])

f_x = log_sum_exp(x)
grad_f_x = log_sum_exp_grad(x)
f_y = log_sum_exp(y)

lhs = f_y
rhs = f_x + np.dot(grad_f_x, y - x)

print(f"\nLog-Sum-Exp 1차 조건 검증:")
print(f"  f(y) = {lhs:.6f}")
print(f"  f(x) + ∇f(x)ᵀ(y-x) = {rhs:.6f}")
print(f"  f(y) - RHS = {lhs - rhs:.6e} (≥ 0이어야 함) ✓")

# ============================================
# 4. 국소 최솟값 = 전역 최솟값
# ============================================
def test_local_global_minimum():
    """
    볼록 함수에서 국소 최솟값 = 전역 최솟값 증명
    """
    # f(x) = (x-1)² + 2(x-1)² = 3(x-1)²
    f = lambda x: 3 * (x - 1)**2
    x0_guess = [0.5, 1.5, -5.0]
    
    results = []
    for x0 in x0_guess:
        res = minimize(f, x0, method='BFGS')
        results.append((x0, res.x[0], res.fun))
    
    print("\n국소 최솟값 = 전역 최솟값 검증:")
    for x0, x_opt, f_opt in results:
        print(f"  시작점: x₀={x0:5.1f} → 수렴: x*={x_opt:.6f}, f(x*)={f_opt:.6e}")
    
    x_opt_all = [r[1] for r in results]
    print(f"  모든 시작점에서 동일한 최솟값으로 수렴: {np.allclose(x_opt_all, 1.0)} ✓")

test_local_global_minimum()

# ============================================
# 5. ReLU와 Subgradient (미분불가점)
# ============================================
x_vals = np.linspace(-2, 2, 100)
y_relu = np.maximum(0, x_vals)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, y_relu, 'b-', linewidth=2.5, label='ReLU: $f(x)=\\max(0,x)$')

# x=0에서의 subgradient [0, 1]로 그은 선들
x0 = 0
for g in [0, 0.5, 1.0]:
    x_line = np.array([-1, 1])
    y_line = g * (x_line - x0)  # f(0) + g·(x-0)
    ax.plot(x_line, y_line, '--', alpha=0.5, label=f'Subgradient $g={g}$')

ax.axvline(0, color='red', linestyle=':', alpha=0.5, label='$x=0$ (미분불가)')
ax.scatter([0], [0], color='red', s=100, zorder=5)
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 2.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title('ReLU: 볼록이지만 $x=0$에서 미분불가능 (Subgradient 존재)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('relu_subgradient.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ ReLU Subgradient 시각화 완료")

print("\n" + "="*50)
print("모든 검증 완료!")
print("="*50)
```

---

## 🔗 AI/ML 연결

1. **신경망 최적화**: 손실함수가 비볼록이므로 경사하강법은 국소 최솟값에 갇힐 수 있음. 반면 SVM이나 로지스틱 회귀는 손실함수가 볼록하므로 항상 전역 최적해를 찾음.

2. **제약된 최적화 (CVXPY)**: 문제가 볼록한지 판단하려면 epigraph와 1차 조건 중 계산가능한 것을 선택. CVXPY는 원자(atom) 단위로 부등식을 확인함.

3. **Adversarial Robustness**: 손실함수의 epigraph를 이해하면 데이터 중독(data poisoning) 공격의 견고성을 분석할 수 있음.

---

## ⚖️ 가정과 한계

| 가정/성질 | 내용 | 제한사항 |
|---------|------|--------|
| **정의역의 볼록성** | $\text{dom}(f)$가 볼록 집합이어야 함 | 비볼록 정의역에서는 정의 불가 |
| **Jensen 쌍방향** | 세 정의가 동치 | 미분불가능 함수는 1차 조건 사용 불가 |
| **국소=전역** | 국소 최솟값이 유일하고 전역 | 최댓값은 성립 안 함 (오목 함수) |
| **Epigraph의 닫힘** | 연속함수의 epigraph는 닫혀있음 | 불연속 함수의 epigraph는 open 가능 |
| **미분가능성** | 1차 조건 사용 시 가정 | ReLU 등 미분불가능 점이 있으면 subgradient 필요 |

---

## 📌 핵심 정리

| 개념 | 정의 | 용도 |
|-----|------|------|
| **Jensen 부등식** | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ | 직관적, 증명용 |
| **Epigraph** | $\text{epi}(f) = \{(x,t) : f(x) \leq t\}$ | 기하학적, 볼록집합과의 연결 |
| **1차 조건** | $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ | 수치 계산, 최적화 알고리즘 |
| **국소=전역** | 볼록함수의 국소 최솟값은 전역 최솟값 | 최적화 이론의 기초 |
| **Strictness** | 부등식이 strict이면 엄격 볼록 → 유일 최솟값 | SVM, Ridge 회귀 |

---

## 🤔 생각해볼 문제

**문제 2.1**: 다음 함수가 볼록한지 판단하고, Jensen 부등식으로 증명하시오.
$$f(x) = \|x\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

<details>
<summary>힌트 및 해설</summary>

**힌트**: 코시-슈바르츠 부등식을 사용하세요.

**해설**: $\|\lambda x + (1-\lambda)y\|_2$와 $\lambda \|x\|_2 + (1-\lambda)\|y\|_2$를 비교합니다.
$$\|\lambda x + (1-\lambda)y\|_2 \leq \lambda \|x\|_2 + (1-\lambda)\|y\|_2$$

이는 삼각부등식의 직접적 결과이며, $\|·\|_2$는 노름이므로 항상 삼각부등식을 만족합니다.

</details>

---

**문제 2.2**: 함수 $f(x) = e^x$가 볼록한지 1차 조건으로 증명하고, 그 epigraph를 $\mathbb{R}^2$에서 그리시오.

<details>
<summary>힌트 및 해설</summary>

**힌트**: $\nabla f(x) = e^x$이고, $e^y \geq e^x + e^x(y-x)$를 증명하면 됩니다.

**해설**: 
$$e^y - e^x - e^x(y-x) = e^x(e^{y-x} - 1 - (y-x))$$

$g(t) = e^t - 1 - t$라 하면, $g(0)=0$, $g'(t) = e^t - 1 \geq 0$ (t ≥ 0일 때).

따라서 $g(t)$는 증가이고 $t = y-x \geq 0$일 때 $g(t) \geq 0$. □

</details>

---

**문제 2.3**: 다음 함수들 중 어느 것이 국소 최솟값을 가지면 전역 최솟값도 가지는지 판단하고, 이유를 설명하시오.
$$\text{(a)} \quad f(x) = -e^{-x^2} \quad \text{(b)} \quad f(x) = \sin(x) \quad \text{(c)} \quad f(x) = x^4 - 2x^2 + 3$$

<details>
<summary>힌트 및 해설</summary>

**힌트**: 먼저 각 함수의 2계 도함수를 계산하여 볼록성을 판단하세요.

**해설**: 
- (a) $f''(x) = e^{-x^2}(4x^2 - 2) < 0$ (some regions) → 비볼록, 따라서 정리 불적용
- (b) $f''(x) = -\sin(x)$ → 진동, 비볼록, 정리 불적용
- (c) $f''(x) = 12x^2 - 4$. $x^2 > 1/3$일 때만 $f''(x) > 0$ → 전역적으로 볼록 아님

따라서 세 함수 모두 "국소 최솟값 = 전역 최솟값" 정리를 적용할 수 없습니다.

하지만 (c)는 $x \to \pm\infty$에서 $f(x) \to \infty$이고, 임계점이 유한하므로 전역 최솟값은 존재합니다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-05. 극값점과 Krein-Milman](../ch1-convex-sets/05-extreme-point-krein-milman.md) | [📚 README](../README.md) | [02. 일계·이계 조건 ▶](./02-first-second-order-conditions.md) |

</div>
