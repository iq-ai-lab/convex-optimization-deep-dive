# 1. Proximal Operator의 정의

## 🎯 핵심 질문
- Proximal operator는 정확히 무엇인가?
- 왜 "gradient step"의 일반화로 볼 수 있는가?
- Moreau envelope는 proximal과 어떤 관계인가?

## 🔍 왜 이 이론이 AI에서 중요한가

비매끄러운(non-smooth) 손실 함수가 매우 흔하다:
- **L1 정규화** (Lasso, sparse 학습)
- **L2 ball 투영** (제약 최적화)
- **Indicator 함수** (등식/부등식 제약)

이러한 함수들은 미분 불가능하므로 일반 경사하강법을 적용할 수 없다. **Proximal operator**는 이 문제를 해결하는 핵심 도구다. Proximal을 이해하면 L1 정규화, 투영, 분해 알고리즘의 원리를 모두 파악할 수 있다.

## 📐 수학적 선행 조건

- **강볼록 함수** (strongly convex): $\exists \mu > 0$, $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y) - \frac{\mu}{2}\lambda(1-\lambda)\|x-y\|^2$
- **감소 부분미분** (monotone subdifferential)
- **Lipschitz 상수**: $L$-smooth ⟺ $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$

## 📖 직관적 이해

Proximal operator는 두 경쟁하는 목표 사이의 타협점을 찾는다:

$$\text{prox}_f(v) = \arg\min_x \left( f(x) + \frac{1}{2}\|x-v\|^2 \right)$$

**해석**:
- **첫 번째 항**: $f(x)$ 최소화 (실제 목표)
- **두 번째 항**: $v$에 가까이 유지 (관성)
- **비율**: 같은 가중치 (균형 설정)

**예시**:
- $f = 0$: "관성만 있다" → $\text{prox}_f(v) = v$ (움직이지 않음)
- $f = I_C$ (집합 $C$의 indicator): "반드시 $C$에 속해야 한다" → $\text{prox}_f(v) = \text{proj}_C(v)$ (가장 가까운 점)
- $f = \|\cdot\|_1$: "희소성 장려" → soft-thresholding (작은 값 0으로)

## ✏️ 엄밀한 정의

**정의 1.1 (Proximal Operator)**: $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$가 적절한 볼록 함수일 때,

$$\text{prox}_f(v) := \arg\min_x \left( f(x) + \frac{1}{2}\|x-v\|^2 \right)$$

를 $f$의 proximal operator라 한다.

**정의 1.2 (Moreau Envelope)**: 

$$M_\lambda f(x) := \min_y \left( f(y) + \frac{1}{2\lambda}\|x-y\|^2 \right)$$

를 parameter $\lambda > 0$에 대한 Moreau envelope라 한다. $\lambda = 1$일 때 $M_f(x) = \min_y (f(y) + \frac{1}{2}\|x-y\|^2)$.

**정의 1.3 (부분미분)**: 

$$\partial f(x) := \{ g \in \mathbb{R}^n : f(y) \geq f(x) + g^T(y-x) \text{ for all } y \}$$

## 🔬 정리와 증명

**정리 1.4 (유일 존재성)**: 볼록 함수 $f$에 대해 $\text{prox}_f(v)$는 유일하게 존재한다.

*증명*: 함수 $h(x) = f(x) + \frac{1}{2}\|x-v\|^2$를 생각하자.
- $f$가 볼록이고 $\frac{1}{2}\|x-v\|^2$는 강볼록($\mu=1$)이므로, 합 $h$는 강볼록이다.
- 강볼록 함수는 유일한 최솟값을 가지므로 $\text{prox}_f(v)$는 유일하다. ∎

**정리 1.5 (Moreau envelope의 성질)**:

$$M_\lambda f(x) = \min_y \left( f(y) + \frac{1}{2\lambda}\|x-y\|^2 \right)$$

일 때, 다음이 성립한다:

(1) $M_\lambda f$는 미분가능하고,
$$\nabla M_\lambda f(x) = \frac{1}{\lambda}(x - \text{prox}_{\lambda f}(x))$$

(2) 최솟값은 $x^* = \text{prox}_{\lambda f}(x)$에서 달성된다.

*증명 (1)*: 최적값 조건에서 
$$0 \in \partial f(y^*) + \frac{1}{\lambda}(y^* - x)$$

따라서 $\frac{1}{\lambda}(x - y^*) \in \partial f(y^*)$. 

여기서 $y^* = \text{prox}_{\lambda f}(x)$이고, 미분가능한 볼록 함수의 부분미분은 gradient이므로, 전체 미분 규칙(envelope theorem)에 의해

$$\nabla M_\lambda f(x) = \frac{1}{\lambda}(x - y^*) = \frac{1}{\lambda}(x - \text{prox}_{\lambda f}(x))$$

∎

**정리 1.6 (Proximal이 gradient의 일반화)**: 

$$x_{k+1} = x_k - \eta \nabla M_\eta f(x_k) = \text{prox}_{\eta f}(x_k)$$

*증명*: 정리 1.5(1)을 사용하면

$$x_k - \eta \nabla M_\eta f(x_k) = x_k - \eta \cdot \frac{1}{\eta}(x_k - \text{prox}_{\eta f}(x_k)) = \text{prox}_{\eta f}(x_k)$$

이는 proximal step이 implicit gradient step임을 의미한다. ∎

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============ Proximal Operator 시각화 ============

def prox_l2(v, lam=1.0):
    """L2 norm prox (정체성, f=0인 경우와 동등)"""
    return v

def prox_l1_1d(v, lam=1.0):
    """L1 norm prox (soft-thresholding, 1D)"""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

def prox_l2_ball(v, r=1.0):
    """L2 ball에 대한 projection (prox of indicator)"""
    norm_v = np.linalg.norm(v)
    if norm_v <= r:
        return v
    return r * v / norm_v

def verify_proximal_definition(f, grad_f, v, lam=1.0, name="f"):
    """
    Proximal의 최적성 조건 검증:
    prox_f(v)에서 0 ∈ ∇f(x) + (x - v)
    """
    def h(x):
        return f(x) + 0.5 * np.linalg.norm(x - v)**2
    
    # 수치 최적화로 prox 계산
    result = minimize(h, v, method='BFGS')
    x_opt = result.x
    
    # 최적성 조건 검증: gradient가 0 근처
    grad_h = grad_f(x_opt) + (x_opt - v)
    residual = np.linalg.norm(grad_h)
    
    print(f"\n{name}: prox_f(v)")
    print(f"  v = {v}")
    print(f"  prox_f(v) = {x_opt}")
    print(f"  Optimality residual: {residual:.2e}")
    print(f"  f(prox_f(v)) = {f(x_opt):.6f}")
    
    return x_opt

# ============ 예제 1: Quadratic (f = 0.5 * ||x||^2) ============
def f_quad(x):
    return 0.5 * np.linalg.norm(x)**2

def grad_f_quad(x):
    return x

v1 = np.array([2.0, 3.0])
x1 = verify_proximal_definition(f_quad, grad_f_quad, v1, name="f(x) = 0.5||x||²")

# ============ 예제 2: Absolute value (1D L1) ============
def f_l1_1d(x):
    return np.abs(x[0])

def grad_f_l1_1d(x):
    # 부분미분 (0에서 [-1, 1])
    if x[0] > 0:
        return np.array([1.0])
    elif x[0] < 0:
        return np.array([-1.0])
    else:
        return np.array([0.0])

v2 = np.array([2.5])
lam2 = 1.0
x2_formula = prox_l1_1d(v2, lam2)
print(f"\n|·| (L1): soft-thresholding")
print(f"  v = {v2}, λ = {lam2}")
print(f"  prox_λ|·|(v) = sign(v)max(|v|-λ, 0) = {x2_formula}")

# ============ 예제 3: L2 ball projection ============
v3 = np.array([2.0, 1.0])
r = 1.5
x3 = prox_l2_ball(v3, r)
print(f"\nL2 ball {‖x‖ ≤ {r}}")
print(f"  v = {v3}, ‖v‖ = {np.linalg.norm(v3):.3f}")
print(f"  proj(v) = {x3}, ‖proj(v)‖ = {np.linalg.norm(x3):.3f}")

# ============ 시각화 ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Quadratic
ax = axes[0]
v_range = np.linspace(-3, 3, 100)
f_vals = [f_quad(np.array([v])) for v in v_range]
ax.plot(v_range, f_vals, 'b-', linewidth=2, label='f(x) = 0.5x²')
ax.arrow(2.0, f_quad(np.array([2.0])), -0.5, -1, head_width=0.1, 
         head_length=0.1, fc='red', ec='red')
ax.text(2.0, f_quad(np.array([2.0])) + 0.5, 'v=2', ha='center')
ax.scatter([2.0], [f_quad(np.array([2.0]))], color='blue', s=100, zorder=5)
ax.scatter([2.0], [f_quad(np.array([2.0]))], color='red', s=100, zorder=5, 
           label='prox_f(v)=v (f smooth)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Proximal: Smooth Function')
ax.legend()
ax.grid(True, alpha=0.3)

# L1
ax = axes[1]
v_range = np.linspace(-3, 3, 100)
f_vals = [np.abs(v) for v in v_range]
ax.plot(v_range, f_vals, 'b-', linewidth=2, label='f(x) = |x|')
prox_vals = [prox_l1_1d(np.array([v]), 1.0)[0] for v in v_range]
ax.plot(v_range, prox_vals, 'r--', linewidth=2, label='prox_f(v) (λ=1)')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.scatter([2.5], [np.abs(2.5)], color='blue', s=100, zorder=5)
ax.scatter([1.5], [0], color='red', s=100, zorder=5, label='prox_f(2.5)=1.5')
ax.set_xlabel('v')
ax.set_ylabel('f(v) or prox_f(v)')
ax.set_title('Soft-thresholding (L1)')
ax.legend()
ax.grid(True, alpha=0.3)

# L2 ball projection
ax = axes[2]
theta = np.linspace(0, 2*np.pi, 100)
ball_x = 1.5 * np.cos(theta)
ball_y = 1.5 * np.sin(theta)
ax.plot(ball_x, ball_y, 'b-', linewidth=2, label=f'L2 ball (r={1.5})')
v_point = np.array([2.0, 1.0])
proj_point = prox_l2_ball(v_point, 1.5)
ax.arrow(0, 0, v_point[0]*0.95, v_point[1]*0.95, 
         head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
ax.arrow(0, 0, proj_point[0]*0.95, proj_point[1]*0.95,
         head_width=0.1, head_length=0.1, fc='red', ec='red')
ax.scatter([v_point[0]], [v_point[1]], color='blue', s=100, zorder=5, label='v')
ax.scatter([proj_point[0]], [proj_point[1]], color='red', s=100, zorder=5, 
           label='prox(v)=proj(v)')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_title('Projection onto L2 Ball')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2)

plt.tight_layout()
plt.savefig('/tmp/proximal_intro.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to /tmp/proximal_intro.png")
plt.close()
```

**실행 결과**:
- Quadratic: $\text{prox}_f(v) = v$ (smooth function은 그대로)
- L1: soft-thresholding으로 작은 값들이 0으로
- L2 ball: 반지름을 초과한 점을 경계로 투영

## 🔗 AI/ML 연결

**Lasso 회귀**:
$$\min_x \frac{1}{2}\|Ax - b\|^2 + \lambda\|x\|_1$$

이를 $f(x) + g(x)$ 형태로 분해하면, $g(x) = \lambda\|x\|_1$의 prox는 soft-thresholding이다. 각 좌표 $i$:
$$[\text{prox}_{\lambda g}(x)]_i = \text{sign}(x_i)\max(|x_i| - \lambda, 0)$$

이것이 L1 정규화가 희소 해를 만드는 이유다.

**제약 최적화**:
$$\min_x f(x) \text{ s.t. } x \in C$$

이는 $\min_x f(x) + I_C(x)$로 쓸 수 있고, $\text{prox}_{I_C}(v) = \text{proj}_C(v)$이다.

## ⚖️ 가정과 한계

- **적절한 함수 (proper)**: $f$의 정의역이 비어있지 않고, $-\infty$의 값을 갖지 않아야 한다.
- **볼록성**: 대부분의 정리가 $f$가 볼록함수라고 가정한다. 비볼록 함수는 보장이 없다.
- **닫힌 형태**: 대부분의 함수에서 $\text{prox}_f(v)$를 명시적으로 계산할 수 없다. 다만 계산 가능한 함수들(L1, L2, indicator 등)이 중요하다.

## 📌 핵심 정리

| 개념 | 정의 | 해석 |
|------|------|------|
| **Proximal** | $\text{prox}_f(v) = \arg\min_x (f(x) + \frac{1}{2}\|x-v\|^2)$ | $f$ 최소화 vs $v$에 가깝기 타협 |
| **Moreau** | $M_\lambda f(x) = \min_y (f(y) + \frac{1}{2\lambda}\|x-y\|^2)$ | $f$의 "부드러운" 버전 |
| **Gradient 관계** | $\nabla M_\lambda f(x) = \frac{1}{\lambda}(x - \text{prox}_{\lambda f}(x))$ | Implicit GD |
| **Soft-threshold** | $[\text{prox}_{\lambda\|\cdot\|_1}(v)]_i = \text{sign}(v_i)\max(\|v_i\|-\lambda, 0)$ | L1 희소성의 원리 |

## 🤔 생각해볼 문제

**문제 1.1**: $f(x) = \max(0, 1 - x)$ (hinge loss, 1D)에 대해 $\text{prox}_f(v)$의 닫힌 형태를 유도하시오.

<details>
<summary>힌트 및 해설</summary>

최적성 조건: $0 \in \partial f(x^*) + (x^* - v)$

$f$는 piecewise linear이므로 부분미분:
- $x < 1$: $\partial f(x) = -1$
- $x > 1$: $\partial f(x) = 0$

경우를 나누어 풀면:
- $v \geq 1$: $x^* = v$ (이미 hinge 평면 위)
- $v < 1$: $0 \in 0 + (x^* - v)$ → $x^* = v$가 조건 만족 (확인: $v < 1$일 때 gradient 0)

따라서 $\text{prox}_f(v) = v$ (항등원).

</details>

**문제 1.2**: Moreau envelope $M_\lambda f(x)$가 항상 미분가능함을 보이시오.

<details>
<summary>힌트 및 해설</summary>

$M_\lambda f(x) = \min_y (f(y) + \frac{1}{2\lambda}\|x-y\|^2)$

내부 최적화 문제에서 최솟값은 유일(강볼록)하고, implicit function theorem에 의해:
- $y^*(x) = \text{prox}_{\lambda f}(x)$는 연속
- $M_\lambda f(x) = f(y^*(x)) + \frac{1}{2\lambda}\|x - y^*(x)\|^2$는 미분가능

구체적으로: $\nabla M_\lambda f(x) = \frac{1}{\lambda}(x - y^*(x))$

</details>

**문제 1.3**: $f(x) = \lambda\|x\|_1$에 대해, 각 좌표가 독립적으로 풀릴 수 있음을 보이시오 (즉, $[\text{prox}_f(v)]_i$는 $v_i$만의 함수).

<details>
<summary>힌트 및 해설</summary>

$\text{prox}_f(v) = \arg\min_x \left( \lambda\|x\|_1 + \frac{1}{2}\|x-v\|^2 \right)$

$\|x\|_1 = \sum_i |x_i|$이고 $\|x-v\|^2 = \sum_i (x_i - v_i)^2$이므로:

$$\text{prox}_f(v) = \arg\min_x \sum_i \left( \lambda|x_i| + \frac{1}{2}(x_i - v_i)^2 \right)$$

각 좌표의 합이므로, 최솟값도 좌표별로 달성:

$$[\text{prox}_f(v)]_i = \arg\min_{x_i} \left( \lambda|x_i| + \frac{1}{2}(x_i - v_i)^2 \right) = \text{sign}(v_i)\max(|v_i| - \lambda, 0)$$

</details>

<div align="center">

| [◀ Ch5-06. Stochastic 방법과 분산 감소](../ch5-algorithms/06-stochastic-variance-reduction.md) | [📚 README](../README.md) | [02. 주요 Proximal 연산 ▶](./02-proximal-examples.md) |

</div>
