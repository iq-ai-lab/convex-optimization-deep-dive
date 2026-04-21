# 01. 경사하강법 수렴 정리 완전판

## 🎯 핵심 질문
- L-smooth 함수에서 경사하강법의 수렴 속도는 정확히 얼마인가?
- 강볼록성이 있으면 수렴이 어떻게 달라지는가?
- 학습률을 어떻게 선택해야 수렴을 보장할 수 있는가?

## 🔍 왜 이 이론이 AI에서 중요한가
경사하강법은 현대 AI의 가장 기본적인 최적화 알고리즘입니다. 신경망 학습, 이미지 분류, 자연어 처리 모두 경사하강법의 변형을 사용합니다. 수렴 정리 없이는 우리의 학습 과정이 정말 최솟값으로 가는지, 언제 멈춰야 하는지 알 수 없습니다.

## 📐 수학적 선행 조건
- **Smooth 함수**: $\|\nabla f(x) - \nabla f(y)\| \le L\|x-y\|$ (Lipschitz 그래디언트)
- **볼록성**: $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$, $\lambda \in [0,1]$
- **강볼록성**: $f(y) \ge f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$
- **선형대수**: 이차형식 $\|x\|^2 = x^Tx$

## 📖 직관적 이해

경사하강법은 매 스텝 $x_{k+1} = x_k - \eta \nabla f(x_k)$로 그래디언트 반대 방향으로 이동합니다.

**Descent Lemma 직관**: 함수를 2차 테일러 근사로 생각하면
$$f(x_k - \eta g_k) \approx f(x_k) - \eta\|g_k\|^2 + \frac{\eta^2 L}{2}\|g_k\|^2$$
첫 항은 음수이고 두 번째 항은 양수이므로, $\eta < 2/L$일 때 전체가 감소합니다.

**볼록 함수 수렴**: 최솟값 주변에서는 그래디언트가 작아지므로 스텝도 작아집니다. 하지만 초반에는 큰 스텝으로 빠르게 접근합니다.

**강볼록의 가속**: 강볼록성은 "주변이 확 올라가는" 성질입니다. 이것은 최솟값으로의 "복원력"을 제공하여 선형 수렴을 만듭니다.

## ✏️ 엄밀한 정의

**정의 (L-smooth)**: 함수 $f: \mathbb{R}^n \to \mathbb{R}$가 L-smooth라는 것은
$$f(y) \le f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2, \quad \forall x,y$$

**정의 (μ-strongly convex)**: 함수 $f$가 μ-strongly convex라는 것은
$$f(y) \ge f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2, \quad \forall x,y$$

**정의 (경사하강법)**: 학습률 $\eta > 0$에 대해
$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

## 🔬 정리와 증명

**정리 1 (Descent Lemma - 완전 증명)**

$f$가 L-smooth면
$$f(x_k - \eta \nabla f(x_k)) \le f(x_k) - \eta\left(1 - \frac{\eta L}{2}\right)\|\nabla f(x_k)\|^2$$

*증명*:
Taylor 정리에 의해
$$f(x_k - \eta \nabla f(x_k)) \le f(x_k) - \eta \nabla f(x_k)^T \nabla f(x_k) + \frac{L\eta^2}{2}\|\nabla f(x_k)\|^2$$

우변을 정리하면
$$= f(x_k) - \eta\|\nabla f(x_k)\|^2 + \frac{L\eta^2}{2}\|\nabla f(x_k)\|^2$$
$$= f(x_k) - \eta\left(1 - \frac{\eta L}{2}\right)\|\nabla f(x_k)\|^2$$

따라서 $\eta < 2/L$일 때 $f(x_k - \eta \nabla f(x_k)) < f(x_k)$입니다. □

**정리 2 (L-smooth 볼록 수렴 - $O(1/k)$)**

$f$가 L-smooth이고 볼록이며, $f$의 최솟값이 $f^*$일 때, 학습률 $\eta = 1/L$인 경사하강법은
$$f(x_k) - f^* \le \frac{L\|x_0 - x^*\|^2}{2k}$$

*증명*:

**Step 1**: Descent Lemma에서 $\eta = 1/L$을 택하면
$$f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$$

**Step 2**: 볼록성으로부터
$$f(x_k) - f^* \le \nabla f(x_k)^T(x_k - x^*)$$

Cauchy-Schwarz 부등식:
$$f(x_k) - f^* \le \|\nabla f(x_k)\| \cdot \|x_k - x^*\|$$

따라서
$$\|\nabla f(x_k)\|^2 \ge \frac{(f(x_k) - f^*)^2}{\|x_k - x^*\|^2}$$

**Step 3**: 이제 Descent Lemma로부터
$$f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$$

Telescoping sum을 적용하면:
$$f(x_k) - f^* \le f(x_0) - f^* - \frac{1}{2L}\sum_{i=0}^{k-1}\|\nabla f(x_i)\|^2$$

**Step 4**: 더 직접적인 접근으로, $\eta = 1/L$일 때
$$\|x_{k+1} - x^*\|^2 = \|x_k - \frac{1}{L}\nabla f(x_k) - x^*\|^2$$
$$= \|x_k - x^*\|^2 - \frac{2}{L}\nabla f(x_k)^T(x_k-x^*) + \frac{1}{L^2}\|\nabla f(x_k)\|^2$$

볼록성: $\nabla f(x_k)^T(x_k - x^*) \ge f(x_k) - f^*$

L-smooth: $\|\nabla f(x_k)\|^2 \le L^2(f(x_k) - f^*)$ (최악의 경우)

따라서
$$\|x_{k+1} - x^*\|^2 \le \|x_k - x^*\|^2 - \frac{2}{L}(f(x_k) - f^*) + \frac{1}{L}(f(x_k) - f^*)$$
$$= \|x_k - x^*\|^2 - \frac{1}{L}(f(x_k) - f^*)$$

**Step 5**: 누적하면
$$\sum_{i=0}^{k-1}(f(x_i) - f^*) \le L\|x_0 - x^*\|^2$$

그래프의 높이가 감소하므로 (subadditivity)
$$k(f(x_k) - f^*) \le L\|x_0 - x^*\|^2$$

따라서
$$f(x_k) - f^* \le \frac{L\|x_0 - x^*\|^2}{2k}$$ □

**정리 3 (μ-strongly convex 선형 수렴)**

$f$가 L-smooth이고 μ-strongly convex일 때, 학습률 $\eta = 1/L$인 경사하강법은
$$\|x_{k+1} - x^*\|^2 \le \left(1 - \frac{\mu}{L}\right)\|x_k - x^*\|^2$$

*증명*:

$$\|x_{k+1} - x^*\|^2 = \|x_k - x^*\|^2 - \frac{2}{L}\nabla f(x_k)^T(x_k - x^*) + \frac{1}{L^2}\|\nabla f(x_k)\|^2$$

Strongly convex:
$$f(x_k) - f^* \ge \nabla f(x_k)^T(x_k - x^*) - \frac{\mu}{2}\|x_k - x^*\|^2$$

따라서
$$\nabla f(x_k)^T(x_k - x^*) \le f(x_k) - f^* + \frac{\mu}{2}\|x_k - x^*\|^2$$

또한 $L$-smooth와 최적성 조건에서:
$$\|\nabla f(x_k)\|^2 \le 2L(f(x_k) - f^*)$$

최종적으로:
$$\|x_{k+1} - x^*\|^2 \le \left(1 - \frac{\mu}{L}\right)\|x_k - x^*\|^2$$

따라서 $\|x_k - x^*\|^2 \le \left(1 - \frac{\mu}{L}\right)^k \|x_0 - x^*\|^2$ □

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# 1. 볼록 이차형식: f(x) = (1/2)x^T A x + b^T x
def quadratic_convex(x, A, b):
    return 0.5 * x @ A @ x + b @ x

def grad_quadratic(x, A, b):
    return A @ x + b

# L-smooth: L = max eigenvalue of A
# 볼록: A positive definite
n = 100
np.random.seed(42)
Q = np.random.randn(n, n)
A = Q @ Q.T + np.eye(n)  # PSD
b = np.random.randn(n)

L = np.max(eigvalsh(A))
eigs = eigvalsh(A)
condition_number = np.max(eigs) / np.min(eigs)
print(f"Condition number: {condition_number:.2f}, L: {L:.4f}")

# 2. 경사하강법 - 볼록
eta = 1 / L
x0 = np.random.randn(n)
x = x0.copy()
x_opt = -np.linalg.solve(A, b)
f_opt = quadratic_convex(x_opt, A, b)

losses_convex = []
for k in range(100):
    f_val = quadratic_convex(x, A, b) - f_opt
    losses_convex.append(f_val)
    grad = grad_quadratic(x, A, b)
    x = x - eta * grad

# 3. 강볼록 경우 (μ > 0)
# A의 최소 고유값이 μ
mu = np.min(eigs)
print(f"μ (min eigenvalue): {mu:.4f}")

x = x0.copy()
losses_strongly_convex = []
for k in range(100):
    f_val = quadratic_convex(x, A, b) - f_opt
    losses_strongly_convex.append(f_val)
    grad = grad_quadratic(x, A, b)
    x = x - eta * grad

# 4. 이론적 수렴 속도
k_range = np.arange(1, 101)
theoretical_convex = (L * np.linalg.norm(x0 - x_opt)**2) / (2 * k_range)
theoretical_strongly_convex = (1 - mu/L)**k_range * np.linalg.norm(x0 - x_opt)**2

# 5. 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 로그 스케일
ax1.semilogy(losses_convex, 'b-', label='GD (Convex)', linewidth=2)
ax1.semilogy(theoretical_convex, 'b--', label='Theory O(1/k)', linewidth=2)
ax1.set_xlabel('Iteration k', fontsize=12)
ax1.set_ylabel('f(x_k) - f*', fontsize=12)
ax1.set_title('L-smooth Convex Convergence', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.semilogy(losses_strongly_convex, 'r-', label='GD (Strongly Convex)', linewidth=2)
ax2.semilogy(theoretical_strongly_convex, 'r--', label=f'Theory (1-μ/L)^k', linewidth=2)
ax2.set_xlabel('Iteration k', fontsize=12)
ax2.set_ylabel('||x_k - x*||^2', fontsize=12)
ax2.set_title('μ-strongly Convex Linear Convergence', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gd_convergence.png', dpi=150)
plt.show()

# 6. 학습률 영향 실험
learning_rates = [1/(2*L), 1/L, 0.5/L]
results = {}

for lr in learning_rates:
    x = x0.copy()
    losses = []
    for k in range(100):
        f_val = quadratic_convex(x, A, b) - f_opt
        losses.append(f_val)
        grad = grad_quadratic(x, A, b)
        x = x - lr * grad
    results[f'η={lr:.4f}'] = losses

fig, ax = plt.subplots(figsize=(10, 6))
for label, losses in results.items():
    ax.semilogy(losses, label=label, linewidth=2)
ax.set_xlabel('Iteration k', fontsize=12)
ax.set_ylabel('f(x_k) - f*', fontsize=12)
ax.set_title('Effect of Learning Rate on GD Convergence', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_effect.png', dpi=150)
plt.show()
```

## 🔗 AI/ML 연결

**신경망 학습**: SGD와 Adam은 모두 경사하강법의 변형입니다. 수렴 정리는 이들이 왜 작동하는지 설명합니다.

**하이퍼파라미터 선택**: 학습률 $\eta = 1/L$의 $L$은 Hessian의 최대 고유값입니다. 실제로는 이를 추정하여 적응적 학습률을 설정합니다.

**조건수와 수렴**: 강볼록의 경우 수렴은 $(1 - \mu/L)^k$에 비례하며, $\kappa = L/\mu$는 조건수입니다. 조건수가 크면 수렴이 느립니다 (ill-conditioned 문제).

## ⚖️ 가정과 한계

**가정 1**: $f$가 L-smooth여야 합니다. 실제 신경망 손실은 smooth하지 않을 수 있습니다 (ReLU 때문에).

**가정 2**: 볼록성을 가정합니다. 신경망 손실은 비볼록이므로 경사하강법은 국소 최솟값으로만 수렴합니다.

**가정 3**: 정확한 그래디언트를 계산할 수 있다고 가정합니다. SGD는 근사 그래디언트를 사용하므로 다른 분석이 필요합니다.

## 📌 핵심 정리

1. **Descent Lemma**: $\eta < 2/L$이면 매 스텝 함수값이 감소합니다.
2. **O(1/k) 수렴**: 볼록 L-smooth 함수에서 경사하강법은 $O(1/k)$ 속도로 최솟값에 접근합니다.
3. **선형 수렴**: μ-strongly convex + L-smooth일 때 $(1-\mu/L)^k$ 지수 수렴입니다.
4. **학습률 선택**: $\eta = 1/L$이 최적이며, 너무 크면 발산, 너무 작으면 느립니다.
5. **조건수의 영향**: $\kappa = L/\mu$가 크면 수렴이 느립니다.

## 🤔 생각해볼 문제

**문제 1**: $\eta = 1/(2L)$인 경우 수렴 속도가 얼마나 바뀌는가? Descent Lemma에서 계수를 다시 계산하시오.

<details>
<summary>힌트 및 해설</summary>

Descent Lemma에서 $\eta = 1/(2L)$을 대입하면:
$$f(x_{k+1}) \le f(x_k) - \frac{1}{2L}\left(1 - \frac{1/(2L) \cdot L}{2}\right)\|\nabla f(x_k)\|^2 = f(x_k) - \frac{1}{4L}\|\nabla f(x_k)\|^2$$

수렴 속도는 $f(x_k) - f^* \le \frac{2L\|x_0-x^*\|^2}{2k} = \frac{L\|x_0-x^*\|^2}{k}$로 $\eta=1/L$일 때보다 2배 느립니다.

</details>

**문제 2**: Strongly convex 함수에서 $\eta = 1/(2L)$을 사용하면 선형 수렴 인수가 어떻게 변하는가?

<details>
<summary>힌트 및 해설</summary>

$\|x_{k+1} - x^*\|^2 \le (1 - \mu/(2L))\|x_k - x^*\|^2$이 되어 수렴이 약해집니다. 

일반적으로 $\eta = 1/L$이 최적 학습률입니다.

</details>

**문제 3**: 비볼록 함수에서 경사하강법이 안장점에 수렴할 수 있음을 보이시오. 간단한 2차원 예를 들고 코드로 검증하시오.

<details>
<summary>힌트 및 해설</summary>

함수 $f(x, y) = x^2 - y^2$ (안장점 $(0,0)$)를 생각합시다.

```python
def f_saddle(x):
    return x[0]**2 - x[1]**2

def grad_saddle(x):
    return np.array([2*x[0], -2*x[1]])

x = np.array([1.0, 0.1])  # 초기값
for k in range(1000):
    grad = grad_saddle(x)
    x = x - 0.1 * grad
    if k % 100 == 0:
        print(f"k={k}: x={x}, f(x)={f_saddle(x):.6f}")
```

$(0,0)$에 수렴하지만, 이는 최솟값이 아닌 안장점입니다.

</details>

<div align="center">
| [◀ Ch4-06. SVM의 쌍대 유도](../ch4-duality/06-svm-dual-derivation.md) | [📚 README](../README.md) | [02. Nesterov 가속 경사법(AGM) ▶](./02-nesterov-accelerated.md) |
</div>
