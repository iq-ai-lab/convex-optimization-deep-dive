# 1. Logistic Regression은 볼록이다

## 🎯 핵심 질문

Logistic Regression은 머신러닝의 가장 기본적인 분류 알고리즘이다. 그런데 왜 이것이 "최적화 관점에서 다루기 좋은" 문제일까? 수학적으로 어떤 성질 때문에 전역 최적해를 찾을 수 있을까?

## 🔍 왜 이 이론이 AI에서 중요한가

실제 머신러닝에서는 다음과 같은 문제에 직면한다:
- 분류 모델 학습 중 수렴 보장이 되는가?
- 찾은 해가 전역 최적해인가 (국소 최적해 아닌가)?
- 정규화를 추가했을 때는 어떻게 되는가?

Logistic Regression의 **볼록성**은 이 모든 질문에 긍정적 답변을 제공한다. 이는 이후 더 복잡한 모델(SVM, 신경망 등)을 이해하기 위한 기초다.

## 📐 수학적 선행 조건

- **헤시안**: f''(w) = ∇²f(w)
- **양반정부호(PSD)**: M ≽ 0 ⟺ xᵀMx ≥ 0 for all x
- **강볼록성**: ∇²f ≽ λI for some λ > 0
- **최대우도 추정(MLE)**: 확률 모델 p(y|x;w)에서 log p 최대화

## 📖 직관적 이해

Logistic Regression의 손실함수는 각 데이터 포인트에 대해 다음과 같다:

$$\ell(w) = \frac{1}{n} \sum_{i=1}^{n} \log(1 + \exp(-y_i w^T x_i))$$

여기서 $y_i \in \{-1, +1\}$은 라벨이다. 이 함수는:
1. **아래로 볼록한 곡선** (U자 모양): 한 방향으로만 최솟값을 가짐
2. **미분 가능**: 경사 하강법을 적용할 수 있음
3. **강제성(Coercive)**: ‖w‖ → ∞일 때 ℓ(w) → ∞

따라서 유일한 전역 최솟값이 **반드시** 존재한다.

## ✏️ 엄밀한 정의

### Logistic Regression 설정

이진 분류 문제: $(x_i, y_i)_{i=1}^n$, 여기서 $x_i \in \mathbb{R}^d$, $y_i \in \{-1, +1\}$

**목적함수** (로지스틱 손실):
$$\ell(w) = \frac{1}{n} \sum_{i=1}^{n} \log(1 + \exp(-y_i w^T x_i))$$

또는 $y_i \in \{0, 1\}$을 사용하면:
$$\ell(w) = \frac{1}{n} \sum_{i=1}^{n} [y_i \log \sigma(w^T x_i) + (1-y_i) \log(1-\sigma(w^T x_i))]$$

여기서 $\sigma(z) = \frac{1}{1+\exp(-z)}$는 시그모이드 함수다.

### L2 정규화

실제로는 정규화 항을 추가한다:
$$f(w) = \ell(w) + \frac{\lambda}{2} \|w\|^2$$

여기서 $\lambda \geq 0$는 정규화 계수다.

## 🔬 정리와 증명

### 정리 1: Logistic Loss는 볼록함수다

**증명:**

로지스틱 손실 $g(z) = \log(1 + \exp(-z))$의 헤시안을 계산하자.

$$g'(z) = \frac{-\exp(-z)}{1+\exp(-z)} = -\frac{1}{1+\exp(z)}$$

$$g''(z) = \frac{\exp(z)}{(1+\exp(z))^2} = \sigma(z)(1-\sigma(z))$$

$z$의 값에 상관없이 $0 < \sigma(z)(1-\sigma(z)) \leq 1/4$이다. 따라서 $g''(z) > 0$이고, $g$는 **강볼록이다**.

이제 전체 목적함수를 보자:
$$\ell(w) = \frac{1}{n} \sum_{i=1}^{n} \log(1 + \exp(-y_i w^T x_i))$$

$z_i = y_i w^T x_i$라 하면:
$$\nabla \ell(w) = \frac{1}{n} \sum_{i=1}^{n} -y_i x_i \sigma(-z_i)$$

**헤시안:**
$$\nabla^2 \ell(w) = \frac{1}{n} \sum_{i=1}^{n} x_i x_i^T \sigma(z_i)(1-\sigma(z_i))$$

$p_i = \sigma(z_i) \in (0, 1)$이므로 $p_i(1-p_i) > 0$이다. 따라서:
$$\nabla^2 \ell(w) = \frac{1}{n} X^T D X$$

여기서 $D = \text{diag}(p_1(1-p_1), \ldots, p_n(1-p_n))$는 모두 양수인 대각 원소를 가진 대각 행렬이다.

$X^T D X$는 반정부호(positive semidefinite)이다. (Xᵀ[양수]X 형태)

**따라서 ℓ(w)는 볼록함수다.** ∎

### 정리 2: L2 정규화된 Logistic Regression은 강볼록이다

**증명:**

$$f(w) = \ell(w) + \frac{\lambda}{2} \|w\|^2$$

위에서:
$$\nabla^2 f(w) = \frac{1}{n} X^T D X + \lambda I$$

$\lambda > 0$이면 $\nabla^2 f(w) \succeq \lambda I$이다.

**따라서 f는 강볼록이다.** ∎

### 정리 3: 전역 최적해의 존재과 유일성

강볼록 함수는:
- **유일한 전역 최솟값**을 가진다
- **극한점에서 발산**: ‖w‖ → ∞일 때 f(w) → ∞
- **경사 하강법 수렴**: 모든 초기값에서 유일한 최적해로 수렴

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 데이터 생성
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                          n_redundant=0, random_state=42)
y = 2*y - 1  # {0,1} -> {-1,+1}
X = StandardScaler().fit_transform(X)

# 로지스틱 손실 함수
def logistic_loss(w, X, y):
    z = y * (X @ w)
    return np.mean(np.log(1 + np.exp(-z)))

# 정규화 손실
def regularized_loss(w, X, y, lam):
    return logistic_loss(w, X, y) + (lam / 2) * np.sum(w**2)

# 그레디언트
def gradient(w, X, y, lam):
    z = y * (X @ w)
    p = 1 / (1 + np.exp(-z))
    grad_loss = -np.mean(X * (y * (1 - p))[:, None], axis=0)
    return grad_loss + lam * w

# 헤시안 검증
def hessian(w, X, y, lam):
    z = y * (X @ w)
    p = 1 / (1 + np.exp(-z))
    D = p * (1 - p)  # 대각 원소
    H_loss = (X.T * D) @ X / len(y)
    return H_loss + lam * np.eye(X.shape[1])

# 경사 하강법
def gradient_descent(X, y, lam, lr=0.1, max_iter=1000, tol=1e-6):
    w = np.zeros(X.shape[1])
    losses = [regularized_loss(w, X, y, lam)]
    
    for _ in range(max_iter):
        grad = gradient(w, X, y, lam)
        w_new = w - lr * grad
        loss_new = regularized_loss(w_new, X, y, lam)
        losses.append(loss_new)
        
        if np.abs(loss_new - losses[-2]) < tol:
            break
        w = w_new
    
    return w, np.array(losses)

# CVXPY를 통한 최적화
def cvxpy_solution(X, y, lam):
    w = cp.Variable(X.shape[1])
    z = y * (X @ w)
    objective = cp.sum(cp.logistic(-z)) / len(y) + (lam / 2) * cp.sum_squares(w)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()
    return np.array(w.value)

# 실행
lam = 0.01
w_gd, losses = gradient_descent(X, y, lam, lr=0.1, max_iter=500)
w_cvxpy = cvxpy_solution(X, y, lam)

print(f"Gradient Descent Solution: {w_gd}")
print(f"CVXPY Solution: {w_cvxpy}")
print(f"Difference: {np.linalg.norm(w_gd - w_cvxpy):.2e}")

# 헤시안이 PSD 확인
H = hessian(w_gd, X, y, lam)
eigvals = np.linalg.eigvalsh(H)
print(f"Hessian 최소 고유값: {eigvals.min():.6f} (≥ 0이면 PSD)")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 손실 수렴
axes[0].semilogy(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Objective Value')
axes[0].set_title('Convergence of Gradient Descent')
axes[0].grid(True)

# 결정 경계
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
Z = np.sign(xx * w_gd[0] + yy * w_gd[1])

axes[1].contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
axes[1].scatter(X[y==1, 0], X[y==1, 1], label='y=+1', s=50)
axes[1].scatter(X[y==-1, 0], X[y==-1, 1], label='y=-1', s=50)
axes[1].set_title('Decision Boundary')
axes[1].legend()
axes[1].grid(True)

# 헤시안 고유값
axes[2].bar(range(len(eigvals)), eigvals, color='steelblue')
axes[2].set_xlabel('Eigenvalue Index')
axes[2].set_ylabel('Eigenvalue')
axes[2].set_title('Hessian Eigenvalues (all ≥ 0)')
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.savefig('logistic_regression_convex.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Logistic Regression is convex: Hessian is PSD")
print("✓ Unique global optimum guaranteed")
```

## 🔗 AI/ML 연결

**Logistic Regression의 의미:**
1. **이진 분류**: 확률 모델 p(y=1|x) = σ(wᵀx)
2. **최대우도 추정**: 로지스틱 손실은 -log p(y|x;w)에서 유래
3. **정규화의 역할**: 
   - L2 정규화 → 강볼록 → 수치 안정성, 고속 수렴
   - 과적합 방지
4. **일반화**: SVM(여백), 신경망(첫 층)의 기초

## ⚖️ 가정과 한계

| 가정 | 의미 | 한계 |
|------|------|------|
| 선형 분리가능 | w가 존재해서 좋은 분류 가능 | 비선형 문제는 X → φ(X) 변환 필요 |
| 독립 표본 | i.i.d. 가정 | 시계열, 종속 데이터는 다른 기법 필요 |
| 로지스틱 모델 | 확률 = 1/(1+exp(-wᵀx)) | 실제 분포가 다르면 성능 저하 |

## 📌 핵심 정리

1. **로지스틱 손실은 강볼록**: g''(z) = σ(z)(1-σ(z)) > 0
2. **헤시안 PSD**: ∇²ℓ = (1/n)XᵀDX where D ≽ 0
3. **전역 최적해 유일**: 강볼록 + 연속 → 유일 최솟값
4. **L2 정규화 강화**: ∇²f ≽ λI for f(w) = ℓ(w) + (λ/2)‖w‖²
5. **효율적 학습**: GD, Newton, SGD 모두 수렴 보장

## 🤔 생각해볼 문제

**문제 1:** L1 정규화 logistic regression f(w) = ℓ(w) + λ‖w‖₁은 강볼록인가? 왜?

<details>
<summary>힌트 및 해설</summary>

L1 노름 ‖w‖₁ = Σ|wⱼ|은 미분불가능한 점이 있다 (e.g., w=0에서).
따라서 전체 f는 미분불가능하지만, **여전히 볼록이다**.

헤시안 대신 **부분미분(subdifferential)**을 사용해야 한다.
∂‖w‖₁ = {g : gⱼ ∈ sign(wⱼ) if wⱼ≠0, gⱼ∈[-1,1] if wⱼ=0}

결론: 강볼록은 아니지만 (미분 불가능), 여전히 **유일 최솟값을 가질 수 있다**.

</details>

**문제 2:** X의 열벡터들이 선형종속이면 (e.g., 중복된 특징), Logistic Regression은 어떻게 되는가?

<details>
<summary>힌트 및 해설</summary>

이 경우 ∇²ℓ = (1/n)XᵀDX는 **반정부호(PSD)이지만 정부호는 아니다** (singular).

따라서:
- ℓ(w)는 여전히 **볼록**이지만
- **최솟값이 유일하지 않다** (최솟값을 달성하는 w들의 집합이 affine subspace)
- L2 정규화를 추가하면 ∇²f = XᵀDX + λI ≻ 0 → 다시 강볼록!

**실무:** 다중공선성이 있으면 L2 정규화 필수.

</details>

**문제 3:** 경사 하강법의 수렴 속도는? 어떤 조건에서 더 빠른가?

<details>
<summary>힌트 및 해설</summary>

강볼록 함수 f(w)에 대해 학습률 η = 1/L (L = Lipschitz 상수)일 때:

$$\|w_k - w^*\| \leq (1-\mu/L)^k \|w_0 - w^*\|$$

여기서 μ = ∇²f의 최소 고유값 (강볼록 계수).

조건수(condition number) κ = L/μ가 작을수록 (즉, 데이터가 "잘 조건화"되면) 수렴이 빠르다.

**개선 방법:**
1. **정규화 증가** (λ 증대) → μ 증대 → κ 감소
2. **데이터 스케일링** (정규화) → L 감소 → κ 감소

</details>

<div align="center">

| [◀ Ch6-06. Douglas-Rachford, Primal-Dual Splitting](../ch6-proximal/06-operator-splitting.md) | [📚 README](../README.md) | [02. Support Vector Machine의 완전 유도 ▶](./02-svm-complete.md) |

</div>
