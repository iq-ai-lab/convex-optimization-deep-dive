# 4. 딥러닝은 왜 비볼록인데 동작하는가

## 🎯 핵심 질문

신경망은 비볼록 최적화 문제인데 왜 경사 하강법이 잘 작동하는가? 국소 최솟값(local minima)에 빠지지 않는가? 실제로는 전역 최솟값에 가까운 해를 찾는가?

## 🔍 왜 이 이론이 AI에서 중요한가

1. **이론과 실제의 간극**: 볼록 이론은 없지만 실제로는 잘 작동
2. **Over-parameterization 혁명**: 파라미터를 충분히 많이 하면 전역 최솟값이 풍부
3. **NTK 이론**: 무한 폭에서 신경망 = 선형 모델 + 커널
4. **Loss Landscape**: 기하학적으로 왜 극솟값이 적은가
5. **일반화 이론**: Sharp vs Flat minima의 일반화 능력 차이

## 📐 수학적 선행 조건

- **헤시안 스펙트럼**: 고유값과 조건수
- **로그 배리어(log-barrier) 함수**: 비볼록의 극단적 예
- **암묵적 정규화**: SGD의 자동 정규화 효과
- **그레디언트 흐름(gradient flow)**: 이산 GD의 연속 한계

## 📖 직관적 이해

### 신경망이 비볼록인 이유

간단한 2층 신경망: $f(x) = w_2 \sigma(w_1 x)$, 여기서 σ는 활성화 함수 (예: ReLU)

**헤시안이 부정부호:**
$$\nabla^2 f = w_2 \nabla^2 \sigma(w_1 x) + \text{고차항}$$

σ가 비볼록 (ReLU, tanh 등)이면, 전체 f의 헤시안도 부정부호 가능.

**파라미터 대칭성:**
- 두 뉴런을 바꾸면: w₁[i] ↔ w₁[j], w₂[i] ↔ w₂[j] → 같은 함수
- 이로 인해 **여러 전역 최솟값** 존재

### Over-parameterization 이론

**핵심 아이디어:**
파라미터 수 p >> 데이터 수 n이면:
- 전역 최솟값이 많음 (실제로 해집합이 고차원)
- GD가 찾는 해는 보통 "good generalization" 특성
- 과적합이 덜 일어남 (이중 강하강법 현상, double descent)

**Theorem (간단한 버전):**
충분히 많은 숨겨진 유닛이 있으면, 네트워크는 임의의 연속함수를 근사 가능 → 손실을 0에 수렴시킬 수 있는 w 존재.

### Neural Tangent Kernel (NTK)

**핵심:** 무한 폭(무한한 뉴런) 극한에서 신경망은 **선형 모델**처럼 행동한다.

**정의:**
$$K_{ij}(t) = \nabla_\theta f(x_i; \theta(t))^T \nabla_\theta f(x_j; \theta(t))$$

이는 두 입력 xᵢ, xⱼ의 "그레디언트 유사도"를 측정.

**극한:** 무한 폭에서 $K_{ij}(t) ≈ K_{ij}(0)$ (시간 불변)

따라서 학습 역학: $\frac{d\hat{y}_i}{dt} = -\eta \sum_j K_{ij} (y_i - \hat{y}_i)$

이는 **커널 회귀**와 동일!

## ✏️ 엄밀한 정의

### 비볼록성의 예: 간단한 2층 네트워크

신경망:
$$f(x; w_1, w_2) = \frac{1}{\sqrt{m}} \sum_{i=1}^m w_{2i} \sigma(w_{1i}^T x)$$

입력 x ∈ ℝ^d, 첫 번째 층 폭 m, 활성화함수 σ.

**데이터:** (x_k, y_k), k = 1,...,n

**손실:**
$$L(w) = \frac{1}{2n} \sum_{k=1}^n (f(x_k; w) - y_k)^2$$

**헤시안의 부정부호 성질:**

σ = ReLU일 때, $\nabla^2_w L$은 부정부호이다. (증명 생략, 고차원에서는 사실)

### Neural Tangent Kernel 정식화

파라미터 θ를 따라 그레디언트 흐름:
$$\frac{d\theta_t}{dt} = -\nabla_\theta L(\theta_t)$$

예측값:
$$\hat{y}_i(t) = f(x_i; \theta_t)$$

**NTK 행렬:**
$$K = \nabla_\theta f(x_1; \theta_0), \ldots, \nabla_\theta f(x_n; \theta_0)]$$

**NTK 극한 (m → ∞):**
$$\frac{d\hat{y}_i}{dt} = -\sum_{j=1}^n K_{ij}(y_i - \hat{y}_i)$$

이는 **선형 동역학**이고, 닫힌 형태의 해가 존재한다.

## 🔬 정리와 증명

### 정리 1: 무한 폭에서 NTK는 상수다

**가정:** 네트워크 폭 m → ∞, 파라미터를 적절히 초기화.

**결론:** 학습 동안 K(t) → K(0) (확률 1)

**증명 스케치:**
- 파라미터 변화량 ||θₜ - θ₀||을 추적
- Over-parameterization: 폭이 충분하면 파라미터 변화가 무시할 수 있을 정도로 작음
- 따라서 ∇_θ f ≈ ∇_θ f(0) 유지 (테일러 전개)
- 결과: K(t) ≈ K(0)

### 정리 2: NTK 수렴성

**가정:** K = K(0) (상수 NTK), K 가역.

**결론:**
$$\|y - \hat{y}(t)\|_2 \leq \|y - \hat{y}(0)\|_2 \exp(-t \lambda_{\min}(K))$$

즉, 손실이 지수 속도로 수렴한다.

**증명:**
선형 시스템: $\frac{d(y - \hat{y})}{dt} = -K(y - \hat{y})$

해: $y(t) - \hat{y}(t) = e^{-Kt}(y(0) - \hat{y}(0))$

$\|e^{-Kt}\| \leq e^{-\lambda_{\min}(K) \cdot t}$ (K PSD이므로)

### 정리 3: Loss Landscape — 극값의 연결성

**Manifold of Minima:**
고차원 에서는 손실 함수의 극솟값들이 **저손실 경로(low-loss path)**로 연결되어 있다.

즉, 두 전역 최솟값 w₁, w₂ 사이에 L(w(s)) ≈ L(w₁) for s ∈ [0, 1]인 곡선이 존재.

**의미:** 국소 극솟값이 "덫"처럼 작동하지 않음.

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# 1D 데이터로 신경망 학습 및 loss landscape 시각화
np.random.seed(42)

# 데이터: 간단한 함수 y = sin(x)
x_train = np.linspace(-np.pi, np.pi, 20)
y_train = np.sin(x_train) + np.random.randn(20) * 0.1
x_train = (x_train - x_train.mean()) / x_train.std()
y_train = (y_train - y_train.mean()) / y_train.std()

# 2층 신경망
def neural_network(x, w1, w2):
    """f(x) = w2^T * ReLU(w1^T * x)"""
    hidden = np.maximum(w1 @ x, 0)  # ReLU
    return w2 @ hidden

def network_batch(X, w1, w2):
    """배치 예측"""
    m = X.shape[1]
    hidden = np.maximum(w1 @ X, 0)  # (hidden_dim, m)
    return w2 @ hidden  # (1, m)

def loss_function(params, X, y, hidden_dim):
    """손실 함수"""
    d = X.shape[0]
    w1 = params[:d*hidden_dim].reshape(hidden_dim, d)
    w2 = params[d*hidden_dim:].reshape(1, hidden_dim)
    
    y_pred = network_batch(X, w1, w2)
    loss = np.mean((y_pred - y)**2)
    return loss

def gradient(params, X, y, hidden_dim):
    """수치 그레디언트"""
    eps = 1e-5
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (loss_function(params_plus, X, y, hidden_dim) - 
                   loss_function(params_minus, X, y, hidden_dim)) / (2 * eps)
    return grad

# 신경망 학습
hidden_dim = 5
d = 1
X_train = x_train.reshape(1, -1)
y_train_vec = y_train.reshape(1, -1)

# 초기화
w1_init = np.random.randn(hidden_dim, d) * 0.1
w2_init = np.random.randn(1, hidden_dim) * 0.1
params_init = np.concatenate([w1_init.flatten(), w2_init.flatten()])

# 최적화
result = minimize(lambda p: loss_function(p, X_train, y_train_vec, hidden_dim),
                 params_init, method='L-BFGS-B', jac=lambda p: gradient(p, X_train, y_train_vec, hidden_dim))
params_opt = result.x
w1_opt = params_opt[:hidden_dim*d].reshape(hidden_dim, d)
w2_opt = params_opt[hidden_dim*d:].reshape(1, hidden_dim)

print(f"Initial loss: {loss_function(params_init, X_train, y_train_vec, hidden_dim):.4f}")
print(f"Final loss: {loss_function(params_opt, X_train, y_train_vec, hidden_dim):.4f}")

# Loss Landscape 시각화 (2D 파라미터 공간)
fig = plt.figure(figsize=(16, 12))

# 2D Loss Surface (w1[0,0], w2[0,0] 변화)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')

w1_0_range = np.linspace(-1, 1, 50)
w2_0_range = np.linspace(-1, 1, 50)
W1_0, W2_0 = np.meshgrid(w1_0_range, w2_0_range)
Z = np.zeros_like(W1_0)

for i in range(len(w1_0_range)):
    for j in range(len(w2_0_range)):
        params_test = params_init.copy()
        params_test[0] = W1_0[j, i]
        params_test[hidden_dim*d] = W2_0[j, i]
        Z[j, i] = loss_function(params_test, X_train, y_train_vec, hidden_dim)

surf = ax1.plot_surface(W1_0, W2_0, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w1[0,0]')
ax1.set_ylabel('w2[0,0]')
ax1.set_zlabel('Loss')
ax1.set_title('Loss Landscape (2D slice)')
fig.colorbar(surf, ax=ax1)

# Contour plot
ax2 = fig.add_subplot(2, 3, 2)
contour = ax2.contour(W1_0, W2_0, Z, levels=15, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('w1[0,0]')
ax2.set_ylabel('w2[0,0]')
ax2.set_title('Loss Landscape (contour)')
ax2.grid(True, alpha=0.3)

# NTK 계산 및 시각화
def compute_ntk(X, params, hidden_dim, d):
    """Neural Tangent Kernel 계산"""
    n = X.shape[1]
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # 수치 그레디언트
            eps = 1e-4
            grad_i = np.zeros(len(params))
            grad_j = np.zeros(len(params))
            
            for k in range(len(params)):
                params_plus = params.copy()
                params_plus[k] += eps
                params_minus = params.copy()
                params_minus[k] -= eps
                
                w1_p = params_plus[:hidden_dim*d].reshape(hidden_dim, d)
                w2_p = params_plus[hidden_dim*d:].reshape(1, hidden_dim)
                w1_m = params_minus[:hidden_dim*d].reshape(hidden_dim, d)
                w2_m = params_minus[hidden_dim*d:].reshape(1, hidden_dim)
                
                y_p_i = network_batch(X[:, i:i+1], w1_p, w2_p)[0, 0]
                y_m_i = network_batch(X[:, i:i+1], w1_m, w2_m)[0, 0]
                grad_i[k] = (y_p_i - y_m_i) / (2 * eps)
                
                y_p_j = network_batch(X[:, j:j+1], w1_p, w2_p)[0, 0]
                y_m_j = network_batch(X[:, j:j+1], w1_m, w2_m)[0, 0]
                grad_j[k] = (y_p_j - y_m_j) / (2 * eps)
            
            K[i, j] = grad_i @ grad_j
    
    return K

K_init = compute_ntk(X_train, params_init, hidden_dim, d)

ax3 = fig.add_subplot(2, 3, 3)
im = ax3.imshow(K_init, cmap='coolwarm')
ax3.set_xlabel('Data index i')
ax3.set_ylabel('Data index j')
ax3.set_title('Neural Tangent Kernel (NTK)')
fig.colorbar(im, ax=ax3)

# 학습 곡선
ax4 = fig.add_subplot(2, 3, 4)
losses = []
params_current = params_init.copy()
lr = 0.01
for epoch in range(100):
    loss = loss_function(params_current, X_train, y_train_vec, hidden_dim)
    losses.append(loss)
    grad = gradient(params_current, X_train, y_train_vec, hidden_dim)
    params_current -= lr * grad

ax4.semilogy(losses, 'b-', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Learning Curve')
ax4.grid(True, alpha=0.3)

# 예측 비교
ax5 = fig.add_subplot(2, 3, 5)
x_test = np.linspace(-2, 2, 100).reshape(1, -1)
y_pred_init = network_batch(x_test, w1_init, w2_init)[0]
y_pred_opt = network_batch(x_test, w1_opt, w2_opt)[0]

ax5.plot(x_test[0], np.sin(x_test[0]), 'g-', linewidth=2, label='True: sin(x)')
ax5.scatter(x_train, y_train, c='red', s=50, label='Training data')
ax5.plot(x_test[0], y_pred_init, 'b--', linewidth=1, alpha=0.5, label='Initial network')
ax5.plot(x_test[0], y_pred_opt, 'b-', linewidth=2, label='Learned network')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('Network Fit')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 고유값 스펙트럼
ax6 = fig.add_subplot(2, 3, 6)
eigvals = np.linalg.eigvalsh(K_init)
eigvals = np.sort(eigvals)[::-1]
ax6.semilogy(range(len(eigvals)), eigvals, 'ro-', linewidth=2)
ax6.set_xlabel('Index')
ax6.set_ylabel('Eigenvalue')
ax6.set_title('NTK Eigenvalue Spectrum')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('deep_learning_nonconvex.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nNTK shape: {K_init.shape}")
print(f"NTK min eigenvalue: {np.min(eigvals):.4f}")
print(f"NTK max eigenvalue: {np.max(eigvals):.4f}")
print(f"NTK condition number: {np.max(eigvals) / np.min(eigvals):.2f}")
```

## 🔗 AI/ML 연결

1. **현대 딥러닝의 성공**: Over-parameterization이 핵심
2. **쌍대 접근**: 볼록이 아니지만 극솟값이 "좋음"
3. **암묵적 정규화**: SGD가 자동으로 smooth한 해 추구
4. **일반화 보장**: NTK 이론이 수렴성과 일반화 연결
5. **실제 효율성**: 신경망은 거의 선형처럼 동작 (특정 영역에서)

## ⚖️ 가정과 한계

| 항목 | NTK 이론 | 실제 |
|------|---------|------|
| 파라미터 수 | 무한 (m → ∞) | 유한 |
| 학습률 | 무한소 | 고정값 |
| 커널 | 고정 K(0) | 시간 변화 |
| 데이터 | 임의 | 구조있을 수 있음 |

## 📌 핵심 정리

1. **비볼록이지만 작동**: Over-parameterization이 극솟값을 "좋은" 곳에 배치
2. **NTK**: 무한 폭에서 신경망은 커널 방법
3. **Loss Landscape**: 극솟값이 저손실 경로로 연결
4. **암묵적 정규화**: SGD가 자동으로 smooth한 해 추구
5. **경험적 성공**: 이론이 따라가는 중

## 🤔 생각해볼 문제

**문제 1:** Over-parameterization이 왜 일반화를 해치지 않는가? (과적합 역설)

<details>
<summary>힌트 및 해설</summary>

**이중 강하강법 (Double Descent):**

고전 편향-분산 트레이드오프:
- 모델 복잡도 ↑ → 분산 ↑, 편향 ↓
- 어떤 지점에서 최적

그런데 극도로 많은 파라미터 (p > n):
- 다시 테스트 오류 ↓!
- 손실을 0으로 완벽히 적합 가능
- 암묵적 정규화로 부드러운 해만 찾음

이는 **암묵적 정규화** (implicit regularization)의 결과.

</details>

**문제 2:** NTK 이론이 현대 신경망 (depth >> width)에도 적용되는가?

<details>
<summary>힌트 및 해설</summary>

**한계:**
- NTK는 무한 폭을 가정 (m → ∞)
- 실제: m은 유한, 깊이는 커질 수 있음
- 깊은 네트워크는 그레디언트가 "소실" (vanishing gradients)

**개선:**
- **Tensor Programs**: 너비와 깊이 모두 큰 극한
- **스케일링 법칙**: 합리적인 파라미터 범위에서 경험적으로 작동

결론: NTK는 **부분적 설명**이지, 전체 그림이 아니다.

</details>

**문제 3:** 국소 극솟값이 나쁜 특성을 가질 가능성은?

<details>
<summary>힌트 및 해설</summary>

**현재 합의:**
- 손실 함수의 극솟값은 보통 **비슷한 손실값** 가짐
- 국소 극솟값이 전역 극솟값보다 훨씬 나쁜 경우는 드뭄
- 고차원에서는 국소 극솟값이 "거의 전역" (folklore)

**이론적 근거:**
- Spurious minima는 매우 높은 손실을 가짐 (empirically)
- 대부분의 국소 극솟값이 리지(ridge)로 연결 (mode connectivity)

**실무:** 초기값과 학습률에 따라 수렴이 다르지만, 대부분 합리적인 해.

</details>

<div align="center">

| [◀ 03. Regularization의 기하 — L1 vs L2](./03-regularization-geometry.md) | [📚 README](../README.md) | [05. Online Convex Optimization과 Regret 경계 ▶](./05-online-convex-optimization.md) |

</div>
