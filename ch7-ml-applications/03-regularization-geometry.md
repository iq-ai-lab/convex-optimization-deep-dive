# 3. Regularization의 기하 — L1 vs L2

## 🎯 핵심 질문

Ridge regression (L2)과 Lasso (L1)는 같은 정규화인데 왜 결과가 다를까? Lasso는 왜 자동으로 변수 선택(sparsity)을 하는가? 이것을 기하학적으로 어떻게 이해할 수 있을까?

## 🔍 왜 이 이론이 AI에서 중요한가

1. **변수 선택**: 고차원 데이터(n << p)에서 중요한 특징 선택
2. **해석 가능성**: Lasso는 일부 계수를 정확히 0으로 → 선택된 특징만 사용
3. **계산 효율**: Sparse 해는 메모리와 속도 향상
4. **통계 이론**: 일관된 모델 선택(consistent model selection)
5. **기하 직관**: 최적화를 시각적으로 이해

## 📐 수학적 선행 조건

- **노름(norm)**: ‖·‖₁, ‖·‖₂, ‖·‖∞
- **제약 조건의 기하**: 볼(ball)과 다면체(polyhedron)
- **부분미분(subdifferential)**: 미분 불가능한 함수 다루기
- **등고선(contour)**: 목적함수의 기하적 표현

## 📖 직관적 이해

### Ridge Regression (L2)

제약 형태:
$$\min_w f(w) \quad \text{s.t.} \quad \|w\|_2 \leq r$$

여기서 $f$는 손실함수 (e.g., 제곱 오차), $r$은 반경.

**기하:**
- 제약 집합: **구(sphere)** $\{w : \|w\|_2 \leq r\}$
- 손실의 등고선: 동심원
- 최적해: 등고선과 구의 **접점**

접점은 **아무 곳이나** 가능하므로 (구의 표면 어디든), 일반적으로 **모든 wⱼ ≠ 0**.

### Lasso (L1)

제약 형태:
$$\min_w f(w) \quad \text{s.t.} \quad \|w\|_1 \leq r$$

여기서 $\|w\|_1 = \sum_j |w_j|$.

**기하:**
- 제약 집합: **마름모(diamond)** (2D) 또는 교차 다면체(cross-polytope) (d-D)
- 손실의 등고선: 동심원 (L2와 같음)
- 최적해: 등고선과 마름모의 **접점**

마름모의 **꼭짓점(vertex)**은 축(axis)에 정렬되어 있다! (e.g., 2D에서 (r,0), (0,r) 등)

따라서 최적해가 꼭짓점에서 나타나면 → **일부 wⱼ = 0 (정확히!)**

## ✏️ 엄밀한 정의

### 문제 설정

Loss function: $f(w) = \frac{1}{2}\|X w - y\|^2$ (선형 회귀)

#### Ridge Regression
$$\min_w \left\{ \frac{1}{2}\|Xw - y\|^2 + \frac{\lambda}{2}\|w\|_2^2 \right\}$$

또는 제약 형태:
$$\min_w \|Xw - y\|^2 \quad \text{s.t.} \quad \|w\|_2 \leq r$$

#### Lasso
$$\min_w \left\{ \frac{1}{2}\|Xw - y\|^2 + \lambda\|w\|_1 \right\}$$

또는 제약 형태:
$$\min_w \|Xw - y\|^2 \quad \text{s.t.} \quad \|w\|_1 \leq r$$

#### Elastic Net (혼합)
$$\min_w \left\{ \frac{1}{2}\|Xw - y\|^2 + \alpha\lambda\|w\|_1 + \frac{(1-\alpha)\lambda}{2}\|w\|_2^2 \right\}$$

## 🔬 정리와 증명

### 정리 1: Lasso의 Sparsity (KKT 조건을 통한 증명)

Lasso 제약 형태에서 최적해 $w^*$에 대해, 부분미분(subdifferential)의 KKT 조건:

$$X^T(Xw^* - y) \in \lambda \partial\|w^*\|_1$$

여기서 $\partial\|w\|_1$은 $\|w\|_1$의 부분미분:
$$(\partial\|w\|_1)_j = \begin{cases} \text{sign}(w_j) & \text{if } w_j \neq 0 \\ [-1, 1] & \text{if } w_j = 0 \end{cases}$$

**만약** $w_j^* = 0$이고, 최적성 조건에서
$$|(X^T(Xw^* - y))_j| < \lambda$$

그러면 $w_j$는 **0으로 유지**된다. (2계 이차 조건으로 증명 가능)

**결론:** Lasso는 충분히 작은 계수를 정확히 0으로 축소. ∎

### 정리 2: Ridge의 연속 축소

Ridge regression의 폐쇄해:
$$w^*_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T y$$

비교: OLS 해는 $w^*_{\text{OLS}} = (X^T X)^{-1} X^T y$

**특이값 분해:** $X = U \Sigma V^T$라 하면,

$$w^*_{\text{Ridge}} = V \left( \Sigma^T \Sigma + \lambda I \right)^{-1} \Sigma^T U^T y$$

OLS와의 비율:
$$\frac{w^*_{\text{Ridge}, j}}{w^*_{\text{OLS}, j}} = \frac{\sigma_j^2}{\sigma_j^2 + \lambda}$$

각 항은 $[0, 1)$ 범위이며, **균등하게** 축소 (항상 0이 아님). ∎

### 정리 3: 고차원에서의 일관된 모델 선택 (Lasso)

**가정:** 진정한 모델은 sparse: $y = X_S w_S^* + \epsilon$, 여기서 $|S| = s << p$.

충분히 큰 n에서, Lasso는:
1. **정확한 모델 식별**: $\text{supp}(w^*_{\text{Lasso}}) = S$ (확률 1로)
2. **매개변수 추정의 수렴률**: $\|w^*_{\text{Lasso}} - w^*\|_2 = O_p(\sqrt{s \log p / n})$

Ridge는 이런 보장이 없음 (모든 계수가 non-zero). ∎

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# 2D 데이터: 시각화를 위해
np.random.seed(42)
n, d = 50, 2
X = np.random.randn(n, d)
w_true = np.array([3.0, 1.0])
y = X @ w_true + np.random.randn(n) * 0.1
X = StandardScaler().fit_transform(X)
y = (y - y.mean()) / y.std()

# Ridge Regression (CVXPY)
def ridge_regression(X, y, lam):
    w = cp.Variable(X.shape[1])
    objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + (lam / 2) * cp.sum_squares(w))
    problem = cp.Problem(objective)
    problem.solve()
    return np.array(w.value)

# Lasso (CVXPY)
def lasso(X, y, lam):
    w = cp.Variable(X.shape[1])
    objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * cp.norm(w, 1))
    problem = cp.Problem(objective)
    problem.solve()
    return np.array(w.value)

# Elastic Net (CVXPY)
def elastic_net(X, y, lam, alpha=0.5):
    w = cp.Variable(X.shape[1])
    l1_penalty = alpha * cp.norm(w, 1)
    l2_penalty = (1 - alpha) * cp.sum_squares(w)
    objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * (l1_penalty + l2_penalty / 2))
    problem = cp.Problem(objective)
    problem.solve()
    return np.array(w.value)

# Regularization Path (λ 변화에 따른 해의 궤적)
lambdas = np.logspace(-3, 1, 100)
w_ridge_path = []
w_lasso_path = []

for lam in lambdas:
    w_ridge_path.append(ridge_regression(X, y, lam))
    w_lasso_path.append(lasso(X, y, lam))

w_ridge_path = np.array(w_ridge_path)
w_lasso_path = np.array(w_lasso_path)

# 2D 기하 시각화
fig = plt.figure(figsize=(16, 12))

# 1. Ridge vs Lasso 비교 (2D)
ax1 = plt.subplot(2, 3, 1)
lam_demo = 0.5

# 손실함수의 등고선
w1 = np.linspace(-3, 3, 200)
w2 = np.linspace(-3, 3, 200)
W1, W2 = np.meshgrid(w1, w2)
Z_loss = np.zeros_like(W1)
for i in range(len(w1)):
    for j in range(len(w2)):
        w_test = np.array([W1[j, i], W2[j, i]])
        Z_loss[j, i] = 0.5 * np.sum((X @ w_test - y)**2)

ax1.contour(W1, W2, Z_loss, levels=20, colors='gray', alpha=0.5)

# Ridge 제약: 원
theta = np.linspace(0, 2*np.pi, 100)
r_ridge = np.sqrt(2 / lam_demo)  # ‖w‖²_2 ≤ 2/λ 대응
ax1.plot(r_ridge * np.cos(theta), r_ridge * np.sin(theta), 'b-', linewidth=2, label='L2 ball')

# Lasso 제약: 마름모
r_lasso = 1 / lam_demo
ax1.plot([r_lasso, 0, -r_lasso, 0, r_lasso], [0, r_lasso, 0, -r_lasso, 0], 'r-', linewidth=2, label='L1 ball')

# 최적해
w_ridge_sol = ridge_regression(X, y, lam_demo)
w_lasso_sol = lasso(X, y, lam_demo)
ax1.plot(w_ridge_sol[0], w_ridge_sol[1], 'bo', markersize=10, label='Ridge optimum')
ax1.plot(w_lasso_sol[0], w_lasso_sol[1], 'rs', markersize=10, label='Lasso optimum')

ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_title(f'L1 vs L2 Constraint (λ={lam_demo})')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 2. Regularization Path: Ridge
ax2 = plt.subplot(2, 3, 2)
ax2.plot(np.log10(lambdas), w_ridge_path[:, 0], 'b-', linewidth=2, label='w[0]')
ax2.plot(np.log10(lambdas), w_ridge_path[:, 1], 'r-', linewidth=2, label='w[1]')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('log10(λ)')
ax2.set_ylabel('Coefficient Value')
ax2.set_title('Ridge Regularization Path')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Regularization Path: Lasso
ax3 = plt.subplot(2, 3, 3)
ax3.plot(np.log10(lambdas), w_lasso_path[:, 0], 'b-', linewidth=2, label='w[0]')
ax3.plot(np.log10(lambdas), w_lasso_path[:, 1], 'r-', linewidth=2, label='w[1]')
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('log10(λ)')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Lasso Regularization Path (Sparse!)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 고차원 데이터: Sparsity 비교
ax4 = plt.subplot(2, 3, 4)
X_hd, y_hd = make_regression(n_samples=100, n_features=50, n_informative=10, 
                             noise=10, random_state=42)
X_hd = StandardScaler().fit_transform(X_hd)
y_hd = (y_hd - y_hd.mean()) / y_hd.std()

lam_hd = 0.1
w_ridge_hd = ridge_regression(X_hd, y_hd, lam_hd)
w_lasso_hd = lasso(X_hd, y_hd, lam_hd)

sparsity_ridge = np.sum(np.abs(w_ridge_hd) < 1e-4)
sparsity_lasso = np.sum(np.abs(w_lasso_hd) < 1e-4)

ax4.bar(['Ridge', 'Lasso'], [np.sum(np.abs(w_ridge_hd) > 1e-4), 
                              np.sum(np.abs(w_lasso_hd) > 1e-4)], color=['blue', 'red'])
ax4.set_ylabel('Non-zero Coefficients')
ax4.set_title(f'Sparsity Comparison (p=50)\nRidge: {50-sparsity_ridge}, Lasso: {50-sparsity_lasso}')
ax4.set_ylim([0, 50])

# 5. Elastic Net 비교
ax5 = plt.subplot(2, 3, 5)
alphas = np.linspace(0, 1, 5)
colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

for alpha, color in zip(alphas, colors):
    w_en = elastic_net(X_hd, y_hd, 0.1, alpha=alpha)
    num_nonzero = np.sum(np.abs(w_en) > 1e-4)
    ax5.bar(alpha, num_nonzero, color=color, width=0.15, 
           label=f'α={alpha:.2f}')

ax5.set_xlabel('Elastic Net α')
ax5.set_ylabel('Non-zero Coefficients')
ax5.set_title('Elastic Net: Blend of L1 and L2')
ax5.set_ylim([0, 50])
ax5.grid(True, axis='y', alpha=0.3)

# 6. L1 ball 기하 (마름모 구조)
ax6 = plt.subplot(2, 3, 6)
# L1 제약의 꼭짓점 표현
vertices = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 0)]
vertices = np.array(vertices)
ax6.fill(vertices[:, 0], vertices[:, 1], alpha=0.3, color='red', label='L1 ball')
ax6.plot(vertices[:, 0], vertices[:, 1], 'r-', linewidth=2)

# L2 ball
theta = np.linspace(0, 2*np.pi, 100)
ax6.fill(np.cos(theta), np.sin(theta), alpha=0.2, color='blue', label='L2 ball')
ax6.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)

# L∞ ball
square = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
square = np.array(square)
ax6.plot(square[:, 0], square[:, 1], 'g--', linewidth=2, label='L∞ ball')

ax6.set_xlim(-1.5, 1.5)
ax6.set_ylim(-1.5, 1.5)
ax6.set_xlabel('w1')
ax6.set_ylabel('w2')
ax6.set_title('Unit Balls: L1 (마름모) vs L2 (원) vs L∞')
ax6.legend()
ax6.set_aspect('equal')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_geometry.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Ridge solution: {w_ridge_sol}")
print(f"Lasso solution: {w_lasso_sol}")
print(f"Lasso sparsity: {np.sum(np.abs(w_lasso_sol) < 1e-5)} zeros")
print(f"\nHigh-dim Sparsity (p=50):")
print(f"  Ridge non-zeros: {50 - sparsity_ridge}")
print(f"  Lasso non-zeros: {50 - sparsity_lasso}")
```

## 🔗 AI/ML 연결

1. **고차원 통계**: n << p에서 유일하게 작동하는 방법은 L1
2. **변수 선택**: Lasso는 자동 특징 선택 (feature selection)
3. **해석 가능성**: Sparse 모델은 어떤 변수가 중요한지 명확
4. **계산 효율**: Non-zero 계수만 저장/계산
5. **통계 보장**: L1은 "oracle property" (최적 부분집합 선택 일관성)

## ⚖️ 가정과 한계

| 항목 | Ridge | Lasso |
|------|-------|-------|
| 계수 축소 | 연속적 (0이 아님) | 불연속적 (정확히 0) |
| 계산 복잡도 | 폐쇄해 존재 | 반복 알고리즘 필요 |
| Sparsity | 없음 | 자동 |
| 상관 변수 처리 | 균등 가중 | 임의적 선택 |
| 고차원 일관성 | 아님 | 네 (조건부) |

## 📌 핵심 정리

1. **기하 직관**: L2는 구, L1은 마름모 → 꼭짓점에서 축 정렬
2. **Sparsity의 기원**: L1 제약의 꼭짓점 = 축 정렬
3. **부분미분**: 미분 불가능한 L1을 다루는 도구
4. **Regularization Path**: λ 변화에 따른 해의 궤적 추적
5. **Elastic Net**: L1과 L2의 장점을 결합 (상관 변수 처리 개선)

## 🤔 생각해볼 문제

**문제 1:** L∞ 정규화 (Chebyshev norm) ‖w‖∞ = maxⱼ|wⱼ|는 어떤 기하를 가지는가?

<details>
<summary>힌트 및 해설</summary>

제약 집합: ‖w‖∞ ≤ r ⟺ -r ≤ wⱼ ≤ r for all j

이는 **초정육면체(hypercube)** {w : |wⱼ| ≤ r for all j}다.

기하적으로:
- 2D: 정사각형
- d-D: d차원 정육면체
- 각 변수에 대해 독립적인 상한

L∞는 L1보다 덜 sparse하지만, Ridge보다 더 sparse한 중간 특성.

</details>

**問題 2:** Ridge와 Lasso 중 어느 것이 "더 나은가"?

<details>
<summary>힌트 및 해설</summary>

**Ridge가 좋은 경우:**
- 모든 변수가 중요할 가능성
- 상관된 변수들이 많음
- 폐쇄해가 필요한 경우
- 계산 속도 중요

**Lasso가 좋은 경우:**
- 진정한 sparse 모델 (적은 변수가 중요)
- 변수 선택 자동화 필요
- 해석 가능성 중요
- p >> n 상황

**최선의 선택:** Elastic Net (두 방법의 블렌딩)
$$\min_w \left\{ \|Xw-y\|^2 + \alpha\lambda\|w\|_1 + (1-\alpha)\lambda\|w\|^2_2 \right\}$$

α = 0.5일 때 균형 잡힘.

</details>

**문제 3:** 정규화 계수 λ를 어떻게 선택하는가?

<details>
<summary>힌트 및 해설</summary>

**교차검증(Cross-Validation):**
1. λ 범위 설정 (e.g., 10^{-3} ~ 10^3)
2. K-fold CV (e.g., 5-fold)로 각 λ에 대해 검증 오류 계산
3. 평균 검증 오류가 최소인 λ 선택

**통계적 방법:**
- **AIC/BIC**: 모델 복잡도 페널티 포함
- **Sure Screening**: 고차원에서 변수 사전 필터링

**규칙:**
- CV 곡선 최솟값 λ 선택
- 또는 "1-SE rule" (가장 단순한 모델 중 CV 오류 1 표준편차 내)

</details>

<div align="center">

| [◀ 02. Support Vector Machine의 완전 유도](./02-svm-complete.md) | [📚 README](../README.md) | [04. 딥러닝은 왜 비볼록인데 동작하는가 ▶](./04-deep-learning-non-convex.md) |

</div>
