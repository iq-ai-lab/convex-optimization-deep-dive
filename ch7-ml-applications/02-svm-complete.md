# 2. Support Vector Machine의 완전 유도

## 🎯 핵심 질문

SVM은 왜 "support vector"라는 특별한 점들에 집중할까? 쌍대(dual) 문제는 무엇이고, 커널 트릭은 어떻게 비선형 분류를 가능하게 할까?

## 🔍 왜 이 이론이 AI에서 중요한가

SVM은 1990년대 가장 강력한 분류 알고리즘이었고, 오늘날도 여전히 중요하다:
1. **볼록 최적화의 실제 응용**: 프라이멀-쌍대 이론이 직접 활용됨
2. **커널 메서드**: 고차원 특징 공간으로의 암묵적 변환
3. **여백(margin) 개념**: 일반화 이론과 직결됨
4. **정규화**: L2 norm 페널티로 과적합 방지

## 📐 수학적 선행 조건

- **라그랑주 쌍대성** (Lagrange duality)
- **KKT 조건**
- **제약 최적화** (constrained optimization)
- **Mercer 조건**: 커널 함수의 유효성
- **Frobenius 노름**: ‖M‖_F = √Σᵢⱼ Mᵢⱼ²

## 📖 직관적 이해

### Hard-margin SVM (복습)

두 클래스를 **분리하는 초평면**을 찾되, **마진(여백)을 최대화**한다:

$$\max_{w,b} \frac{2}{\|w\|} \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \forall i$$

이는 다음과 같다:
$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \forall i$$

### Soft-margin SVM (새로운 아이디어)

실제 데이터는 완벽히 분리되지 않는다. **슬랙 변수(slack variable)** ξᵢ ≥ 0을 도입:

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0$$

**해석:**
- C가 크면: 오분류 허용 안 함 (hard-margin처럼)
- C가 작으면: 약간의 오분류 허용 (더 부드러운 경계)

## ✏️ 엄밀한 정의

### Soft-margin SVM (프라이멀)

주어진 데이터 $(x_i, y_i)_{i=1}^n$, $x_i \in \mathbb{R}^d$, $y_i \in \{-1, +1\}$에 대해:

$$\text{Primal:} \quad \min_{w,b,\xi} \left\{ \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i \right\}$$
$$\text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### 쌍대 문제 유도

라그랑지안:
$$\mathcal{L}(w,b,\xi,\alpha,\beta) = \frac{1}{2}\|w\|^2 + C \sum_i \xi_i - \sum_i \alpha_i(y_i(w^T x_i + b) - 1 + \xi_i) - \sum_i \beta_i \xi_i$$

여기서 $\alpha_i, \beta_i \geq 0$는 쌍대 변수(라그랑주 승수).

**정류 조건:**
- $\frac{\partial \mathcal{L}}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0$ → $w = \sum_i \alpha_i y_i x_i$
- $\frac{\partial \mathcal{L}}{\partial b} = - \sum_i \alpha_i y_i = 0$ → $\sum_i \alpha_i y_i = 0$
- $\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \beta_i = 0$ → $\alpha_i + \beta_i = C$

$\beta_i \geq 0$에서 **$\alpha_i \leq C$** 제약이 나온다.

### 쌍대 문제

$$\text{Dual:} \quad \max_{\alpha} \left\{ \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j \right\}$$
$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

**행렬 표기:**
$$\text{Dual:} \quad \max_{\alpha} \left\{ \mathbf{1}^T \alpha - \frac{1}{2} \alpha^T Y X X^T Y \alpha \right\}$$

여기서 $Y = \text{diag}(y_1, \ldots, y_n)$.

## 🔬 정리와 증명

### 정리 1: SVM 쌍대 문제는 볼록 이차 계획법이다

**증명:**

목적함수: $f(\alpha) = \mathbf{1}^T \alpha - \frac{1}{2} \alpha^T Q \alpha$, 여기서 $Q = Y X X^T Y$.

$Q$의 성질:
- $Q = Y X X^T Y = (YX)(YX)^T$ (정부호)
- $\alpha^T Q \alpha = \|(YX)^T \alpha\|^2 \geq 0$

따라서 $-\frac{1}{2} \alpha^T Q \alpha$는 볼록 (음수 반정부호).

제약 조건: 선형 부등식과 선형 등식 → 볼록 집합.

**결론:** 이는 **볼록 이차 계획법(QP)**이며, 전역 최적해가 유일하게 존재한다. ∎

### 정리 2: 커널 트릭 (Kernel Trick)

**Mercer 조건:**
함수 K: ℝ^d × ℝ^d → ℝ는 **유효한 커널**이다 ⟺ 모든 유한 데이터 집합 {x₁,...,xₙ}에 대해 그람 행렬 K = [K(xᵢ,xⱼ)]가 반정부호다.

**쌍대 문제에서의 적용:**

원래 쌍대: Kᵢⱼ = xᵢᵀxⱼ를 사용
커널화: Kᵢⱼ = K(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ)로 대체

$$\text{Kernel SVM Dual:} \quad \max_{\alpha} \left\{ \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \right\}$$

계산 복잡도: O(d³) (원래) → O(n²) (커널 SVM, n은 데이터 수)

**흔한 커널들:**
1. **Linear:** K(x, x') = xᵀx'
2. **Polynomial:** K(x, x') = (xᵀx' + 1)^p
3. **RBF (Radial Basis Function):** K(x, x') = exp(-γ‖x - x'‖²)

### 정리 3: 최적해 구조 (KKT 조건)

최적 ($w^*, \alpha^*, \xi^*$)에서:
- $\alpha_i^* > 0$ → i번째 데이터는 **support vector** (경계 또는 경계 내)
- $\alpha_i^* = 0$ → i번째 데이터는 무시됨
- $0 < \alpha_i^* < C$ → **경계 위** (yᵢ(wᵀxᵢ + b) = 1 - ξᵢ, ξᵢ = 0)
- $\alpha_i^* = C$ → **경계 안** (yᵢ(wᵀxᵢ + b) < 1)

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# 비선형 데이터 생성
np.random.seed(42)
X, y = make_circles(n_samples=300, noise=0.1, factor=0.3)
y = 2*y - 1  # {0,1} -> {-1,+1}
X = StandardScaler().fit_transform(X)

# 1. Linear SVM (CVXPY)
def linear_svm_cvxpy(X, y, C):
    n, d = X.shape
    w = cp.Variable(d)
    b = cp.Variable()
    xi = cp.Variable(n)
    
    objective = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))
    constraints = [
        cp.multiply(y, X @ w + b) >= 1 - xi,
        xi >= 0
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    
    return np.array(w.value), b.value, np.array(xi.value)

# 2. 커널 함수들
def kernel_linear(X1, X2):
    return X1 @ X2.T

def kernel_poly(X1, X2, degree=2):
    return (X1 @ X2.T + 1) ** degree

def kernel_rbf(X1, X2, gamma=1.0):
    # ‖x_i - x_j‖² = ‖x_i‖² - 2x_i·x_j + ‖x_j‖²
    sq_norm_X1 = np.sum(X1**2, axis=1, keepdims=True)
    sq_norm_X2 = np.sum(X2**2, axis=1, keepdims=True).T
    sq_dists = sq_norm_X1 - 2 * X1 @ X2.T + sq_norm_X2
    return np.exp(-gamma * sq_dists)

# 3. Kernel SVM (CVXPY)
def kernel_svm_cvxpy(X, y, C, kernel_func):
    n = len(y)
    K = kernel_func(X, X)  # 그람 행렬
    Y = np.diag(y)  # y를 대각 행렬로
    
    alpha = cp.Variable(n)
    objective = cp.Minimize(
        0.5 * cp.quad_form(alpha, Y @ K @ Y) - cp.sum(alpha)
    )
    constraints = [
        cp.sum(cp.multiply(alpha, y)) == 0,
        alpha >= 0,
        alpha <= C
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    
    return np.array(alpha.value)

# 4. 예측 함수
def predict_linear(w, b, X):
    return np.sign(X @ w + b)

def predict_kernel(alpha, X_train, y_train, b, X_test, kernel_func):
    K_test = kernel_func(X_test, X_train)  # (n_test, n_train)
    decision = K_test @ (alpha * y_train) + b
    return np.sign(decision)

# 실행
C = 1.0

# Linear SVM
w_lin, b_lin, xi_lin = linear_svm_cvxpy(X, y, C)
print(f"Linear SVM trained. Support vectors: {np.sum(xi_lin > 1e-5)}")

# Kernel SVMs
alpha_rbf = kernel_svm_cvxpy(X, y, C, 
                              lambda X1, X2: kernel_rbf(X1, X2, gamma=10.0))

# b 계산 (support vector 중 0 < alpha < C인 것 사용)
sv_indices = (alpha_rbf > 1e-5) & (alpha_rbf < C - 1e-5)
if np.any(sv_indices):
    b_rbf = np.mean(y[sv_indices] - kernel_rbf(X[sv_indices], X) @ (alpha_rbf * y))
else:
    b_rbf = 0

print(f"RBF SVM trained. Support vectors: {np.sum(alpha_rbf > 1e-5)}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 메시
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Linear SVM
Z_lin = predict_linear(w_lin, b_lin, grid_points).reshape(xx.shape)
axes[0].contourf(xx, yy, Z_lin, levels=[-1, 0, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
axes[0].contour(xx, yy, Z_lin, levels=[0], colors='black', linewidths=2)
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', s=30, label='y=+1')
axes[0].scatter(X[y==-1, 0], X[y==-1, 1], c='red', s=30, label='y=-1')
sv_lin = (np.sum(xi_lin > 1e-5))
axes[0].set_title(f'Linear SVM (C={C}, SV={sv_lin})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RBF SVM
Z_rbf = predict_kernel(alpha_rbf, X, y, b_rbf, grid_points, 
                       lambda X1, X2: kernel_rbf(X1, X2, gamma=10.0)).reshape(xx.shape)
axes[1].contourf(xx, yy, Z_rbf, levels=[-1, 0, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
axes[1].contour(xx, yy, Z_rbf, levels=[0], colors='black', linewidths=2)
axes[1].scatter(X[y==1, 0], X[y==1, 1], c='blue', s=30, label='y=+1')
axes[1].scatter(X[y==-1, 0], X[y==-1, 1], c='red', s=30, label='y=-1')
sv_rbf = np.sum(alpha_rbf > 1e-5)
axes[1].set_title(f'RBF SVM (C={C}, γ=10, SV={sv_rbf})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Linear SVM and Kernel SVM successfully trained and compared")
```

## 🔗 AI/ML 연결

1. **분류의 기초**: 선형/비선형 분류 모두 처리
2. **여백 최대화**: 일반화 이론과 직결 (VC 차원)
3. **커널 메서드**: 신경망이 나오기 전 비선형 학습의 주류
4. **정규화의 구체화**: C 값이 정규화 강도 조절
5. **쌍대 관점**: 각 데이터 포인트의 역할을 명확히 함

## ⚖️ 가정과 한계

| 항목 | 가정 | 한계 |
|------|------|------|
| 이진 분류 | 두 클래스만 처리 | 다중 클래스는 One-vs-Rest, One-vs-One 필요 |
| 볼록성 | 프라이멀/쌍대 모두 볼록 | 보장된 전역 최적해 (계산 비용 증가 가능) |
| 커널 선택 | Mercer 조건 만족 필요 | 사용자가 커널 선택해야 함 |
| 계산 복잡도 | O(n² ~ n³) | 매우 큰 데이터셋에서는 근사 필요 |

## 📌 핵심 정리

1. **Soft-margin SVM**: 슬랙 변수로 오분류 허용
2. **쌍대 문제**: 볼록 이차 계획법 → 전역 최적해 보장
3. **Support Vectors**: α > 0인 데이터만 의사결정에 영향
4. **커널 트릭**: 암묵적 고차원 특징 매핑 (계산 효율)
5. **KKT 조건**: 최적해 구조 설명 (어떤 데이터가 중요한가)

## 🤔 생각해볼 문제

**문제 1:** C → 0일 때와 C → ∞일 때 SVM은 어떻게 변하는가?

<details>
<summary>힌트 및 해설</summary>

**C → 0 (약한 정규화):**
- ξ 페널티가 약함 → 많은 오분류 허용
- 대부분의 α = 0 (support vector 적음)
- 넓은 마진 (wide margin)

**C → ∞ (강한 정규화):**
- ξ 페널티가 강함 → 거의 오분류 안 함
- Hard-margin SVM에 수렴
- 좁은 마진 (narrow margin), 과적합 위험

최적 C는 교차검증(cross-validation)으로 선택.

</details>

**문제 2:** RBF 커널에서 γ (gamma)를 크게 하면 무슨 일이 일어나는가?

<details>
<summary>힌트 및 해설</summary>

K(x, x') = exp(-γ‖x-x'‖²)에서:

**γ 크면:**
- K(x, x')은 x ≈ x'일 때만 크고, 멀면 0에 가까움
- "국소적" 영향: 각 데이터가 주변만 영향
- 결정 경계가 복잡하고 구불거림
- **과적합** 위험

**γ 작으면:**
- K(x, x') ≈ exp(0) = 1 (모든 x, x'에 대해 비슷)
- "전역적" 영향: 모든 데이터 균등하게 영향
- 결정 경계가 부드러움
- 과소적합(underfitting) 위험

최적 γ도 교차검증으로 선택.

</details>

**문제 3:** Mercer 조건이 왜 필요한가? Mercer 조건 없는 "커널"을 사용하면?

<details>
<summary>힌트 및 해설</summary>

Mercer 조건 ⇔ K가 특정 φ에 대해 φ(x)ᵀφ(x')로 표현 가능.

이 조건이 없으면:
- 그람 행렬 K = [K(xᵢ,xⱼ)]가 반정부호가 아님
- 쌍대 목적함수가 **볼록이 아님**
- 전역 최적해 보장 없음
- 계산 불안정

따라서 Mercer 조건은 **쌍대의 볼록성과 계산 가능성을 보장하는 필수 조건**.

실무: scikit-learn의 SVC는 자동으로 유효한 커널만 지원.

</details>

<div align="center">

| [◀ 01. Logistic Regression은 볼록이다](./01-logistic-regression-convex.md) | [📚 README](../README.md) | [03. Regularization의 기하 — L1 vs L2 ▶](./03-regularization-geometry.md) |

</div>
