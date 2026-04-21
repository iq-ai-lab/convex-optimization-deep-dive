# 2. 주요 Proximal 연산

## 🎯 핵심 질문
- 실제로 계산 가능한 proximal operators는 어떤 함수들인가?
- 각각의 닫힌 형태는 무엇인가?
- 이들이 ML에서 어떻게 쓰이는가?

## 🔍 왜 이 이론이 AI에서 중요한가

Proximal 기반 알고리즘의 실용성은 **닫힌 형태(closed-form) proximal**의 존재에 달려있다. 
- **Soft-thresholding**: L1 정규화, sparse coding, feature selection
- **Projection**: 제약 최적화, box constraint, probability simplex
- **Group Lasso**: structured sparsity (SNP, genomics)
- **Nuclear norm**: low-rank matrix completion, 추천 시스템

이 섹션에서 배우는 5가지 proximal은 현대 ML의 기본 도구상자다.

## 📐 수학적 선행 조건

- **Separable 함수**: $f(x) = \sum_i f_i(x_i)$ → $\text{prox}_f(v)_i = \text{prox}_{f_i}(v_i)$
- **Scaling 성질**: $\text{prox}_{\lambda f}(v) = \arg\min_x (\lambda f(x) + \frac{1}{2}\|x-v\|^2)$
- **SVD 분해**: 행렬의 nuclear norm prox 유도에 필요

## 📖 직관적 이해

**Proximal operators의 일관된 패턴**:

1. **Sparse 패턴 (Soft-thresholding)**: "작은 값은 버리고, 큰 값은 조금 줄인다"
2. **Projection**: "경계를 벗어나면 경계로 당긴다"
3. **Nuclear norm**: "작은 singular value는 버리고, 큰 것은 유지"

핵심은 **좌표 독립성(separability)** 또는 **SVD 분해**를 활용하는 것이다.

## ✏️ 엄밀한 정의

**정의 2.1 (L1 norm의 Prox - Soft-thresholding)**:

$$S_\lambda(v) := \text{prox}_{\lambda\|\cdot\|_1}(v)_i = \text{sign}(v_i)\max(|v_i| - \lambda, 0)$$

**정의 2.2 (L2 ball의 Prox - Projection)**:

$$\text{proj}_{B_r}(v) := \text{prox}_{I_{\|·\|_2 \leq r}}(v) = \begin{cases} v & \text{if } \|v\| \leq r \\ \frac{rv}{\|v\|} & \text{if } \|v\| > r \end{cases}$$

**정의 2.3 (Simplex의 Prox - Projection onto Simplex)**:

$$\Delta_n := \{x \in \mathbb{R}^n : \mathbf{1}^T x = 1, x \geq 0\}$$

$$\text{proj}_{\Delta_n}(v) := \arg\min_{x \in \Delta_n} \|x - v\|^2$$

**정의 2.4 (Group Lasso의 Prox)**:

$$\text{prox}_{\lambda\sum_g \|x_g\|_2}(v)_g = \left(1 - \frac{\lambda}{\|v_g\|_2}\right)_+ v_g$$

여기서 $(t)_+ = \max(t, 0)$.

**정의 2.5 (Nuclear norm의 Prox)**:

$$\text{prox}_{\lambda\|·\|_*}(V) = U\text{diag}(S_\lambda(\sigma))V^T$$

여기서 $V = U\text{diag}(\sigma)V^T$는 SVD.

## 🔬 정리와 증명

**정리 2.6 (Soft-thresholding의 유도)**:

1D 문제를 풀자: $\min_x (\lambda|x| + \frac{1}{2}(x-v)^2)$

*증명*: 부분미분 $\partial |x| = \begin{cases} \{1\} & x > 0 \\ [-1,1] & x = 0 \\ \{-1\} & x < 0 \end{cases}$

최적성 조건: $0 \in \lambda \partial |x^*| + (x^* - v)$

**경우 1** ($x^* > 0$): $0 \in \{\lambda\} + (x^* - v)$ → $x^* = v - \lambda$
- 유효 조건: $v - \lambda > 0$ → $v > \lambda$

**경우 2** ($x^* < 0$): $0 \in \{-\lambda\} + (x^* - v)$ → $x^* = v + \lambda$
- 유효 조건: $v + \lambda < 0$ → $v < -\lambda$

**경우 3** ($x^* = 0$): $0 \in [-\lambda, \lambda] + (0 - v)$ → $|v| \leq \lambda$

따라서:
$$S_\lambda(v) = \begin{cases} v - \lambda & v > \lambda \\ 0 & |v| \leq \lambda \\ v + \lambda & v < -\lambda \end{cases} = \text{sign}(v)\max(|v| - \lambda, 0)$$ ∎

**정리 2.7 (L2 ball projection의 유도)**:

$\min_x \|x - v\|^2$ s.t. $\|x\| \leq r$

*증명*: KKT 조건:
$$0 = 2(x^* - v) + \mu x^*, \quad \mu \geq 0, \quad \mu(\|x^*\| - r) = 0$$

**경우 1** ($\|v\| \leq r$): $\mu = 0$ → $x^* = v$

**경우 2** ($\|v\| > r$): $\mu > 0$ → $x^* = \frac{v}{1 + \mu/2}$
- 제약: $\|x^*\| = r$ → $\frac{\|v\|}{1 + \mu/2} = r$ → $1 + \mu/2 = \|v\|/r$
- 따라서: $x^* = \frac{r}{\|v\|}v$ ∎

**정리 2.8 (Nuclear norm prox의 성질)**:

$$\text{prox}_{\lambda\|·\|_*}(V) = U\text{diag}(S_\lambda(\sigma))V^T$$

여기서 $V = U\text{diag}(\sigma)V^T$는 SVD.

*증명 스케치*: Nuclear norm은 singular value들의 합: $\|V\|_* = \sum_i \sigma_i$.

따라서 문제는:
$$\min_X \|X\|_* + \frac{1}{2}\|X - V\|_F^2$$

SVD 불변성(unitary invariant)에 의해, 각 singular value에 soft-thresholding을 독립적으로 적용:
$$\sigma_i^* = \text{sign}(\sigma_i)\max(|\sigma_i| - \lambda, 0)$$ ∎

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.special import xlogy

# ============ 1. Soft-thresholding (L1) ============

def soft_threshold(v, lam):
    """
    prox_{λ||·||_1}(v) = sign(v) * max(|v| - λ, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

def verify_soft_threshold():
    """Soft-thresholding의 최적성 조건 검증"""
    v = np.array([1.5, -2.3, 0.4, -0.1])
    lam = 0.5
    x_star = soft_threshold(v, lam)
    
    # 최적성: 0 ∈ λ∂|x*| + (x* - v)
    # 즉, v - x* ∈ λ∂|x*|
    residual = v - x_star
    
    print("\n=== Soft-thresholding Verification ===")
    print(f"v = {v}, λ = {lam}")
    print(f"prox(v) = {x_star}")
    print(f"v - prox(v) = {residual} (should be in λ∂|x|)")
    
    # 각 좌표 검증
    for i, (vi, xi, ri) in enumerate(zip(v, x_star, residual)):
        if xi > 0:
            expected_subgrad = lam  # ∂|x|={1}
        elif xi < 0:
            expected_subgrad = -lam  # ∂|x|={-1}
        else:
            expected_subgrad = [-lam, lam]  # ∂|x|=[-1,1]
        print(f"  x[{i}]: v={vi:.2f}, x*={xi:.2f}, v-x*={ri:.2f} " +
              f"(expected λ∂|x*|={expected_subgrad})")

verify_soft_threshold()

# ============ 2. L2 Ball Projection ============

def proj_l2_ball(v, r):
    """
    prox_{I_{||·||_2 ≤ r}}(v) = r*v/max(r, ||v||)
    """
    norm_v = np.linalg.norm(v)
    if norm_v <= r:
        return v
    return r * v / norm_v

def verify_l2_ball():
    """L2 ball projection 검증"""
    v = np.array([3.0, 4.0])  # ||v|| = 5
    r = 2.0
    x_star = proj_l2_ball(v, r)
    
    print("\n=== L2 Ball Projection ===")
    print(f"v = {v}, ||v|| = {np.linalg.norm(v):.3f}")
    print(f"r = {r}")
    print(f"proj(v) = {x_star}, ||proj(v)|| = {np.linalg.norm(x_star):.3f}")
    print(f"Distance to boundary: {np.linalg.norm(x_star) - r:.2e}")

verify_l2_ball()

# ============ 3. Simplex Projection (Euclidean) ============

def proj_simplex(v, s=1.0):
    """
    min ||x - v||^2 s.t. 1^T x = s, x ≥ 0
    
    Algorithm (Michelot, 1986):
    1. Sort v in descending order
    2. Find threshold: θ = (cumsum(v) - s) / (1:n)
    3. ρ = max{j : v_j - θ_j > 0}
    4. λ = (sum(v[1:ρ]) - s) / ρ
    5. x = max(v - λ, 0)
    """
    n = len(v)
    u = np.sort(v)[::-1]  # 내림차순
    
    cssv = np.cumsum(u)
    rho = np.arange(1, n + 1)
    theta = (cssv - s) / rho
    
    # ρ = max{j : u[j-1] - θ[j-1] > 0}
    idx = np.where(u > theta)[0]
    rho_idx = idx[-1] if len(idx) > 0 else 0
    theta_star = theta[rho_idx]
    
    return np.maximum(v - theta_star, 0)

def verify_simplex():
    """Simplex projection 검증"""
    v = np.array([2.5, 1.0, -0.5, 3.0])
    x_star = proj_simplex(v)
    
    print("\n=== Simplex Projection ===")
    print(f"v = {v}")
    print(f"proj(v) = {x_star}")
    print(f"sum(proj) = {np.sum(x_star):.6f} (should be 1)")
    print(f"min(proj) = {np.min(x_star):.6f} (should be ≥ 0)")

verify_simplex()

# ============ 4. Group Lasso ============

def prox_group_lasso(v, groups, lam):
    """
    prox_{λ Σ||x_g||_2}(v) = (1 - λ/||v_g||_2)_+ * v_g per group
    
    Args:
        v: input vector
        groups: list of group indices (e.g., [[0,1], [2,3], [4]])
        lam: regularization parameter
    """
    x = np.zeros_like(v)
    for group in groups:
        v_g = v[group]
        norm_g = np.linalg.norm(v_g)
        if norm_g > 0:
            shrink_factor = max(1 - lam / norm_g, 0)
            x[group] = shrink_factor * v_g
    return x

def verify_group_lasso():
    """Group Lasso prox 검증"""
    v = np.array([1.5, 2.0, 0.3, 0.1, 2.5])
    groups = [[0, 1], [2, 3], [4]]  # 두 개의 그룹과 한 개의 싱글톤
    lam = 0.8
    
    x_star = prox_group_lasso(v, groups, lam)
    
    print("\n=== Group Lasso ===")
    print(f"v = {v}")
    print(f"groups = {groups}, λ = {lam}")
    print(f"prox(v) = {x_star}")
    
    for i, group in enumerate(groups):
        v_g = v[group]
        x_g = x_star[group]
        norm_v_g = np.linalg.norm(v_g)
        norm_x_g = np.linalg.norm(x_g)
        print(f"  Group {i}: ||v_g|| = {norm_v_g:.3f}, " +
              f"||x*_g|| = {norm_x_g:.3f}, shrink = {norm_x_g/norm_v_g:.3f}")

verify_group_lasso()

# ============ 5. Nuclear Norm (via SVD) ============

def prox_nuclear_norm(V, lam):
    """
    prox_{λ||·||_*}(V) = U diag(soft_threshold(σ, λ)) V^T
    """
    U, sigma, Vt = svd(V, full_matrices=False)
    sigma_thresh = soft_threshold(sigma, lam)
    return U @ np.diag(sigma_thresh) @ Vt

def verify_nuclear_norm():
    """Nuclear norm prox 검증"""
    V = np.array([[3.0, 2.0, 1.0],
                  [1.0, 4.0, 2.0]])
    lam = 1.5
    
    X_star = prox_nuclear_norm(V, lam)
    
    U, sigma_orig, _ = svd(V, full_matrices=False)
    sigma_thresh = soft_threshold(sigma_orig, lam)
    
    print("\n=== Nuclear Norm ===")
    print(f"V shape = {V.shape}")
    print(f"Original singular values: {sigma_orig}")
    print(f"Thresholded singular values: {sigma_thresh}")
    print(f"||V||_* = {np.sum(sigma_orig):.3f}")
    print(f"||X*||_* = {np.sum(sigma_thresh):.3f}")

verify_nuclear_norm()

# ============ 시각화 ============

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Soft-thresholding
ax = axes[0, 0]
v_range = np.linspace(-3, 3, 100)
lams = [0.5, 1.0, 1.5]
colors = ['blue', 'green', 'red']
for lam, color in zip(lams, colors):
    s_vals = [soft_threshold(np.array([v]), lam)[0] for v in v_range]
    ax.plot(v_range, s_vals, color=color, linewidth=2, label=f'λ={lam}')
ax.plot(v_range, v_range, 'k--', alpha=0.3, label='Identity')
ax.set_xlabel('v')
ax.set_ylabel('prox(v)')
ax.set_title('Soft-thresholding (L1)')
ax.grid(True, alpha=0.3)
ax.legend()

# 2. L2 ball projection 
ax = axes[0, 1]
theta = np.linspace(0, 2*np.pi, 100)
for r, style in [(1.0, '-'), (1.5, '--'), (2.0, ':')]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, style, linewidth=2, label=f'r={r}')
# 예시 점들
v_points = np.array([[1.5, 1.5], [2.5, 2.5], [3.0, 0.0]])
colors_pts = ['blue', 'red', 'green']
for v, color in zip(v_points, colors_pts):
    proj = proj_l2_ball(v, 1.5)
    ax.scatter([v[0]], [v[1]], color=color, s=100, marker='o')
    ax.scatter([proj[0]], [proj[1]], color=color, s=100, marker='x')
    ax.arrow(v[0], v[1], proj[0]-v[0], proj[1]-v[1], 
             head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.5)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_title('L2 Ball Projection')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axis('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# 3. Simplex projection
ax = axes[1, 0]
# 2-simplex (삼각형)
simplex_2d = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
ax.plot(simplex_2d[:, 0], simplex_2d[:, 1], 'k-', linewidth=2, label='Simplex')
# 정점들
for v_2d in [[2.0, 0.5], [0.3, 0.3], [-0.5, 1.5]]:
    v = np.array([v_2d[0], v_2d[1], 0])
    x = proj_simplex(v)
    ax.scatter([v[0]], [v[1]], s=100, marker='o')
    ax.scatter([x[0]], [x[1]], s=100, marker='x')
    ax.arrow(v[0], v[1], x[0]-v[0], x[1]-v[1],
             head_width=0.05, head_length=0.05, alpha=0.5)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_title('Simplex Projection (2D)')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 1.5)

# 4. Nuclear norm singular values
ax = axes[1, 1]
n_trials = 10
for trial in range(n_trials):
    V = np.random.randn(3, 5)
    U, sigma, _ = svd(V, full_matrices=False)
    sigma_thresh = soft_threshold(sigma, 1.5)
    ax.plot(range(len(sigma)), sigma, 'b-', alpha=0.3)
    ax.plot(range(len(sigma)), sigma_thresh, 'r-', alpha=0.3)
ax.set_xlabel('Index')
ax.set_ylabel('Singular Value')
ax.set_title(f'Nuclear Norm: Original (blue) vs Thresholded (red, λ=1.5)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/proximal_examples.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to /tmp/proximal_examples.png")
plt.close()

print("\n" + "="*50)
print("Summary: 5 Key Proximal Operators")
print("="*50)
print("1. Soft-thresholding: sparse regularization")
print("2. L2 ball projection: box constraints")
print("3. Simplex projection: probability constraints")
print("4. Group Lasso: structured sparsity")
print("5. Nuclear norm: low-rank constraints")
```

**구현 검증**:
```
=== Soft-thresholding Verification ===
v = [ 1.5 -2.3  0.4 -0.1], λ = 0.5
prox(v) = [ 1.   -1.8  0.   -0. ]
v - prox(v) = [ 0.5 -0.5  0.4 -0.1]

=== L2 Ball Projection ===
v = [3. 4.], ||v|| = 5.000
proj(v) = [1.2 1.6], ||proj(v)|| = 2.000

=== Simplex Projection ===
sum(proj) = 1.000000
```

## 🔗 AI/ML 연결

**Sparse Coding**: $\min_x \|Ax - b\|^2 + \lambda\|x\|_1$ → soft-thresholding

**Constrained Regression**: $\min_x \|Ax - b\|^2$ s.t. $\|x\|_2 \leq r$ → L2 ball projection

**Multi-task Learning**: $\min_X \|AX - B\|_F^2 + \lambda\sum_i \|X_i\|_2$ → group Lasso

**Matrix Completion**: $\min_X \|X - B\|_F^2 + \lambda\|X\|_*$ → nuclear norm prox

## ⚖️ 가정과 한계

- **닫힌 형태 필요**: 모든 함수에서 closed-form prox가 존재하지는 않음
- **계산 비용**: SVD (O(n³)), simplex projection (O(n log n))
- **수치 안정성**: singular value가 매우 작을 때 nuclear norm 계산 주의

## 📌 핵심 정리

| Operator | Closed-form | 계산복잡도 | ML 응용 |
|----------|------------|----------|--------|
| Soft-thresh | $\text{sign}(v)\max(\|v\|-\lambda,0)$ | O(n) | Lasso, sparse |
| L2 ball | $rv/\max(r,\|v\|)$ | O(n) | Constrained |
| Simplex | Sort + threshold | O(n log n) | Probability |
| Group Lasso | Per-group soft-thresh | O(n) | Structured |
| Nuclear | SVD + soft-thresh | O(n³) | Low-rank |

## 🤔 생각해볼 문제

**문제 2.1**: L∞ norm의 dual이 L1 norm임을 이용하여, $\text{prox}_{\lambda\|\cdot\|_\infty}(v)$의 형태를 추론하시오.

<details>
<summary>힌트 및 해설</summary>

$\|x\|_\infty = \max_i |x_i|$의 conjugate는 $\|y\|_1$.

Moreau decomposition: $x + \text{prox}_{\lambda f}(x) = x + \text{prox}_{\lambda f^*}(x/\lambda)$ 형태를 이용하면...

실제로 계산하면: $\text{prox}_{\lambda\|\cdot\|_\infty}(v)$ = soft-threshold on L1 dual

복잡하므로 보통 iteration으로 해결.

</details>

**문제 2.2**: Group Lasso에서 그룹 크기가 다를 때, 크기가 큰 그룹이 더 많이 정규화되는 현상을 설명하시오.

<details>
<summary>힌트 및 해설</summary>

정규화 항: $\lambda \sum_g \|x_g\|_2$

큰 그룹 $g$에 대해: $\|x_g\|_2 = \sqrt{\sum_{i \in g} x_i^2}$

그룹이 클수록 같은 $\lambda$에서 더 큰 threshold를 받음:

threshold = $\lambda / \|v_g\|_2$

해결책: $\|x_g\|_2 / \sqrt{|g|}$ 형태로 정규화 (scaled group Lasso)

</details>

**문제 2.3**: Nuclear norm prox의 계산 복잡도가 O(n³)인 이유를 설명하고, 저랭크 행렬 근사에서는 왜 이것이 문제가 될 수 있는지 논의하시오.

<details>
<summary>힌트 및 해설</summary>

SVD 계산이 O(n³) (full), O(nm·min(n,m)) (reduced)

큰 행렬의 경우:
- n=10000 → SVD가 몇 시간 필요
- 해결책: randomized SVD, power iteration, truncated SVD

실제 matrix completion (Netflix):
- 수억 개의 미지정 값
- 전체 SVD 불가능
- 대신: alternating minimization, SGD 사용

</details>

<div align="center">

| [◀ 01. Proximal Operator의 정의](./01-proximal-operator.md) | [📚 README](../README.md) | [03. ISTA와 FISTA ▶](./03-ista-fista.md) |

</div>
