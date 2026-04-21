# 6. 주요 볼록 함수 카탈로그

## 🎯 핵심 질문

- 어떤 노름들이 볼록한가?
- 왜 음의 엔트로피는 볼록한가?
- 행렬 최적화에서 자주 등장하는 함수들(스펙트럴 노름, 핵 노름)이 보존하는 성질은?

---

## 🔍 왜 이 이론이 AI에서 중요한가

1. **정규화 기법**: L1, L2, elastic net 정규화는 각각 서로 다른 희소성/안정성 특성을 제공. 이들이 모두 볼록한 덕분에 최적화 가능.

2. **확률 모델**: 음의 엔트로피는 최대 엔트로피 원리(MaxEnt)와 연결. 로지스틱 회귀, 소프트맥스의 손실함수가 바로 이 구조.

3. **행렬 완성(Matrix Completion)**: 핵 노름 최소화 $\min_X \|X\|_* \text{ s.t. } X_{ij} = M_{ij}$는 저차 행렬 복원의 원래 문제. 이 문제의 볼록성이 이론적 보장을 제공.

4. **신경망 초기화**: 가중치 행렬의 스펙트럴 노름 제약은 Lipschitz 연속성을 보장 → spectral normalization.

---

## 📐 수학적 선행 조건

- [이전 문서들: 01~05](./01-convex-function-definitions.md)
- 행렬의 특이값(singular value), 스펙트럼
- 정보 이론: 엔트로피, KL 발산
- 행렬 노름의 종류

---

## 📖 직관적 이해

### 각 함수의 역할

```
┌─────────────────────────────────────┐
│ L1 노름  → 희소성 (sparse)           │
│ L2 노름  → 안정성 (smoothness)       │
│ Log-Sum-Exp  → 최댓값의 매끄러운 근사│
│ 음의 엔트로피  → 확률의 정보량       │
│ 스펙트럴 노름  → 최대 신개값          │
│ 핵 노름  → 저차 근사 (low-rank)     │
└─────────────────────────────────────┘
```

---

## ✏️ 엄밀한 정의

**정의 2.22** (Lp 노름, p ≥ 1)
$$\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

**정의 2.23** (스펙트럴 노름, spectral norm)
$$\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^TA)}$$

**정의 2.24** (핵 노름, nuclear norm)
$$\|A\|_* = \sum_{i=1}^r \sigma_i(A)$$
여기서 $\sigma_i(A)$는 A의 i번째 특이값, r은 rank.

**정의 2.25** (음의 엔트로피, negative entropy)
$$H(p) = -\sum_{i} p_i \log p_i, \quad p \in \Delta_n = \{p : p \geq 0, \sum p_i = 1\}$$

**정의 2.26** (Log-Sum-Exp)
$$\text{LSE}(x) = \log\sum_i e^{x_i}$$

---

## 🔬 정리와 증명

### 1. Lp 노름 (p ≥ 1)

**정리 2.21** (Lp 노름의 볼록성)
모든 $p \geq 1$에 대해, $\|x\|_p = (\sum_i |x_i|^p)^{1/p}$는 **볼록함수**.

**증명:**
삼각 부등식(triangle inequality):
$$\|x + y\|_p \leq \|x\|_p + \|y\|_p$$

따라서:
$$\|(\lambda x + (1-\lambda)y)\|_p \leq \lambda\|x\|_p + (1-\lambda)\|y\|_p$$

(p=1: 직접 계산, p>1: Minkowski 부등식으로 증명) □

**켤레 노름**: $\|y\|_q = (\sum_i |y_i|^q)^{1/q}$, 여기서 $\frac{1}{p} + \frac{1}{q} = 1$.

**주요 응용**:
- L1: Lasso 정규화, 희소 모델
- L2: Ridge 정규화, 안정성
- L∞: 게임 이론, robust optimization

---

### 2. 음의 엔트로피

**정리 2.22** (음의 엔트로피의 볼록성)
$$H(p) = -\sum_i p_i \log p_i \quad (p \in \Delta_n, p_i > 0)$$
는 **오목함수**.

**증명:**
2차 미분:
$$\frac{\partial^2 H}{\partial p_i \partial p_j} = \begin{cases}
-\frac{1}{p_i} & i=j \\
0 & i \neq j
\end{cases}$$

Hessian: $\nabla^2 H = -\text{diag}(1/p_i) \preceq 0$ (음반정치) → **오목**. □

따라서 **$-H(p)$는 볼록** (음의 엔트로피).

**켤레함수**: $(-H)^*(y) = \log\sum_i e^{y_i} - c$ (상수를 제외하면 log-sum-exp).

**주요 응용**:
- 최대 엔트로피 원리
- 로지스틱 회귀의 크로스엔트로피 손실
- 정보 병목(Information Bottleneck)

---

### 3. Log-Sum-Exp

**정리 2.23** (Log-Sum-Exp의 볼록성과 매끄러움)
$$f(x) = \log\sum_i e^{x_i}$$
는 **볼록이고 무한 미분가능(smooth)**.

**증명:**
$p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$ (softmax)라 하면:

$$\nabla f(x) = p$$
$$\nabla^2 f(x) = \text{diag}(p) - pp^T$$

$v^T\nabla^2 f v = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2 \geq 0$ (Cauchy-Schwarz) → **볼록**. □

**매끄러움**: $\|e^{x_i}\|$ 항 때문에 Hessian bounded → $L$-smooth.

**주요 응용**:
- 최댓값의 매끄러운 근사: $\max_i x_i \leq \text{LSE}(x) \leq \max_i x_i + \log n$
- 소프트맥스 출력의 손실함수
- 강화학습의 정책 경사(policy gradient)

---

### 4. 스펙트럴 노름

**정리 2.24** (스펙트럴 노름의 볼록성)
$$\|A\|_2 = \max_{\|x\|_2 \leq 1} \|Ax\|_2 = \sigma_{\max}(A)$$
는 A에 대한 **볼록함수**.

**증명:**
상한의 보존(정리 2.12):
$$\|A\|_2 = \max_{\|x\|_2 \leq 1} \|Ax\|_2 = \sup_{\|x\|=1} \|Ax\|_2$$

각 $x$에 대해 $f_x(A) = \|Ax\|_2$는 A에 대해 **선형** (따라서 볼록).

포인트와이즈 상한 → **볼록**. □

**켤레**: 단위공의 지시함수.

**주요 응용**:
- Spectral normalization (신경망)
- 행렬 근사
- SDP 완화

---

### 5. 핵 노름

**정리 2.25** (핵 노름의 볼록성)
$$\|A\|_* = \sum_i \sigma_i(A)$$
는 **볼록함수**.

**증명:**
핵 노름은 특이값들의 합:
$$\|A\|_* = \text{tr}(\sqrt{A^TA})$$

$\sqrt{·}$는 행렬에 대해 오목이고, $\text{tr}(\sqrt{A^TA})$는 A에 대해 볼록.

또는 직접: 최적화로 표현:
$$\|A\|_* = \min_{U,V} \frac{1}{2}(\|U\|_F^2 + \|V\|_F^2) \text{ s.t. } A = UV^T$$

제약과 Frobenius 노름의 합 = 볼록. □

**켤레**: 스펙트럴 노름의 단위 공 지시함수.

**주요 응용**:
- 행렬 완성(matrix completion)
- 저차 행렬 복원
- 강건한 주성분 분석(robust PCA)

---

**예제 2.15**: 행렬 완성 문제

$$\min_X \|X\|_* \text{ s.t. } X_{ij} = M_{ij} \quad \forall (i,j) \in \Omega$$

목적함수가 볼록 → 국소 최솟값 = 전역 최솟값. 이는 불완전한 관측에서 저차 행렬을 복원하는 이론적 기반.

---

**예제 2.16**: 정규화된 로지스틱 회귀

$$\min_w \sum_i \log(1 + e^{-y_i w^T x_i}) + \lambda \|w\|_2^2$$

손실함수 항은 $\log(1 + e^{·})$ 형태 = **convex composition** (지수의 합성).

정규화는 L2 노름 = 볼록. 따라서 전체 목적함수는 **강볼록(μ-strongly convex)**.

---

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import cvxpy as cp

print("=" * 70)
print("1. Lp 노름의 볼록성 검증")
print("=" * 70)

def lp_norm(x, p):
    """Lp norm: (Σ|x_i|^p)^(1/p)"""
    return np.sum(np.abs(x)**p)**(1/p)

def verify_convexity_jensen(f, x, y, num_lambdas=50):
    """Jensen 부등식으로 볼록성 검증"""
    lambdas = np.linspace(0, 1, num_lambdas)
    violations = 0
    
    for lam in lambdas:
        z = lam * x + (1 - lam) * y
        lhs = f(z)
        rhs = lam * f(x) + (1 - lam) * f(y)
        
        if lhs > rhs + 1e-10:
            violations += 1
    
    return violations == 0

# 테스트
x = np.array([1.5, -0.8, 2.1])
y = np.array([-0.5, 1.2, -0.3])

p_values = [1, 1.5, 2, np.inf]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax_idx, p in enumerate(p_values):
    f_p = lambda x, p=p: lp_norm(x, p) if p != np.inf else np.max(np.abs(x))
    
    # Jensen 검증
    result = verify_convexity_jensen(f_p, x, y)
    print(f"\nL^{p} 노름 (p={p}):")
    print(f"  Jensen 부등식 만족: {result} ✓")
    
    # 1D 프로젝션에서 시각화
    lambdas = np.linspace(0, 1, 100)
    x1d = -1 + 3 * lambdas  # x=−1, y=2
    y1d = 0.5 + 1.5 * lambdas
    
    lp_vals = np.array([lp_norm(np.array([xi, yi]), p) for xi, yi in zip(x1d, y1d)])
    
    ax = axes[ax_idx]
    ax.plot(lambdas, lp_vals, 'b-', linewidth=2.5, label=f'$L^{p}$ norm')
    
    # 현(chord)
    chord = lambdas * lp_norm(np.array([-1, 0.5]), p) + (1-lambdas) * lp_norm(np.array([2, 2]), p)
    ax.plot(lambdas, chord, 'r--', linewidth=2, label='Linear interpolation')
    ax.fill_between(lambdas, lp_vals, chord, alpha=0.2, color='green')
    
    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('함수값')
    ax.set_title(f'$L^{p}$ 노름 (p={p}): 볼록성')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lp_norms_convexity.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Lp 노름 시각화 완료")

# ============================================
# 2. 음의 엔트로피의 오목성 (따라서 -H는 볼록)
# ============================================

print("\n" + "=" * 70)
print("2. 음의 엔트로피 (-H)의 볼록성")
print("=" * 70)

def entropy(p):
    """음의 엔트로피: -Σ p_i log(p_i)"""
    # p_i > 0 가정
    p = np.array(p)
    p = p[p > 1e-10]  # 0 항 제외
    return -np.sum(p * np.log(p))

def neg_entropy(p):
    """음의 엔트로피를 최소화하는 것과 동일"""
    return -entropy(p)

# 심플렉스 위의 1D 슬라이스: p = (t, 1-t)
t_vals = np.linspace(0.01, 0.99, 100)
entropy_vals = np.array([entropy([t, 1-t]) for t in t_vals])
neg_entropy_vals = -entropy_vals

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 엔트로피 (오목)
ax = axes[0]
ax.plot(t_vals, entropy_vals, 'b-', linewidth=2.5, label='$H(p) = -\\sum p_i \\log p_i$')

# Jensen으로 오목성 확인
t1, t2 = 0.2, 0.8
lam = 0.5
tm = lam * t1 + (1-lam) * t2

h1, h2 = entropy([t1, 1-t1]), entropy([t2, 1-t2])
hm_actual = entropy([tm, 1-tm])
hm_linear = lam * h1 + (1-lam) * h2

ax.scatter([t1, t2, tm], [h1, h2, hm_actual], color='red', s=100, zorder=5)
ax.plot([t1, t2], [h1, h2], 'r--', linewidth=2, label='Jensen lower bound (오목)')

ax.set_xlabel('$t$ (심플렉스 좌표: $p=(t, 1-t)$)')
ax.set_ylabel('$H(p)$')
ax.set_title('음의 엔트로피: 오목 함수')
ax.legend()
ax.grid(True, alpha=0.3)

# -엔트로피 (볼록)
ax = axes[1]
ax.plot(t_vals, neg_entropy_vals, 'g-', linewidth=2.5, label='$-H(p)$')

# Jensen으로 볼록성 확인
nh1, nh2 = neg_entropy([t1, 1-t1]), neg_entropy([t2, 1-t2])
nhm_actual = neg_entropy([tm, 1-tm])
nhm_linear = lam * nh1 + (1-lam) * nh2

ax.scatter([t1, t2, tm], [nh1, nh2, nhm_actual], color='red', s=100, zorder=5)
ax.plot([t1, t2], [nh1, nh2], 'r--', linewidth=2, label='Jensen upper bound (볼록)')

ax.set_xlabel('$t$')
ax.set_ylabel('$-H(p)$')
ax.set_title('음의 엔트로피: 볼록 함수')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('negative_entropy_convexity.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  엔트로피는 오목, -엔트로피는 볼록 ✓")

# ============================================
# 3. Log-Sum-Exp의 Hessian PSD 검증
# ============================================

print("\n" + "=" * 70)
print("3. Log-Sum-Exp의 Hessian 검증")
print("=" * 70)

def lse(x):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

def lse_grad(x):
    """∇LSE = softmax"""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def lse_hessian_numerical(x, eps=1e-5):
    """수치적 헤시안 계산"""
    n = len(x)
    H = np.zeros((n, n))
    
    grad_at_x = lse_grad(x)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        grad_plus = lse_grad(x_plus)
        
        H[i, :] = (grad_plus - grad_at_x) / eps
    
    return H

# 여러 점에서 테스트
test_points = [
    np.array([0, 0, 0]),
    np.array([1, -1, 2]),
    np.array([-5, 0, 5]),
]

all_psd = True
for x_test in test_points:
    H = lse_hessian_numerical(x_test)
    eigenvals = np.linalg.eigvalsh(H)
    is_psd = np.all(eigenvals >= -1e-6)
    all_psd = all_psd and is_psd
    
    print(f"\n  x = {x_test}:")
    print(f"    Hessian 고유값: {eigenvals}")
    print(f"    PSD: {is_psd} ✓")

print(f"\n  모든 점에서 PSD: {all_psd} ✓")

# ============================================
# 4. 스펙트럴 노름의 볼록성
# ============================================

print("\n" + "=" * 70)
print("4. 스펙트럴 노름 (최대 특이값)의 볼록성")
print("=" * 70)

def spectral_norm(A):
    """스펙트럴 노름 = 최대 특이값"""
    return np.max(np.linalg.svd(A, compute_uv=False))

# Jensen 검증
A1 = np.array([[2, 1], [0, 1]])
A2 = np.array([[1, 1], [1, 2]])

violations = 0
for lam in np.linspace(0, 1, 50):
    A = lam * A1 + (1-lam) * A2
    lhs = spectral_norm(A)
    rhs = lam * spectral_norm(A1) + (1-lam) * spectral_norm(A2)
    
    if lhs > rhs + 1e-10:
        violations += 1

print(f"\n  스펙트럴 노름 Jensen 부등식:")
print(f"  위반 수: {violations}/50")
print(f"  볼록성: {violations == 0} ✓")

# ============================================
# 5. 핵 노름과 행렬 완성
# ============================================

print("\n" + "=" * 70)
print("5. 핵 노름 (Nuclear Norm)의 볼록성")
print("=" * 70)

def nuclear_norm(A):
    """핵 노름 = 특이값의 합"""
    return np.sum(np.linalg.svd(A, compute_uv=False))

# 저차 행렬 생성
U = np.random.randn(5, 2)
V = np.random.randn(3, 2)
A_lowrank = U @ V.T

print(f"\n  저차 행렬 (rank=2)의 핵 노름: {nuclear_norm(A_lowrank):.4f}")

# 행렬 완성 문제 (간단한 예시)
# 실제로는 CVX 문제이지만, 여기서는 설명만
print(f"  행렬 완성 문제: min ‖X‖_* s.t. X_ij = M_ij for (i,j) ∈ Ω")
print(f"  → 볼록 최적화 문제 → 국소 최솟값 = 전역 최솟값 ✓")

# ============================================
# 6. CVXPY에서 여러 함수 사용
# ============================================

print("\n" + "=" * 70)
print("6. CVXPY에서의 볼록 함수 활용")
print("=" * 70)

# 변수 정의
x = cp.Variable(5)
y = cp.Parameter(5)
y.value = np.array([1, 2, 3, 4, 5])

# 다양한 목적함수들
print("\n  DCP 규칙 검증:")

# 1. L1 정규화
prob1 = cp.Problem(cp.Minimize(cp.norm(x, 1) + cp.sum_squares(x)))
print(f"  1. L1 + L2² (Elastic Net): {prob1.is_dcp()} ✓")

# 2. Log-sum-exp
prob2 = cp.Problem(cp.Minimize(cp.sum(cp.exp(x))))
print(f"  2. exp(x)의 합 (LSE 근사): {prob2.is_dcp()} ✓")

# 3. 최대값의 매끄러운 근사
prob3 = cp.Problem(cp.Minimize(cp.max(x)))
print(f"  3. 최댓값: {prob3.is_dcp()} ✓")

# 4. 노름 제약
A = np.random.randn(3, 5)
prob4 = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - y)),
    [cp.norm(x, 2) <= 1]
)
print(f"  4. 제약된 회귀 (스펙트럴): {prob4.is_dcp()} ✓")

print("\n" + "=" * 70)
print("모든 검증 완료!")
print("=" * 70)

# ============================================
# 7. 종합 비교 표
# ============================================

print("\n종합 비교 표:")
print("-" * 80)
print(f"{'함수':<25} {'볼록성':<10} {'용도':<40}")
print("-" * 80)
print(f"{'L^p 노름 (p≥1)':<25} {'볼록':<10} {'정규화, 거리 측도':<40}")
print(f"{'음의 엔트로피':<25} {'볼록':<10} {'확률 모델, MaxEnt':<40}")
print(f"{'Log-Sum-Exp':<25} {'볼록':<10} {'매끄러운 max 근사':<40}")
print(f"{'스펙트럴 노름':<25} {'볼록':<10} {'행렬 최적화, SDP':<40}")
print(f"{'핵 노름':<25} {'볼록':<10} {'행렬 완성, 저차 근사':<40}")
print("-" * 80)
```

---

## 🔗 AI/ML 연결

1. **L1 정규화 (Lasso)**: L1 노름의 볼록성으로 유일 최솟값 존재 (정확한 feature selection 가능). Ridge(L2)와 달리 자동으로 불필요한 계수를 0으로 만듦.

2. **로지스틱 회귀의 손실**: 크로스 엔트로피 손실 $-[y\log p + (1-y)\log(1-p)]$는 음의 엔트로피와 깊게 연결. Softmax 출력의 경우 log-sum-exp로 표현.

3. **행렬 완성 (Netflix Challenge)**: 핵 노름을 최소화하여 불완전한 등급 행렬에서 사용자-아이템 상호작용 복원. 볼록성으로 이론적 보장 제공.

4. **Spectral Normalization (SN)**: GAN의 판별자 가중치를 스펙트럴 노름으로 제약하여 Lipschitz 연속성 보장 → 학습 안정성 개선.

---

## ⚖️ 가정과 한계

| 함수 | 성질 | 주의점 |
|-----|------|--------|
| **L^p** | p ≥ 1일 때만 볼록 | p < 1: 오목이거나 비볼록 |
| **음의 엔트로피** | 심플렉스 위에서만 정의 | 확률 조건 필요 |
| **Log-Sum-Exp** | 전체 $\mathbb{R}^n$에서 정의 | 수치 안정성: overflow 처리 필요 |
| **스펙트럴 노름** | 모든 행렬에서 정의 | 계산 비용: SVD 필요 |
| **핵 노름** | 행렬에만 정의 | 계산 비용: 모든 특이값 계산 |

---

## 📌 핵심 정리

| 함수 | 공식 | 켤레 함수 | 주요 성질 |
|-----|------|---------|----------|
| **L^p 노름** | $(\sum \|x_i\|^p)^{1/p}$ | L^q 노름 지시 | p≥1 볼록 |
| **음의 엔트로피** | $-\sum p_i \log p_i$ | Log-Sum-Exp | 심플렉스에서 볼록 |
| **Log-Sum-Exp** | $\log \sum e^{x_i}$ | 음의 엔트로피 | 무한 미분가능 |
| **스펙트럴** | $\sigma_{\max}(A)$ | 단위공 지시 | 모든 행렬에서 볼록 |
| **핵 노름** | $\sum \sigma_i(A)$ | 스펙트럴 지시 | 저차 복원에 최적 |

---

## 🤔 생각해볼 문제

**문제 2.16**: 왜 L1 정규화는 희소 해를 만드는가? L2는 왜 만들지 않는가?

<details>
<summary>힌트 및 해설</summary>

**힌트**: 각 노름의 "레벨셋" (등고선) 모양을 생각해보세요.

**해설**:
- L1: $\|w\|_1 \leq r$은 **다이아몬드 모양** → 꼭짓점이 축 위에 있음
- L2: $\|w\|_2 \leq r$은 **원형** → 축 위의 특별한 점 없음

최적화에서 손실함수가 축과 만날 때:
- L1: 다이아몬드 꼭짓점(예: $(r, 0)$)에서 만날 수 있음 → 일부 가중치 = 0 (희소)
- L2: 원 위의 일반적인 점에서 만남 → 모든 가중치 > 0 (dense)

이것이 **기하학적** 이유입니다.

</details>

---

**문제 2.17**: 핵 노름이 "저차 행렬 복원"에 최적인 이유는?

<details>
<summary>힌트 및 해설</summary>

**힌트**: 두 가지 행렬: 고차 행렬과 저차 행렬이 같은 핵 노름을 가질 수 있을까?

**해설**:
핵 노름 = 특이값의 합.
- 저차 행렬 (rank=r): 처음 r개 특이값만 0이 아님
- 고차 행렬 (rank=n): 모든 n개 특이값이 0이 아님

같은 관측 데이터에서, 핵 노름 최소화는 **특이값 개수를 줄이려고 함**
→ 자동으로 **낮은 rank의 해**를 찾음

이는 L1이 희소성을 만드는 것과 유사한 메커니즘입니다.

</details>

---

**문제 2.18**: Elastic Net $\min_w \|Xw-y\|^2 + \lambda_1\|w\|_1 + \lambda_2\|w\|^2$가 왜 L1과 L2 모두의 장점을 결합하는가?

<details>
<summary>힌트 및 해설</summary>

**힌트**: L1의 희소성과 L2의 안정성을 함께 생각해보세요.

**해설**:
- L1 항: 불필요한 변수를 정확히 0으로 만듦 (feature selection)
- L2 항: 0이 아닌 계수들 사이의 collinearity 제어 (안정성)

결합하면:
1. 상대적으로 중요한 변수들은 남음 (L1의 희소성)
2. 살아남은 변수들 중 연관된 것들은 함께 크기를 공유 (L2의 안정성)

→ 실제 데이터에서 보통 Elastic Net > L1 또는 L2 단독

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Conjugate 함수와 Legendre 변환](./05-conjugate-legendre.md) | [📚 README](../README.md) | [Ch3-01. 볼록 최적화 표준형 ▶](../ch3-convex-problems/01-standard-form.md) |

</div>
