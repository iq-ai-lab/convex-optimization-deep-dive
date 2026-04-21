# 2. LP, QP, QCQP, SOCP, SDP의 계층

## 🎯 핵심 질문
볼록 최적화 문제들 사이에는 포함 관계(hierarchy)가 있는가? 각 문제 클래스의 계산 복잡도는 어떻게 다른가? 내 ML 문제는 어느 클래스에 속하는가?

## 🔍 왜 이 이론이 AI에서 중요한가
"문제를 알면 알고리즘을 안다" - 각 문제 클래스는 특화된 솔버(solver)를 가집니다. SVM은 QP, Robust Optimization은 SOCP, Semidefinite Relaxation은 SDP로 푸는 것이 효율적입니다. 문제의 구조를 인식하는 것이 최적화의 핵심입니다.

## 📐 수학적 선행 조건
- 표준형과 볼록성의 정의 (Ch3-01)
- 이차형식과 반정치 행렬
- 각 함수 클래스의 볼록성 증명

## 📖 직관적 이해
**문제 클래스의 계층도**:
```
                    SDP (가장 일반적)
                    |
                  SOCP
                 /    \
              QCQP    (robust opt)
              |
              QP      (SVM, portfolio)
              |
              LP      (최적화의 기초)
```

- **LP**: 평면적 최적화, 최적해 = 꼭짓점 (Simplex)
- **QP**: 포물면 최적화, SVM 분류경계 계산
- **SOCP**: 원뿔 형태 제약, 불확실성 모델링
- **SDP**: 행렬의 반정치성 제약, 최강의 이완(relaxation)

## ✏️ 엄밀한 정의

### 정의 1: 선형계획법 (Linear Program, LP)
$$\begin{align}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & Ax \leq b, \quad Ex = f
\end{align}$$

**특징**:
- 목적함수와 모든 제약이 **아핀함수**
- 가능 영역: 다면체(polytope) 
- 최적해: 다면체의 꼭짓점 중 하나 (vertex solution)
- 계산 복잡도: **다항시간** (Simplex 또는 Interior Point Method)

### 정의 2: 이차계획법 (Quadratic Program, QP)
$$\begin{align}
\text{minimize} \quad & \frac{1}{2}x^T P x + q^T x \\
\text{subject to} \quad & Ax \leq b, \quad Ex = f
\end{align}$$

**조건**: $P \succeq 0$ (반정치)

**특징**:
- 목적함수: 이차 (convex paraboloid)
- 제약: 선형
- 최적해 조건: KKT 조건이 충분조건
- 계산: $O(n^3)$ 시간, 매우 효율적

**ML 예**: SVM (soft-margin)
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \, \xi_i \geq 0$$

### 정의 3: 이차제약 이차계획법 (Quadratically Constrained Quadratic Program, QCQP)
$$\begin{align}
\text{minimize} \quad & \frac{1}{2}x^T P_0 x + q_0^T x \\
\text{subject to} \quad & \frac{1}{2}x^T P_i x + q_i^T x \leq r_i, \quad i = 1, \ldots, m \\
& Ax = b
\end{align}$$

**조건**: $P_i \succeq 0$ for all $i$

**특징**:
- 목적함수와 제약 모두 **이차**
- QP의 자연스러운 확장
- 계산: $O(m n^2)$ 시간대

**예**: 에너지 최소화
$$\min \|Ax - b\|_2^2 \quad \text{s.t.} \quad \|x\|_2^2 \leq r^2$$

### 정의 4: 이계원뿔 계획법 (Second-Order Cone Program, SOCP)
$$\begin{align}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & \|A_i x + b_i\|_2 \leq c_i^T x + d_i, \quad i = 1, \ldots, m \\
& Ex = f
\end{align}$$

**특징**:
- 제약: **이계원뿔** (Second-Order Cone) 형태
  $$\text{SOC}_n = \{(x, t) \in \mathbb{R}^{n-1} \times \mathbb{R} \mid \|x\|_2 \leq t\}$$
- 노름 제약을 표현 가능
- 계산: IPM으로 다항시간
- **강점**: 불확실성 모델링

**예**: Robust Linear Programming
$$\min c^T x \quad \text{s.t.} \quad \|(A + \Delta A)x - b\|_2 \leq \epsilon$$

여기서 $\Delta A$는 불확실한 행렬

### 정의 5: 반정치 계획법 (Semidefinite Program, SDP)
$$\begin{align}
\text{minimize} \quad & c^T x \\
\text{subject to} \quad & F(x) := F_0 + \sum_{i=1}^n x_i F_i \succeq 0 \\
& Ax = b
\end{align}$$

**특징**:
- 제약: **행렬의 반정치성** $F(x) \succeq 0$
- $F_i \in \mathbb{S}^p$ (대칭행렬)
- 계산: IPM으로 다항시간 (하지만 상수는 큼)
- **가장 일반적**: QCQP, SOCP ⊂ SDP

**예 1**: 행렬 완성 (Matrix Completion)
$$\min_X \|X\|_* \quad \text{s.t.} \quad X_{ij} = M_{ij} \quad \text{for} \quad (i,j) \in \Omega$$

여기서 $\|X\|_*$는 핵 노름 (nuclear norm) - SDP로 표현 가능

**예 2**: 동기화 문제 (Synchronization)
$$\min_R \|R^T R - I\|_F^2 \quad \text{(비볼록)} \rightarrow \max \text{tr}(F R) \quad \text{s.t.} \quad R \succeq 0, \text{diag}(R)=1 \quad \text{(SDP 완화)}$$

## 🔬 정리와 증명

### 정리 1: 포함 관계 $LP \subseteq QP \subseteq QCQP \subseteq SOCP \subseteq SDP$

**명제**: 모든 LP는 QP이다.

**증명**: LP 문제 $\min c^T x$ s.t. $Ax \leq b$는 QP의 특수한 경우로 볼 수 있습니다.
$$\min \frac{1}{2}x^T (0 \cdot I) x + c^T x = \min c^T x$$
여기서 $P = 0 \succeq 0$ (반정치 만족).

---

**명제**: 모든 QP는 SOCP이다.

**증명**: QP의 이차 목적함수 $\frac{1}{2}\|x\|_Q^2 + q^T x$는 다음과 같이 SOCP로 재작성:
$$\min t \quad \text{s.t.} \quad \left\|\begin{pmatrix} L x \\ q^T x / \|q\| + t \end{pmatrix}\right\|_2 \leq \text{const}$$

여기서 $Q = L^T L$ (Cholesky 분해)

---

**명제**: 모든 SOCP는 SDP이다.

**증명**: SOCP 제약 $\|A_i x + b_i\|_2 \leq c_i^T x + d_i$는 다음 SDP 제약과 동치:
$$\begin{pmatrix} (c_i^T x + d_i) I & A_i^T x + b_i^T \\ A_i x + b_i & c_i^T x + d_i \end{pmatrix} \succeq 0$$

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

print("="*70)
print("1. Linear Program (LP)")
print("="*70)

# LP: min c^T x s.t. Ax <= b
c = np.array([1, 2])  # minimize x1 + 2*x2
A = np.array([[1, 1], [2, 1], [1, 2]])
b = np.array([4, 5, 4])

x = cp.Variable(2)
objective_lp = cp.Minimize(c @ x)
constraints_lp = [A @ x <= b, x >= 0]
prob_lp = cp.Problem(objective_lp, constraints_lp)
prob_lp.solve(solver=cp.GLPK)

print(f"최적값: {prob_lp.value:.6f}")
print(f"최적점: {x.value}")

print("\n" + "="*70)
print("2. Quadratic Program (QP)")
print("="*70)

# QP: min 0.5*x^T P x + q^T x s.t. Ax <= b
P = np.array([[2, 0], [0, 2]])  # identity, positive definite
q = np.array([-4, -2])
A = np.array([[1, 1]])
b = np.array([3])

x = cp.Variable(2)
objective_qp = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)
constraints_qp = [A @ x <= b, x >= 0]
prob_qp = cp.Problem(objective_qp, constraints_qp)
prob_qp.solve(solver=cp.SCS)

print(f"최적값: {prob_qp.value:.6f}")
print(f"최적점: {x.value}")

print("\n" + "="*70)
print("3. Quadratically Constrained QP (QCQP)")
print("="*70)

# QCQP: min 0.5*x^T P0 x + q0^T x s.t. ||x||_2 <= r
P0 = np.eye(2)
q0 = np.array([-2, -1])
r = 1.5

x = cp.Variable(2)
objective_qcqp = cp.Minimize(0.5 * cp.quad_form(x, P0) + q0 @ x)
constraints_qcqp = [cp.norm(x, 2) <= r]
prob_qcqp = cp.Problem(objective_qcqp, constraints_qcqp)
prob_qcqp.solve(solver=cp.SCS)

print(f"최적값: {prob_qcqp.value:.6f}")
print(f"최적점: {x.value}")
print(f"제약 확인 (||x||_2): {np.linalg.norm(x.value):.6f} <= {r}")

print("\n" + "="*70)
print("4. Second-Order Cone Program (SOCP)")
print("="*70)

# SOCP: min c^T x s.t. ||A_i x + b_i||_2 <= c_i^T x + d_i
c = np.array([1, 0])  # minimize x1
A_cone = np.array([[1, 0], [0, 1]])
b_cone = np.zeros(2)
c_cone = np.array([0, 1])
d_cone = 1

x = cp.Variable(2)
objective_socp = cp.Minimize(c @ x)
constraints_socp = [
    cp.norm(A_cone @ x + b_cone, 2) <= c_cone @ x + d_cone,
    x >= 0
]
prob_socp = cp.Problem(objective_socp, constraints_socp)
prob_socp.solve(solver=cp.SCS)

print(f"최적값: {prob_socp.value:.6f}")
print(f"최적점: {x.value}")

print("\n" + "="*70)
print("5. Semidefinite Program (SDP)")
print("="*70)

# SDP: min c^T x s.t. F0 + x1*F1 + x2*F2 >= 0
F0 = np.array([[-1, 0], [0, -1]])
F1 = np.array([[1, 0], [0, 0]])
F2 = np.array([[0, 0], [0, 1]])
c = np.array([1, 1])

x = cp.Variable(2)
mat_const = F0 + x[0]*F1 + x[1]*F2
objective_sdp = cp.Minimize(c @ x)
constraints_sdp = [mat_const >> 0]  # >> means semidefinite
prob_sdp = cp.Problem(objective_sdp, constraints_sdp)
prob_sdp.solve(solver=cp.SCS)

print(f"최적값: {prob_sdp.value:.6f}")
print(f"최적점: {x.value}")

# 시각화: 각 문제의 최적점
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

xx = np.linspace(-1, 5, 200)
yy = np.linspace(-1, 5, 200)
XX, YY = np.meshgrid(xx, yy)

# LP
Z_lp = c[0]*XX + c[1]*YY
ax = axes[0]
contour = ax.contour(XX, YY, Z_lp, levels=15, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)
ax.fill([0, 4, 0], [0, 0, 4], alpha=0.3, color='green')
ax.plot(x.value[0], x.value[1], 'r*', markersize=15)
ax.set_title('LP: min $c^T x$', fontweight='bold')
ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 5)
ax.grid(True, alpha=0.3)

# QP
ax = axes[1]
Z_qp = 0.5*(XX**2 + YY**2) - 4*XX - 2*YY
contour = ax.contour(XX, YY, Z_qp, levels=15, cmap='viridis')
ax.fill([0, 3, 0], [0, 0, 3], alpha=0.3, color='green')
ax.plot(2, 1, 'r*', markersize=15, label='최적점 (2, 1)')
ax.set_title('QP: min $\\frac{1}{2}||x||^2 - 4x_1 - 2x_2$', fontweight='bold')
ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, 5)
ax.grid(True, alpha=0.3)

# QCQP
ax = axes[2]
circle = plt.Circle((0, 0), r, color='green', alpha=0.3)
ax.add_patch(circle)
ax.plot(x.value[0], x.value[1], 'r*', markersize=15)
ax.set_title(f'QCQP: min $||x||^2$ s.t. $||x|| \\leq {r}$', fontweight='bold')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# SOCP
ax = axes[3]
ax.plot(x.value[0], x.value[1], 'r*', markersize=15, label='최적점')
ax.set_title('SOCP: 원뿔 제약', fontweight='bold')
ax.set_xlim(-0.5, 3)
ax.set_ylim(-0.5, 3)
ax.grid(True, alpha=0.3)

# SDP
ax = axes[4]
ax.plot(x.value[0], x.value[1], 'r*', markersize=15, label='최적점')
ax.set_title('SDP: 행렬 반정치 제약', fontweight='bold')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True, alpha=0.3)

# 계층 관계
ax = axes[5]
ax.text(0.5, 0.9, "문제 계층도", ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.7, "SDP ⊇ SOCP ⊇ QCQP", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.5, "   ⊇ QP ⊇ LP", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.25, "더 제한적 → 더 빠른 솔버", ha='center', fontsize=9, 
        style='italic', transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig('/tmp/problem_hierarchy.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("계층 관계 요약")
print("="*70)
print("LP  : 선형 목적함수, 선형 제약 (가장 빠름)")
print("QP  : 이차 목적함수, 선형 제약")
print("QCQP: 이차 목적함수, 이차 제약")
print("SOCP: 선형 목적함수, 노름 제약 (불확실성)")
print("SDP : 가장 일반적, 가장 느림 (하지만 표현력 최고)")
```

## 🔗 AI/ML 연결

| 문제 클래스 | ML 응용 | 표준 솔버 | 시간복잡도 |
|-----------|--------|---------|----------|
| **LP** | 선형 분류, 자원 배분 | Simplex, IPM | $O(n^2m)$ 또는 다항시간 |
| **QP** | SVM (soft-margin), 이차 회귀 | Interior Point | $O(n^3)$ |
| **QCQP** | 제약 최소제곱 | IPM | $O(mn^2)$ |
| **SOCP** | Robust ML, worst-case opt | IPM | $O(m n^{2.5})$ |
| **SDP** | 동기화, 행렬 완성, NTK | IPM | $O(n^{3.5})$ 내외 |

## ⚖️ 가정과 한계

**가정:**
1. 모든 함수가 미분가능 (대부분 만족)
2. 반정치 조건이 명시적으로 만족됨
3. 제약이 Slater 조건 만족 (강한 쌍대성)

**한계:**
- **확장성**: SDP는 수백 개 변수 이상 어려움
- **비볼록 최적화**: 정수 계획, 신경망은 여기 계층에 포함 안 됨
- **비미분**: 절댓값, max 등은 재구성 필요

## 📌 핵심 정리

**문제 계층**: $LP \subset QP \subset QCQP \subset SOCP \subset SDP$

| 클래스 | 목적함수 | 제약 | 특징 |
|-------|--------|------|------|
| LP | 아핀 | 아핀 | 최적해 = 꼭짓점 |
| QP | 이차 | 아핀 | SVM의 기본 형태 |
| QCQP | 이차 | 이차 | 모수 의존 최적화 |
| SOCP | 아핀 | 노름 (cone) | 불확실성 모델링 |
| SDP | 아핀 | 행렬 반정치 | 최강의 이완 |

## 🤔 생각해볼 문제

**문제 1**: 다음 SVM 문제가 QP임을 보이고 표준 QP 형태로 쓰시오.
$$\min_{w,b,\xi} \frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^m \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

<details>
<summary>힌트 및 해설</summary>

변수를 $x = [w; b; \xi]$로 벡터화하면:
- $P = \text{diag}[I_n, 0, 0]$ (첫 $n$개만 이차)
- $q = [0; 0; C\mathbf{1}]$
- 부등식 제약을 다시 쓰면: $-y_i(w^T x_i + b) + 1 - \xi_i \leq 0$

이는 정확히 QP의 표준형입니다.

</details>

**문제 2**: 왜 SOCP가 불확실성(robust optimization)을 다루기에 적합한가? Robust LP 예제를 들고 SOCP로 변환하시오.

<details>
<summary>힌트 및 해설</summary>

표준 LP: $\min c^T x$ s.t. $A x \leq b$

불확실한 행렬: $A + \Delta A$, $\|\Delta A\| \leq \epsilon$

Robust 버전: $\min c^T x$ s.t. $(A + \Delta A)x \leq b$ for all $\|\Delta A\| \leq \epsilon$

이는 동치적으로:
$$\min c^T x \quad \text{s.t.} \quad \max_{\|\Delta A\|} (A+\Delta A)x \leq b$$

Cauchy-Schwarz로:
$$\max_{\|\Delta A\| \leq \epsilon} \|(A+\Delta A)x\| = \|Ax\| + \epsilon \|x\|$$

따라서: $\min c^T x$ s.t. $\|Ax\| + \epsilon\|x\| \leq b$ → **SOCP**!

</details>

**문제 3**: CVXPY에서 다음 QCQP 문제를 풀고, 제약이 활성화(active)되었는지 확인하시오.

```python
import cvxpy as cp
import numpy as np

# QCQP: min ||x - [1,2]||^2 s.t. ||x||^2 <= 0.5
x = cp.Variable(2)
obj = cp.Minimize(cp.sum_squares(x - np.array([1, 2])))
constr = [cp.norm(x, 2) <= 0.5]
prob = cp.Problem(obj, constr)
prob.solve(cp.SCS)

# 최적점에서 제약 ||x||^2 = ?
print(f"최적점: {x.value}")
print(f"제약값 ||x||: {np.linalg.norm(x.value):.6f}")
```

<details>
<summary>힌트 및 해설</summary>

목적함수 최솟값은 $[1, 2]$에서 이루어지지만, 제약 $\|x\| \leq 0.5$ 때문에 최적점은 원점에서 반경 0.5인 원 위에 있습니다.

$[1,2]$에서 원점까지의 거리는 $\sqrt{5} \approx 2.236 > 0.5$이므로 제약이 활성화됩니다.

최적점: $x^* = 0.5 \cdot \frac{[1,2]}{\sqrt{5}} = [0.2236, 0.4472]$ (근사)

제약값: $\|x^*\| = 0.5$ (정확히 경계에 있음)

</details>

<div align="center">
| [◀ 01. 볼록 최적화 문제의 표준형](./01-standard-form.md) | [📚 README](../README.md) | [03. Geometric Programming ▶](./03-geometric-programming.md) |
</div>
