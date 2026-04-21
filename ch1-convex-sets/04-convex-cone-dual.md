# 04. 볼록 콘(Convex Cone)과 쌍대 콘

## 🎯 핵심 질문

- 볼록 콘이란 무엇이며 왜 볼록 집합 중 특별한 위치를 차지하는가?
- 쌍대 콘 $C^* = \{y \mid y^\top x \geq 0\ \forall x \in C\}$의 기하학적 의미는 무엇인가?
- 양의 정부호 콘 $\mathbb{S}^n_+$가 자기 쌍대(self-dual)인 이유는 무엇인가?
- SDP(반정부호 계획)와 SOCP(이차 콘 계획)는 왜 볼록 최적화의 계층에서 중요한 위치를 차지하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

볼록 콘은 볼록 최적화 문제 클래스의 계층 구조 — LP ⊂ QP ⊂ QCQP ⊂ SOCP ⊂ SDP — 의 기반이다.

- **SDP와 딥러닝**: 무한 폭 신경망의 학습이 SDP로 표현된다는 이론 (Neural Tangent Kernel의 선형화). SDP의 가능 영역은 $\mathbb{S}^n_+$ (PSD 콘) 위의 아핀 집합이다.
- **SOCP와 Robust Optimization**: 불확실성을 가진 데이터에서 안전한 분류기를 학습하는 Robust SVM이 SOCP 형태다. 이차 콘(SOC)은 Euclidean 노름 제약의 정확한 표현이다.
- **쌍대 콘과 KKT 조건**: SDP·SOCP의 KKT 조건에서 쌍대 변수는 쌍대 콘에 속해야 한다. $\mathbb{S}^n_+$의 자기 쌍대성은 SDP의 강쌍대성 분석을 간단하게 만든다.
- **Nuclear Norm Minimization (행렬 완성)**: $\min \|X\|_* = \sum \sigma_i(X)$ (핵 노름)는 행렬 랭크 최소화의 볼록 완화이며, 스펙트럼 노름 볼과 핵 노름 볼이 쌍대 관계다.

---

## 📐 수학적 선행 조건

- [01. 볼록 집합의 정의와 예제](./01-convex-set-definition.md): 볼록 집합의 기본 정의
- **내적과 대칭 행렬**: $\langle X, Y \rangle = \text{tr}(X^\top Y)$ (행렬 내적) — [Linear Algebra Deep Dive Ch3](https://github.com/iq-ai-lab/linear-algebra-deep-dive)
- **고유값 분해**: $A = Q\Lambda Q^\top$ — [Linear Algebra Deep Dive Ch5](https://github.com/iq-ai-lab/linear-algebra-deep-dive)
- **양의 정부호 행렬**: $A \succeq 0 \Leftrightarrow$ 모든 고유값 $\geq 0$ — [Linear Algebra Deep Dive Ch5](https://github.com/iq-ai-lab/linear-algebra-deep-dive)

---

## 📖 직관적 이해

### 콘의 시각적 이해

볼록 콘은 "원점에서 방사되는 볼록 집합"이다. 집합 안의 어떤 점도 임의의 양의 스케일링 후 다시 집합 안에 있다.

```
         이차 콘 (SOC)            비음수 직교체
              │                    y
       ╱╲     │ z                  ↑
      ╱  ╲    │                    │   볼록 콘
     ╱    ╲   │                    │  ╱
    ╱      ╲  │                    │ ╱
───────────── │               ─────┼─────── x
              │                    │
  √(x²+y²) ≤ z               x≥0, y≥0
```

### 쌍대 콘의 직관

$C$의 쌍대 콘 $C^*$는 "$C$의 모든 점과 90도 이내의 각도를 이루는 방향들의 집합"이다.

```
    C = 비음수 직교체            C* = 비음수 직교체 (자기 쌍대)
    y↑                           y↑
    │  ╱                         │  ╱
    │ ╱   C                      │ ╱   C*
    │╱                           │╱
────┼──── x                  ────┼──── x
```

쌍대 콘 $C^*$의 원소 $y$는 "$C$의 모든 방향 $x$와 내적이 ≥ 0" — 즉 $y$가 가리키는 방향은 $C$ 안의 어떤 방향과도 "엇나가지 않는다".

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 볼록 콘 (Convex Cone)

집합 $C \subseteq \mathbb{R}^n$이 **볼록 콘**이라는 것은:

$$\forall x_1, x_2 \in C,\ \forall \theta_1, \theta_2 \geq 0:\ \theta_1 x_1 + \theta_2 x_2 \in C$$

이것은 볼록성(볼록 결합 포함)과 콘 조건($\theta x \in C$ for $\theta \geq 0$)을 동시에 만족하는 것과 동치다.

> **주의**: 볼록 콘은 반드시 원점 $0$을 포함한다 ($\theta = 0$으로 놓으면).

### 정의 4.2 — 쌍대 콘 (Dual Cone)

$C \subseteq \mathbb{R}^n$의 **쌍대 콘**:

$$C^* = \{y \in \mathbb{R}^n \mid y^\top x \geq 0\ \forall x \in C\}$$

쌍대 콘은 항상 닫힌 볼록 콘이다 (부등식들의 교집합).

### 정의 4.3 — 자기 쌍대 콘 (Self-Dual Cone)

$C^* = C$이면 $C$를 **자기 쌍대**라 한다.

### 정의 4.4 — 진성 콘 (Proper Cone)

콘 $K$가 **진성 콘**이라는 것은:
1. 볼록이다
2. 닫혀 있다
3. 뾰족하다 (pointed): $x \in K$이고 $-x \in K$이면 $x = 0$
4. 내부가 있다 (solid): $\text{int}(K) \neq \emptyset$

진성 콘은 **일반화 부등식** $x \preceq_K y \Leftrightarrow y - x \in K$를 정의한다.

---

## 🔬 정리와 증명

### 주요 볼록 콘의 예제

#### 예제 1. 비음수 직교체 (Nonnegative Orthant)

$$\mathbb{R}^n_+ = \{x \in \mathbb{R}^n \mid x_i \geq 0\ \forall i\}$$

이것은 LP(선형 계획)의 변수 제약 $x \geq 0$에 해당하는 볼록 콘이다.

**쌍대 콘**: $(\mathbb{R}^n_+)^* = \mathbb{R}^n_+$ (자기 쌍대)

**증명**: $y \in (\mathbb{R}^n_+)^*$이면 $y^\top x \geq 0$ for all $x \geq 0$.  
$x = e_i$ (표준 기저 벡터)를 놓으면 $y_i = y^\top e_i \geq 0$. 따라서 $y \geq 0$.  
역으로 $y \geq 0$이면 $x \geq 0$에 대해 $y^\top x = \sum y_i x_i \geq 0$. $\square$

#### 예제 2. 이차 콘 (Second-Order Cone, SOC)

$$\mathcal{K}_n = \{(x, t) \in \mathbb{R}^n \times \mathbb{R} \mid \|x\|_2 \leq t\}$$

**볼록성 증명**: $(x_1, t_1), (x_2, t_2) \in \mathcal{K}_n$이면 $\|x_1\| \leq t_1$, $\|x_2\| \leq t_2$. $\theta_1, \theta_2 \geq 0$에 대해:

$$\|\theta_1 x_1 + \theta_2 x_2\| \leq \theta_1 \|x_1\| + \theta_2 \|x_2\| \leq \theta_1 t_1 + \theta_2 t_2$$

따라서 $(\theta_1 x_1 + \theta_2 x_2, \theta_1 t_1 + \theta_2 t_2) \in \mathcal{K}_n$. $\square$

**쌍대 콘**: $\mathcal{K}_n^* = \mathcal{K}_n$ (자기 쌍대)

**증명**:  
($\mathcal{K}_n \subseteq \mathcal{K}_n^*$): $(u, s) \in \mathcal{K}_n^*$이면 $(u,s)^\top (x,t) = u^\top x + st \geq 0$ for all $(x,t) \in \mathcal{K}_n$.  
$t = \|x\|$인 $(x, \|x\|)$를 넣으면 $u^\top x + s\|x\| \geq 0$ for all $x$.  
$x = u\|u\|^{-1}\cdot\|u\|$ (즉 $x = u$)로 놓으면 $\|u\|^2 + s\|u\| \geq 0$, 즉 $s + \|u\| \geq 0$ (or $s \geq -\|u\|$).  
Cauchy-Schwarz로 $u^\top x \geq -\|u\|\|x\|$이므로 $u^\top x + st \geq -\|u\|\|x\| + st \geq (-\|u\| + s)t \geq 0$ iff $s \geq \|u\|$. $\square$

#### 예제 3. 양의 준정부호 콘 (PSD Cone)

$$\mathbb{S}^n_+ = \{X \in \mathbb{S}^n \mid X \succeq 0\}$$

($\mathbb{S}^n$: $n \times n$ 실수 대칭 행렬의 집합)

**행렬 내적**: $\langle X, Y \rangle = \text{tr}(XY)$ ($X, Y$가 대칭행렬이면 $\text{tr}(XY) = \text{tr}(YX)$)

**쌍대 콘**: $(\mathbb{S}^n_+)^* = \mathbb{S}^n_+$ (자기 쌍대)

**증명**:  
($\mathbb{S}^n_+ \subseteq (\mathbb{S}^n_+)^*$): $Y \succeq 0$이면 임의의 $X \succeq 0$에 대해 $\text{tr}(XY) = \text{tr}(Y^{1/2}XY^{1/2}) \geq 0$ (∵ $Y^{1/2}XY^{1/2} \succeq 0$).

($(\mathbb{S}^n_+)^* \subseteq \mathbb{S}^n_+$): $Y \in (\mathbb{S}^n_+)^*$이면 $\text{tr}(XY) \geq 0$ for all $X \succeq 0$.  
$X = vv^\top$ ($v \in \mathbb{R}^n$) 으로 놓으면 $\text{tr}(vv^\top Y) = v^\top Y v \geq 0$ for all $v$.  
따라서 $Y \succeq 0$. $\square$

---

### 정리 4.1 — 쌍대 콘의 성질

**명제**: 임의의 집합 $C$에 대해:

1. $C^*$는 항상 닫힌 볼록 콘이다.
2. $C_1 \subseteq C_2 \Rightarrow C_2^* \subseteq C_1^*$ (포함 관계가 역전됨)
3. $(C^*)^* = \text{cl}(\text{conv}(C \cup \{0\}))$ (이중 쌍대)
4. $C$가 닫힌 볼록 콘이면 $(C^*)^* = C$

**증명 (1)**: $C^* = \bigcap_{x \in C} \{y \mid y^\top x \geq 0\}$는 반공간들의 교집합 — 닫힌 볼록 집합. 원점을 포함하고 스케일링에 닫혀 있으므로 볼록 콘. $\square$

**증명 (2)**: $y \in C_2^*$이면 $y^\top x \geq 0$ for all $x \in C_2$. $C_1 \subseteq C_2$이므로 $y^\top x \geq 0$ for all $x \in C_1$. 따라서 $y \in C_1^*$. $\square$

**증명 (4)**: Farkas lemma 또는 bipolar theorem으로 증명. 닫힌 볼록 콘에서 이중 쌍대는 자기 자신. $\square$

---

### 정리 4.2 — 일반화 부등식과 KKT 조건

진성 콘 $K$로 정의되는 일반화 부등식 $x \preceq_K 0$ ($-x \in K$)이 있는 최적화 문제:

$$\min f_0(x) \quad \text{s.t.}\ f_i(x) \preceq_{K_i} 0,\ Ax = b$$

의 KKT 조건에서 쌍대 변수 $\lambda_i$는 **쌍대 콘 $K_i^*$**에 속해야 한다:

$$\lambda_i \in K_i^*,\quad \lambda_i^\top f_i(x^*) = 0\ (\text{complementary slackness})$$

특히 SDP에서 $K = \mathbb{S}^n_+$이면 $K^* = \mathbb{S}^n_+$ (자기 쌍대)이므로 쌍대 변수도 PSD 행렬이어야 한다.

---

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# 1. 이차 콘 (SOC) 시각화
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(15, 5))

# SOC: {(x1, x2, t) | sqrt(x1² + x2²) ≤ t}
ax1 = fig.add_subplot(131, projection='3d')
theta_cone = np.linspace(0, 2*np.pi, 50)
t_vals = np.linspace(0, 2, 30)
T, THETA = np.meshgrid(t_vals, theta_cone)
X1 = T * np.cos(THETA)
X2 = T * np.sin(THETA)
T_surf = T
ax1.plot_surface(X1, X2, T_surf, alpha=0.4, color='steelblue')
ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$'); ax1.set_zlabel('$t$')
ax1.set_title('이차 콘 (SOC)\n$\\|x\\|_2 \\leq t$')

# ─────────────────────────────────────────────
# 2. 쌍대 콘 계산 및 시각화 (2D)
# ─────────────────────────────────────────────

def plot_cone_and_dual(ax, cone_angles, title, color1='steelblue', color2='orange'):
    """
    2D에서 콘과 쌍대 콘을 시각화
    cone_angles: (theta_min, theta_max) — 콘의 각도 범위 (라디안)
    """
    t_min, t_max = cone_angles
    r = 2

    # 콘 영역
    theta_fill = np.linspace(t_min, t_max, 100)
    cone_x = np.append([0], r * np.cos(theta_fill))
    cone_y = np.append([0], r * np.sin(theta_fill))
    ax.fill(cone_x, cone_y, alpha=0.35, color=color1, label='$C$')

    # 쌍대 콘: y^T x ≥ 0 for all x ∈ C
    # 2D에서: 쌍대 콘의 각도 범위 계산
    # x ∈ C의 범위: [t_min, t_max], y는 x와 내적 ≥ 0인 방향
    # 이는 y의 각도가 C의 법선 방향에서 [t_min - π/2, t_max + π/2] 내에 있음
    # 쌍대 콘의 각도: [t_max - π, t_min + π] (대략)
    # 정확히는: dual cone = {y | t_max - π/2 ≤ angle(y) ≤ t_min + π/2 + π}
    # 단순 버전: 직접 계산
    dual_min = t_max - np.pi
    dual_max = t_min + np.pi
    # 정상화
    theta_dual = np.linspace(dual_min, dual_max, 100)
    dual_x = np.append([0], r * np.cos(theta_dual))
    dual_y = np.append([0], r * np.sin(theta_dual))
    ax.fill(dual_x, dual_y, alpha=0.25, color=color2, label='$C^*$')

    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.legend(fontsize=9); ax.set_title(title, fontsize=10)

ax2 = fig.add_subplot(132)
# 비음수 직교체: [0, π/2], 쌍대는 자기 자신
plot_cone_and_dual(ax2, (0, np.pi/2), '비음수 직교체 $\\mathbb{R}^2_+$\n자기 쌍대 ✓')

ax3 = fig.add_subplot(133)
# 60도 콘: [0, π/3]
plot_cone_and_dual(ax3, (-np.pi/6, np.pi/3), '60도 볼록 콘\n쌍대 콘 (보라색)')

plt.suptitle('볼록 콘과 쌍대 콘', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('04-convex-cone-dual.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. PSD 콘의 자기 쌍대성 수치 검증
# ─────────────────────────────────────────────

def is_psd(M, tol=1e-8):
    """양의 준정부호 여부 확인"""
    return np.all(np.linalg.eigvalsh(M) >= -tol)

def check_psd_self_dual(n=3, n_trials=1000):
    """
    Y ∈ S^n_+ iff tr(XY) ≥ 0 for all X ∈ S^n_+ 수치 검증
    """
    violations = 0
    for _ in range(n_trials):
        # PSD 행렬 Y 생성
        V = np.random.randn(n, n)
        Y = V @ V.T  # PSD
        
        # PSD 행렬 X 생성
        U = np.random.randn(n, n)
        X = U @ U.T  # PSD
        
        inner = np.trace(X @ Y)
        if inner < -1e-8:
            violations += 1
    
    # 비PSD Y에 대한 검증
    violations_non_psd = 0
    for _ in range(n_trials):
        # 비PSD 행렬 Y 생성 (음의 고유값 포함)
        Y = np.random.randn(n, n)
        Y = Y + Y.T  # 대칭
        while is_psd(Y):  # PSD가 아닌 것 찾기
            Y = np.random.randn(n, n)
            Y = Y + Y.T
        
        # 반례 찾기: Y의 음의 고유벡터 v에 대해 X = vv^T
        eigvals, eigvecs = np.linalg.eigh(Y)
        neg_idx = np.argmin(eigvals)
        v = eigvecs[:, neg_idx]
        X = np.outer(v, v)  # PSD (rank-1)
        
        inner = np.trace(X @ Y)
        if inner < -1e-8:
            violations_non_psd += 1

    print(f"PSD Y에 대한 tr(XY) < 0 위반 (기대: 0): {violations}")
    print(f"비PSD Y에 대한 tr(XY) < 0 반례 발견 (기대: {n_trials}): {violations_non_psd}")

check_psd_self_dual()
```

**출력**:
```
PSD Y에 대한 tr(XY) < 0 위반 (기대: 0): 0
비PSD Y에 대한 tr(XY) < 0 반례 발견 (기대: 1000): 1000
```

---

## 🔗 AI/ML 연결

### SDP와 딥러닝의 관계

**Neural Network Verification**: 신경망의 안전성 검증(어떤 입력 집합에 대해 출력이 특정 조건을 만족하는지)은 SDP 완화로 풀 수 있다. 비선형 활성화 함수를 선형 부등식으로 근사하면 가능 영역이 LMI(Linear Matrix Inequality) — $\mathbb{S}^n_+$ 위의 아핀 제약.

**LMI 형태**:

$$F(x) = F_0 + x_1 F_1 + \cdots + x_n F_n \succeq 0$$

이 형태의 제약이 있는 최적화가 SDP이며, $F(x)$가 $\mathbb{S}^n_+$ 콘 위에 놓이는 조건이다.

### Robust Optimization과 이차 콘

불확실한 데이터 $a \in \mathcal{U} = \{a_0 + P u \mid \|u\| \leq 1\}$에 대해 안전한 결정을 하는 Robust Linear Program:

$$\min c^\top x \quad \text{s.t.}\ a^\top x \leq b\ \forall a \in \mathcal{U}$$

는 다음 SOCP로 변환된다:

$$\min c^\top x \quad \text{s.t.}\ a_0^\top x + \|P^\top x\|_2 \leq b$$

제약 $\|P^\top x\|_2 \leq b - a_0^\top x$가 이차 콘 조건이다.

---

## ⚖️ 가정과 한계

| 개념 | 핵심 성질 | 한계 |
|------|---------|------|
| 비음수 직교체 $\mathbb{R}^n_+$ | 자기 쌍대, LP의 기반 | 정수 제약 불가 |
| 이차 콘 (SOC) | 자기 쌍대, Euclidean 노름 표현 | 고차 노름 표현 불가 |
| PSD 콘 $\mathbb{S}^n_+$ | 자기 쌍대, SDP의 기반 | 계산 복잡도 $O(n^3)$ per iteration |
| 일반 볼록 콘 | 일반화 부등식 정의 | 쌍대 콘이 자기 쌍대가 아닐 수 있음 |

**표현력의 계층**:

$$\text{LP (선형)} \subset \text{SOCP (이차 콘)} \subset \text{SDP (PSD 콘)}$$

각 계층은 위 콘의 특수 경우다: LP는 $\mathbb{R}^n_+$, SOCP는 SOC, SDP는 $\mathbb{S}^n_+$.

---

## 📌 핵심 정리

$$C^* = \{y \mid y^\top x \geq 0\ \forall x \in C\}$$

| 콘 $C$ | 쌍대 콘 $C^*$ | 자기 쌍대 |
|--------|------------|---------|
| $\mathbb{R}^n_+$ (비음수 직교체) | $\mathbb{R}^n_+$ | ✓ |
| $\mathcal{K}_n$ (이차 콘) | $\mathcal{K}_n$ | ✓ |
| $\mathbb{S}^n_+$ (PSD 콘) | $\mathbb{S}^n_+$ | ✓ |
| $\{x \mid Ax = 0\}$ (선형 부분공간) | $\text{Im}(A^\top)$ (행공간) | 일반적으로 ✗ |

**이중 쌍대 정리**: 닫힌 볼록 콘 $C$에 대해 $(C^*)^* = C$.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\ell_1$ 노름 콘 $\{(x, t) \mid \|x\|_1 \leq t\}$이 볼록 콘임을 증명하고, 그 쌍대 콘을 구하라.

<details>
<summary>힌트 및 해설</summary>

**볼록 콘 증명**: $\theta_1, \theta_2 \geq 0$에 대해 $\|\theta_1 x_1 + \theta_2 x_2\|_1 \leq \theta_1\|x_1\|_1 + \theta_2\|x_2\|_1 \leq \theta_1 t_1 + \theta_2 t_2$.

**쌍대 콘**: $\{(y, s) \mid \|y\|_\infty \leq s\}$ ($\ell_\infty$ 노름 콘).

증명: $(y,s)^\top(x,t) = y^\top x + st \geq 0$ for all $(x,t)$, $\|x\|_1 \leq t$.  
$t = \|x\|_1$로 놓으면 $y^\top x + s\|x\|_1 \geq 0$ for all $x$.  
Hölder 부등식: $|y^\top x| \leq \|y\|_\infty \|x\|_1$이므로 $s \geq \|y\|_\infty$이면 성립.  
반대 방향: $\|y\|_\infty > s$이면 $|y_i| > s$인 $i$에 대해 $x = \text{sign}(y_i) e_i$를 넣으면 $y^\top x + s \cdot 1 = |y_i| + s < 0$ (부호에 따라). 따라서 $\|y\|_\infty \leq s$ 필요.

</details>

**문제 2** (심화): 행렬 핵 노름 $\|X\|_* = \sum_i \sigma_i(X)$ (특이값의 합)과 스펙트럴 노름 $\|X\|_2 = \sigma_{\max}(X)$이 쌍대 노름임을 쌍대 콘을 이용하여 설명하라.

<details>
<summary>힌트 및 해설</summary>

핵 노름 볼 $\{X \mid \|X\|_* \leq 1\}$과 스펙트럴 노름 볼 $\{Y \mid \|Y\|_2 \leq 1\}$은 쌍대 집합이다.  
$\langle X, Y \rangle = \text{tr}(X^\top Y) \leq \|X\|_* \|Y\|_2$ (von Neumann 부등식).  
따라서 $(\{X \mid \|X\|_* \leq 1\})^* = \{Y \mid \max_{\|X\|_*\leq 1} \langle X,Y\rangle \leq 1\} = \{Y \mid \|Y\|_2 \leq 1\}$.  
행렬 완성(Matrix Completion): $\min \|X\|_*$ subject to 관찰된 원소 일치 — 핵 노름이 SDP로 변환 가능한 것은 핵 노름 볼이 $\mathbb{S}^n_+$와 연결되어 있기 때문.

</details>

**문제 3** (AI 연결): SOCP의 가능 영역 $\{x \mid \|Ax + b\|_2 \leq c^\top x + d\}$이 볼록 집합임을 이차 콘과 아핀 변환의 역상(preimage)을 이용하여 증명하라.

<details>
<summary>힌트 및 해설</summary>

이차 콘 $\mathcal{K} = \{(u, t) \mid \|u\| \leq t\}$은 볼록 집합이다.  
아핀 함수 $f(x) = (Ax+b, c^\top x + d)$에 대해 가능 영역 $= f^{-1}(\mathcal{K})$.  
볼록 집합의 아핀 역상은 볼록이므로 (정리 2.3), 가능 영역은 볼록이다. $\square$

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. 분리 초평면 정리](./03-separating-hyperplane.md) | [📚 README](../README.md) | [05. Extreme Point와 Krein-Milman 정리 ▶](./05-extreme-point-krein-milman.md) |

</div>
