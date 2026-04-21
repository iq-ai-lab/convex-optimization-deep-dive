# 5. Online Convex Optimization과 Regret 경계

## 🎯 핵심 질문

온라인 학습(online learning)에서는 손실함수가 시간에 따라 변한다. 미리 알 수 없는 적대적 손실 앞에서 어떻게 최선을 다할 수 있는가? **Regret**이란 무엇이고, 얼마나 작게 만들 수 있는가?

## 🔍 왜 이 이론이 AI에서 중요한가

1. **스트리밍 데이터**: 실시간 의사결정 (금융, 광고 입찰 등)
2. **적응 알고리즘**: 시간 변화하는 환경에서의 학습
3. **강화 학습의 기초**: 탐색-활용(exploration-exploitation) 트레이드오프
4. **이론적 보장**: Regret 경계로 최악의 경우 성능 보장
5. **효율성**: SGD보다 더 일반적인 알고리즘

## 📐 수학적 선행 조건

- **누적 손실 (cumulative loss)**: $\sum_t f_t(x_t)$
- **Regret**: 온라인 알고리즘 vs 최적 고정 결정의 차이
- **Projection**: Πc(x) = argmin_y∈C ‖x - y‖
- **Convex conjugate**: f*(g) = max_x (gᵀx - f(x))

## 📖 직관적 이해

### Online Convex Optimization (OCO) 프레임워크

**게임:**
- T 라운드 진행
- 라운드 t마다:
  1. 알고리즘이 $x_t \in C$ 선택
  2. 자연(adversary)이 손실함수 $f_t: C \to \mathbb{R}$ 공개 (또는 후속 이용 가능)
  3. 알고리즘이 손실 $f_t(x_t)$ 관측

**목표:** 누적 손실을 최소화하되, 사전에 f_t를 모르는 상황에서.

### Regret 정의

$$\text{Regret}(T) = \sum_{t=1}^T f_t(x_t) - \min_{x^* \in C} \sum_{t=1}^T f_t(x^*)$$

**해석:**
- 좌변: 온라인 알고리즘의 누적 손실
- 우변: 최적 고정 결정 x*의 누적 손실
- Regret > 0: 온라인 알고리즘이 최선보다 못함

**목표:** Regret(T) = o(T) 달성 (평균 손실 → 0 as T → ∞)

### Online Gradient Descent (OGD)

**알고리즘:**
$$x_{t+1} = \Pi_C(x_t - \eta_t g_t)$$

여기서:
- $\eta_t$: 라운드 t의 학습률
- $g_t \in \partial f_t(x_t)$: 부분미분
- $\Pi_C$: 제약 집합 C로의 사영

**학습률 스케줄:** $\eta_t = \frac{1}{G\sqrt{t}}$ (G: 그레디언트 상한)

## ✏️ 엄밀한 정의

### Online Convex Optimization 형식

**설정:**
- 볼록 집합 $C \subseteq \mathbb{R}^d$, 직경 $D = \max_{x,y \in C} \|x-y\|$
- T개 라운드, 매 라운드마다 미리 모르는 볼록함수 $f_t: C \to \mathbb{R}$
- 그레디언트 경계: $\|g_t\| \leq G$ for all $g_t \in \partial f_t(x_t)$

**Online GD:**
$$x_{t+1} = \Pi_C(x_t - \eta g_t)$$

### Regret 분석

**정리 (O(√T) Regret for OGD):**

$\eta = \frac{D}{G\sqrt{T}}$로 설정하면,

$$\text{Regret}(T) \leq DG\sqrt{T}$$

**증명:**

Step 1: 정류 조건 (Descent Lemma)
$$\|x_{t+1} - x^*\|^2 \leq \|x_t - x^*\|^2 - 2\eta g_t^T(x_t - x^*) + \eta^2\|g_t\|^2$$

Step 2: 볼록성
$$f_t(x_t) \leq f_t(x^*) + g_t^T(x_t - x^*)$$

따라서
$$g_t^T(x_t - x^*) \geq f_t(x_t) - f_t(x^*)$$

Step 3: 결합
$$f_t(x_t) - f_t(x^*) \leq \frac{1}{2\eta}(\|x_t - x^*\|^2 - \|x_{t+1} - x^*\|^2) + \frac{\eta G^2}{2}$$

Step 4: 망원급 (Telescoping)
$$\sum_{t=1}^T (f_t(x_t) - f_t(x^*)) \leq \frac{\|x_1 - x^*\|^2}{2\eta} + \frac{\eta T G^2}{2}$$

최적 η로 최소화: $\eta = \frac{D}{G\sqrt{T}}$

$$\text{Regret} \leq DG\sqrt{T}$$

### AdaGrad: 적응 학습률

**기본 아이디어:** 좌표마다 다른 학습률 사용 (희소 그레디언트에 효율적)

**업데이트:**
$$x_{t+1,i} = \Pi_{C_i}\left( x_{t,i} - \frac{\eta}{\sqrt{\sum_{s=1}^t g_{s,i}^2}} g_{t,i} \right)$$

**이점:**
- 자주 업데이트되는 좌표: 학습률 감소
- 드물게 업데이트되는 좌표: 학습률 유지
- 희소 데이터에서 효율적

**Regret 경계:**
$$\text{Regret}_{\text{AdaGrad}}(T) = O(G_\infty \sqrt{T} \log T)$$

여기서 $G_\infty = \max_i \sum_t |g_{t,i}|$ (좌표별 누적 손실)

## 🔬 정리와 증명

위의 "✏️ 엄밀한 정의" 섹션에서 O(√T) Regret 증명과 AdaGrad 정의를 제시했다. 이제 핵심 수렴 성질을 재확인한다.

**핵심 보조정리 (Key Lemma):**
$$\sum_{t=1}^T (f_t(x_t) - f_t(x^*)) \leq \frac{\|x_1 - x^*\|^2}{2\eta} + \frac{\eta T G^2}{2}$$

이 부등식은 Descent Lemma와 Telescoping으로부터 유도된다. η = D/(G√T)를 대입하면 O(√T) 경계를 얻는다.

**AdaGrad의 이점:**
- 희소 그레디언트: 많은 gₜ,ᵢ = 0 → 학습률 감소 안 함
- 일반적 그레디언트: 경계 G∞ < ∞∑ₜ|gₜ,ᵢ| 하에서 O(√T log T)

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Online Convex Optimization 시뮬레이션

np.random.seed(42)

# 설정
T = 1000  # 라운드 수
d = 10    # 차원
C_radius = 1.0  # 제약 집합: ℓ2 공 (반경 1)

# 고정 최적 해 (비교 기준)
x_opt = np.ones(d) / np.sqrt(d)  # 단위 구 위의 점

# 1. Online Gradient Descent
def sgd_online(T, d, C_radius, get_loss_and_grad):
    """Online GD with decreasing learning rate"""
    x_t = np.zeros(d)
    cumulative_loss = []
    cumulative_regret = []
    
    opt_loss = 0  # 최적 해의 손실 누적
    
    for t in range(1, T+1):
        loss_t, grad_t = get_loss_and_grad(x_t, t)
        
        # 학습률: η_t = D / (G * sqrt(T))
        G = 1.0  # 그레디언트 상한
        D = 2 * C_radius  # 직경
        eta_t = D / (G * np.sqrt(T))
        
        # GD 업데이트
        x_new = x_t - eta_t * grad_t
        
        # 사영 (ℓ2 공으로)
        norm_x = np.linalg.norm(x_new)
        if norm_x > C_radius:
            x_new = x_new / norm_x * C_radius
        
        x_t = x_new
        
        # 손실 누적
        cumulative_loss.append(loss_t)
        
        # Regret 계산
        opt_loss_t, _ = get_loss_and_grad(x_opt, t)
        opt_loss += opt_loss_t
        
        total_loss = np.sum(cumulative_loss)
        regret = total_loss - opt_loss
        cumulative_regret.append(regret)
    
    return np.array(cumulative_loss), np.array(cumulative_regret), x_t

# 2. AdaGrad
def adagrad_online(T, d, C_radius, get_loss_and_grad):
    """AdaGrad for online learning"""
    x_t = np.zeros(d)
    cumulative_loss = []
    cumulative_regret = []
    
    sum_sq_grad = np.ones(d) * 1e-8  # 수치 안정성
    
    opt_loss = 0
    
    for t in range(1, T+1):
        loss_t, grad_t = get_loss_and_grad(x_t, t)
        
        # AdaGrad 업데이트
        sum_sq_grad += grad_t ** 2
        eta = 0.1 / np.sqrt(sum_sq_grad)
        x_new = x_t - eta * grad_t
        
        # 사영
        norm_x = np.linalg.norm(x_new)
        if norm_x > C_radius:
            x_new = x_new / norm_x * C_radius
        
        x_t = x_new
        
        cumulative_loss.append(loss_t)
        
        opt_loss_t, _ = get_loss_and_grad(x_opt, t)
        opt_loss += opt_loss_t
        
        total_loss = np.sum(cumulative_loss)
        regret = total_loss - opt_loss
        cumulative_regret.append(regret)
    
    return np.array(cumulative_loss), np.array(cumulative_regret), x_t

# 3. 손실 함수 정의: f_t(x) = a_t^T x (선형, 시간 변화)
def get_linear_loss_and_grad(x, t, d, seed_offset=0):
    """선형 손실 함수: f_t(x) = a_t^T x"""
    np.random.seed(seed_offset + t)
    a_t = np.random.randn(d)
    a_t = a_t / np.linalg.norm(a_t)  # 정규화
    
    loss = np.dot(a_t, x)
    grad = a_t
    return loss, grad

# 4. 손실 함수: f_t(x) = (1/2)‖x - c_t‖² (이차, 시간 변화하는 센터)
def get_quadratic_loss_and_grad(x, t, d, seed_offset=0):
    """이차 손실: f_t(x) = (1/2)‖x - c_t‖²"""
    np.random.seed(seed_offset + t)
    c_t = np.random.randn(d)
    c_t = c_t / np.linalg.norm(c_t) * 0.5  # 반경 0.5의 구 위
    
    loss = 0.5 * np.linalg.norm(x - c_t) ** 2
    grad = x - c_t
    return loss, grad

# 실행
print("=== Online Convex Optimization ===\n")

# 선형 손실
print("1. Linear Loss Functions:")
loss_gd_lin, regret_gd_lin, _ = sgd_online(T, d, C_radius, 
                                           lambda x, t: get_linear_loss_and_grad(x, t, d, seed_offset=100))
loss_ada_lin, regret_ada_lin, _ = adagrad_online(T, d, C_radius,
                                                 lambda x, t: get_linear_loss_and_grad(x, t, d, seed_offset=100))

print(f"  OGD Final Regret: {regret_gd_lin[-1]:.2f}")
print(f"  AdaGrad Final Regret: {regret_ada_lin[-1]:.2f}")
print(f"  Theoretical O(√T) = {np.sqrt(T):.2f}")

# 이차 손실
print("\n2. Quadratic Loss Functions:")
loss_gd_quad, regret_gd_quad, _ = sgd_online(T, d, C_radius,
                                             lambda x, t: get_quadratic_loss_and_grad(x, t, d, seed_offset=200))
loss_ada_quad, regret_ada_quad, _ = adagrad_online(T, d, C_radius,
                                                   lambda x, t: get_quadratic_loss_and_grad(x, t, d, seed_offset=200))

print(f"  OGD Final Regret: {regret_gd_quad[-1]:.2f}")
print(f"  AdaGrad Final Regret: {regret_ada_quad[-1]:.2f}")

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. OGD vs AdaGrad - 선형 손실
ax = axes[0, 0]
ax.plot(regret_gd_lin, 'b-', label='OGD', linewidth=2)
ax.plot(regret_ada_lin, 'r-', label='AdaGrad', linewidth=2)
ax.plot(np.sqrt(np.arange(1, T+1)), 'k--', alpha=0.5, label='O(√T) reference')
ax.set_xlabel('Round t')
ax.set_ylabel('Cumulative Regret')
ax.set_title('Regret Curves - Linear Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. OGD vs AdaGrad - 이차 손실
ax = axes[0, 1]
ax.plot(regret_gd_quad, 'b-', label='OGD', linewidth=2)
ax.plot(regret_ada_quad, 'r-', label='AdaGrad', linewidth=2)
ax.plot(np.sqrt(np.arange(1, T+1)), 'k--', alpha=0.5, label='O(√T) reference')
ax.set_xlabel('Round t')
ax.set_ylabel('Cumulative Regret')
ax.set_title('Regret Curves - Quadratic Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Log-log scale (수렴 속도 확인)
ax = axes[0, 2]
t_values = np.arange(1, T+1)
ax.loglog(t_values, regret_gd_lin, 'b-', label='OGD', linewidth=2)
ax.loglog(t_values, regret_ada_lin, 'r-', label='AdaGrad', linewidth=2)
ax.loglog(t_values, t_values ** 0.5, 'k--', alpha=0.5, label='√T reference')
ax.set_xlabel('Round t (log scale)')
ax.set_ylabel('Regret (log scale)')
ax.set_title('Log-log Scale: Regret Rate')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# 4. 순간 손실 비교
ax = axes[1, 0]
window = 50
ax.plot(np.convolve(loss_gd_lin, np.ones(window)/window, mode='valid'), 
       'b-', label='OGD', linewidth=2)
ax.plot(np.convolve(loss_ada_lin, np.ones(window)/window, mode='valid'), 
       'r-', label='AdaGrad', linewidth=2)
ax.set_xlabel('Round t')
ax.set_ylabel(f'Avg Loss (window={window})')
ax.set_title('Moving Average of Instantaneous Loss - Linear')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 평균 손실 (누적 손실 / 라운드 수)
ax = axes[1, 1]
avg_loss_gd_lin = np.cumsum(loss_gd_lin) / np.arange(1, T+1)
avg_loss_ada_lin = np.cumsum(loss_ada_lin) / np.arange(1, T+1)
ax.plot(avg_loss_gd_lin, 'b-', label='OGD', linewidth=2)
ax.plot(avg_loss_ada_lin, 'r-', label='AdaGrad', linewidth=2)
ax.set_xlabel('Round t')
ax.set_ylabel('Average Loss')
ax.set_title('Average Loss Convergence - Linear')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Regret/√T 비율 (이론값과 비교)
ax = axes[1, 2]
sqrt_t = np.sqrt(np.arange(1, T+1))
ratio_gd = regret_gd_lin / sqrt_t
ratio_ada = regret_ada_lin / sqrt_t
ax.plot(ratio_gd, 'b-', label='OGD: Regret/√T', linewidth=2)
ax.plot(ratio_ada, 'r-', label='AdaGrad: Regret/√T', linewidth=2)
ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, label='Theoretical constant')
ax.set_xlabel('Round t')
ax.set_ylabel('Regret / √T')
ax.set_title('Empirical Regret Rate Constant')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('online_convex_optimization.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ Visualizations saved to online_convex_optimization.png")
print(f"✓ O(√T) regret achieved: {regret_gd_lin[-1] / np.sqrt(T):.2f} * √T")
```

## 🔗 AI/ML 연결

1. **강화 학습**: Regret 최소화 = 최적 정책 학습
2. **배치 학습 → 온라인**: SGD는 OCO의 특수한 경우
3. **적응 알고리즘**: AdaGrad, RMSprop, Adam은 모두 OCO 프레임워크
4. **적대적 학습**: 분포 변화, adversarial examples 처리
5. **다중팔 밴딧**: 선택지 하나씩 → 전체 손실 학습

## ⚖️ 가정과 한계

| 항목 | 가정 | 현실 |
|------|------|------|
| 손실 공개 | f_t(x_t) 관측 후 학습 | 때로 전체 함수 필요 |
| 볼록성 | 모든 f_t 볼록 | 신경망은 비볼록 |
| 제약 집합 | C 고정 | 때로 변함 |
| 그레디언트 상한 | ‖g_t‖ ≤ G | 때로 상한 무한 |

## 📌 핵심 정리

1. **Regret 정의**: 온라인 알고리즘 vs 최적 고정 해의 차이
2. **O(√T) 경계**: OGD, AdaGrad 모두 달성 가능
3. **학습률 감소**: ηₜ = 1/(G√T) 최적
4. **적응 학습률**: AdaGrad로 희소 그레디언트 처리
5. **망원급**: Regret 증명의 핵심 기법

## 🤔 생각해볼 문제

**문제 1:** O(√T)보다 더 빠른 수렴이 가능한가? 어떤 조건에서?

<details>
<summary>힌트 및 해설</summary>

**O(√T)는 최악의 경우 (worst-case) 경계다.**

더 빠른 수렴이 가능한 경우:
1. **Strongly convex 손실**: O(log T) 가능
   - ∇²fₜ ≽ μI for all t
   - 직관: 가파른 곡률 → 빠른 수렴

2. **exp-concave 손실**: O(log T) 가능
   - 매우 강한 곡률 조건

3. **손실 의존**: T개 손실의 패턴이 특별한 경우
   - 예: 주기적 손실

**실제:** 머신러닝은 보통 강볼록 아님 → O(√T) 주요.

</details>

**문제 2:** 온라인 학습과 배치 학습의 관계는?

<details>
<summary>힌트 및 해설</summary>

**배치 설정:** 모든 fₜ = f (동일)

이 경우 OCO:
$$\text{Regret}(T) = T · \min_x f(x) - \sum_{t=1}^T f(x_t)$$

만약 알고리즘이 한 점 x*에 수렴하면:
$$\text{Regret}(T) \approx T · (f(x*) - f(x^*_{\text{opt}}))$$

따라서 모든 라운드에서 같은 x를 선택하면 regret = 0 (최적).

**결론:** 온라인 학습이 더 일반적 (시간 변화하는 손실 처리).

배치는 온라인의 특수한 경우 (모든 fₜ = f).

</details>

**問題 3:** Regret 경계가 T에 의존하지 않는 알고리즘이 있는가?

<details>
<summary>힌트 및 해설</summary>

**불가능** (정보 이론적 하한)

**증명 스케치:**
- T 라운드에서 알고리즘은 T개의 선택 x₁,...,xₜ 결정
- 적대자(adversary)는 이들에 대응하는 손실 f₁,...,fₜ 선택
- T가 크면 누적 손실도 크면: Regret(T) ≥ Ω(T)

**best case (O(√T) 달성):**
- T → ∞일 때 평균 regret → 0
- 하지만 누적값은 무한히 증가

**의미:** 온라인 학습은 "장기 평균 성능"만 보장, 초기는 나쁠 수 있음.

</details>

<div align="center">

| | |
|---|---|
| [◀ 04. 딥러닝은 왜 비볼록인데 동작하는가](./04-deep-learning-non-convex.md) | [📚 README](../README.md) |

</div>
