# 5. 쌍대 해석 — 그림자 가격

## 🎯 핵심 질문
- 쌍대 변수가 실제로 무엇을 의미하는가?
- 그림자 가격(shadow price)이란 무엇이고, 어떻게 계산하는가?
- 상보 느슨함이 자원 배분의 의미에서 무엇을 말하는가?

## 🔍 왜 이 이론이 AI에서 중요한가
쌍대 변수는 제약의 중요도를 정량화합니다. 예를 들어, SVM에서 쌍대 변수는 각 표본의 가중치이고, 자원 할당 문제에서는 각 자원의 가치입니다. 이를 해석하면 모델의 의사결정 과정을 이해하고 개선할 수 있습니다.

## 📐 수학적 선행 조건
- Lagrangian과 쌍대 함수 (Chapter 4-1)
- 강쌍대성과 KKT 조건 (Chapters 4-3, 4-4)
- 민감도 분석의 개념

## 📖 직관적 이해
쌍대 변수는 "제약을 조금 완화하면 목적함수가 얼마나 개선되는가"를 나타냅니다. 예를 들어:
- 공장의 생산량 제약을 1시간 더 풀면, 이윤이 λ만큼 증가
- 포트폴리오의 예산 제약을 1달러 더 풀면, 기대 수익이 λ만큼 증가
- SVM의 데이터 포인트의 쌍대 변수가 크면, 그 점이 분류기 결정에 중요

## ✏️ 엄밀한 정의

**파라미터화된 원 문제:**
```
p(u, v) = minimize   f₀(x)
          subject to fᵢ(x) ≤ uᵢ,  i = 1,...,m
                    hⱼ(x) = vⱼ,   j = 1,...,p
```

여기서 $u \in \mathbb{R}^m, v \in \mathbb{R}^p$는 제약의 RHS입니다. 원래 문제는 $u = 0, v = 0$입니다.

**그림자 가격 정리:**

강쌍대성이 성립하고 최적 쌍대 변수 $\lambda^*, \nu^*$가 유일하면,
$$\frac{\partial p(u, v)}{\partial u_i}\bigg|_{u=0,v=0} = -\lambda_i^*$$
$$\frac{\partial p(u, v)}{\partial v_j}\bigg|_{u=0,v=0} = -\nu_j^*$$

**해석:**
- $\lambda_i^* > 0$: 제약 $i$의 RHS를 1단위 증가(완화) → 최적값이 $-\lambda_i^*$만큼 개선
- $\lambda_i^* = 0$: 제약 $i$는 비활성이므로, 조금 완화해도 영향 없음

## 🔬 정리와 증명

**정리 1: 그림자 가격 정리**

*가정:*
- 원 문제가 볼록이고 Slater 조건 만족
- 최적 쌍대 변수 $(\lambda^*, \nu^*)$가 유일
- 함수들이 미분가능

*증명 스케치:*

Lagrangian:
$$L(x, u, v, \lambda, \nu) = f_0(x) + \sum_i \lambda_i (f_i(x) - u_i) + \sum_j \nu_j (h_j(x) - v_j)$$

강쌍대성에서:
$$p(u,v) = \max_{\lambda \geq 0, \nu} \min_x L(x, u, v, \lambda, \nu)$$

포락선 정리(Envelope Theorem):
$$\nabla_u p(u,v) = -\lambda^*(u,v)$$
$$\nabla_v p(u,v) = -\nu^*(u,v)$$

$u = 0, v = 0$일 때 ($\lambda^* = \lambda^*(0,0)$, $\nu^* = \nu^*(0,0)$):
$$\frac{\partial p}{\partial u_i}\bigg|_{u=0,v=0} = -\lambda_i^*$$

∎

**정리 2: 상보 느슨함의 경제 해석**

$\lambda_i^* f_i(x^*) = 0$은:

1. **활성 제약** ($f_i(x^*) = 0$): $\lambda_i^* > 0$ 가능
   - 제약이 최적해를 결정 (경합 자원)
   - 그림자 가격이 양수 (추가 자원의 가치 있음)

2. **비활성 제약** ($f_i(x^*) < 0$): $\lambda_i^* = 0$ 필수
   - 제약이 작동하지 않음 (여유 자원)
   - 추가 완화의 이점 없음 (가격 = 0)

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

print("=" * 70)
print("쌍대 해석: 그림자 가격")
print("=" * 70)

# 예제 1: 공장 생산 최적화
print("\n[예제 1] 공장 생산: 자원의 그림자 가격")
print("-" * 70)

# 제품 1: 이윤 3, 기계 시간 2
# 제품 2: 이윤 5, 기계 시간 4
# 제약: 기계 시간 ≤ 20시간, 물질 ≤ 10단위

# max 3x₁ + 5x₂  →  min -3x₁ - 5x₂
# s.t. 2x₁ + 4x₂ ≤ 20 (기계 시간)
#      x₁ + x₂ ≤ 10  (물질)
#      x₁, x₂ ≥ 0

x = cp.Variable(2)
objective = -3*x[0] - 5*x[1]  # 최소화 = 이윤 최대화의 음수

constraints = [
    2*x[0] + 4*x[1] <= 20,  # 기계 시간
    x[0] + x[1] <= 10,      # 물질
    x[0] >= 0,
    x[1] >= 0,
]

problem = cp.Problem(cp.Minimize(objective), constraints)
problem.solve()

x_opt = x.value
profit_opt = -problem.value

print(f"최적 생산: x₁* = {x_opt[0]:.2f}, x₂* = {x_opt[1]:.2f}")
print(f"최대 이윤: {profit_opt:.2f}")

# 쌍대 변수 (그림자 가격) 추출
lambda_machine = constraints[0].dual_value
lambda_material = constraints[1].dual_value

print(f"\n[그림자 가격]")
print(f"기계 시간의 그림자 가격: λ₁* = {lambda_machine:.4f}")
print(f"물질의 그림자 가격: λ₂* = {lambda_material:.4f}")

# 제약 상태 확인
print(f"\n[제약 상태]")
machine_usage = 2*x_opt[0] + 4*x_opt[1]
material_usage = x_opt[0] + x_opt[1]

print(f"기계 시간 사용: {machine_usage:.2f} / 20 (활성: {abs(20-machine_usage) < 0.01})")
print(f"물질 사용: {material_usage:.2f} / 10 (활성: {abs(10-material_usage) < 0.01})")

# 민감도 분석: RHS 변화에 따른 최적값 변화
print(f"\n[민감도 분석]")
print("기계 시간을 1시간 증가시키면:")
if lambda_machine > 1e-6:
    print(f"  → 이윤이 약 {lambda_machine:.4f} 증가")
    print(f"  → 기계 추가 비용 < {lambda_machine:.4f}이면 이득")
else:
    print(f"  → 이윤 변화 없음 (비활성 제약)")

print("물질을 1단위 증가시키면:")
if lambda_material > 1e-6:
    print(f"  → 이윤이 약 {lambda_material:.4f} 증가")
else:
    print(f"  → 이윤 변화 없음 (비활성 제약)")

# 예제 2: 다양한 RHS에서 민감도 분석
print("\n" + "=" * 70)
print("[예제 2] 매개변수 민감도 분석")
print("-" * 70)

def solve_factory(machine_hours, material_amount):
    """주어진 자원 제약으로 공장 문제 풀이"""
    x = cp.Variable(2)
    objective = -3*x[0] - 5*x[1]
    constraints = [
        2*x[0] + 4*x[1] <= machine_hours,
        x[0] + x[1] <= material_amount,
        x[0] >= 0, x[1] >= 0,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    return -problem.value

# 기계 시간 변화 (물질 = 10 고정)
print("\n기계 시간 변화 (물질 = 10 고정):")
machine_values = [15, 18, 20, 22, 25]
profits_machine = [solve_factory(m, 10) for m in machine_values]

for m, p in zip(machine_values, profits_machine):
    change = p - profit_opt if m > 20 else profit_opt - p
    print(f"  기계 = {m:2d}: 이윤 = {p:.2f}", end="")
    if m > 20:
        print(f" (+ {change:.2f} vs m=20)")
    else:
        print(f" (- {change:.2f} vs m=20)")

# 예제 3: 포트폴리오 최적화
print("\n" + "=" * 70)
print("[예제 3] 포트폴리오: 예산 제약의 그림자 가격")
print("-" * 70)

# 자산: 수익률 [0.05, 0.08, 0.10], 공분산 행렬
returns = np.array([0.05, 0.08, 0.10])
cov = np.array([
    [0.01, 0.002, 0.001],
    [0.002, 0.02, 0.005],
    [0.001, 0.005, 0.03],
])

# 포트폴리오 문제: max rᵀx - 0.5·xᵀΣx s.t. 1ᵀx = B, x ≥ 0
# 위험 회피 계수를 lambda = 0.5로 설정

def portfolio_optimization(budget):
    x = cp.Variable(3)
    risk = cp.quad_form(x, cov)
    return_val = returns @ x
    
    objective = return_val - 0.5 * risk
    constraints = [
        cp.sum(x) == budget,
        x >= 0,
    ]
    
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    
    return problem.value, x.value

# 다양한 예산으로 풀이
budgets = [1, 2, 3, 4, 5]
results = [portfolio_optimization(b) for b in budgets]
objectives = [r[0] for r in results]
allocations = [r[1] for r in results]

print(f"\n예산 변화에 따른 최적 목적함수 값:")
for b, obj in zip(budgets, objectives):
    print(f"  예산 = {b}: 목적함수 = {obj:.6f}")

# 수치 미분으로 그림자 가격 계산
shadow_price_budget = (objectives[-1] - objectives[0]) / (budgets[-1] - budgets[0])
print(f"\n예산 제약의 그림자 가격 (수치): {shadow_price_budget:.6f}")
print("→ 예산을 1단위 증가시키면 목적함수가 약 {:.6f} 증가".format(shadow_price_budget))

# 예제 4: SVM의 쌍대 변수 해석
print("\n" + "=" * 70)
print("[예제 4] SVM: 쌍대 변수 = 표본 가중치")
print("-" * 70)

# 간단한 2D 분류 문제
np.random.seed(42)
n_pos = 20
n_neg = 20

X_pos = np.random.randn(n_pos, 2) + np.array([2, 2])
X_neg = np.random.randn(n_neg, 2) + np.array([-2, -2])

X = np.vstack([X_pos, X_neg])
y = np.array([1]*n_pos + [-1]*n_neg)

# Hard-margin SVM (쌍대 형식)
# max Σαᵢ - 0.5·ΣᵢΣⱼ αᵢαⱼyᵢyⱼKᵢⱼ  s.t. Σαᵢyᵢ = 0, 0 ≤ αᵢ ≤ C

K = X @ X.T  # 선형 커널

alpha = cp.Variable(n_pos + n_neg)
objective = cp.sum(alpha) - 0.5 * cp.quad_form(alpha, np.diag(y) @ K @ np.diag(y))

constraints = [
    alpha >= 0,
    alpha <= 1.0,  # C = 1 (soft margin)
    y @ alpha == 0,
]

problem = cp.Problem(cp.Maximize(objective), constraints)
problem.solve()

alpha_opt = alpha.value

# Support vectors: αᵢ > 10^-5
sv_indices = np.where(alpha_opt > 1e-5)[0]
print(f"Support vectors: {len(sv_indices)} / {n_pos+n_neg}")

# 쌍대 변수의 의미
print(f"\n[쌍대 변수 분석]")
print(f"αᵢ > 0인 표본: Support vectors")
print(f"  - 이들이 분류기 결정에 중요한 역할")
print(f"  - αᵢ가 클수록 영향 큼")

print(f"\nαᵢ = 0인 표본: Non-support vectors")
print(f"  - 삭제해도 최적 분류기 변하지 않음")
print(f"  - 계산 효율성: support vectors만 저장")

top_k = 5
top_indices = np.argsort(alpha_opt)[-top_k:]
print(f"\n상위 {top_k}개 Support Vector의 쌍대 변수:")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. 표본 {idx}: α = {alpha_opt[idx]:.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 왼쪽: 공장 문제의 sensitivity
ax1 = axes[0]
machine_range = np.linspace(10, 30, 50)
profits_range = [solve_factory(m, 10) for m in machine_range]

ax1.plot(machine_range, profits_range, 'b-', linewidth=2, label='최적 이윤')
ax1.axvline(x=20, color='r', linestyle='--', alpha=0.7, label='현재 기계 시간')
ax1.axhline(y=profit_opt, color='r', linestyle='--', alpha=0.7)

# 그림자 가격으로 근사
if lambda_machine > 1e-6:
    linear_approx = profit_opt + lambda_machine * (machine_range - 20)
    ax1.plot(machine_range, linear_approx, 'r--', linewidth=1.5, alpha=0.7, 
            label=f'선형 근사 (shadow price = {lambda_machine:.3f})')

ax1.set_xlabel('기계 시간 (시간)')
ax1.set_ylabel('최적 이윤')
ax1.set_title('민감도 분석: 그림자 가격의 해석')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 오른쪽: SVM의 쌍대 변수
ax2 = axes[1]

# 표본 시각화
ax2.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', marker='o', s=100, alpha=0.6, label='클래스 +1')
ax2.scatter(X_neg[:, 0], X_neg[:, 1], c='red', marker='x', s=100, alpha=0.6, label='클래스 -1')

# Support vectors 강조
sv_pos = sv_indices[sv_indices < n_pos]
sv_neg = sv_indices[sv_indices >= n_pos]

ax2.scatter(X[sv_pos, 0], X[sv_pos, 1], c='blue', s=400, linewidths=2, 
           facecolors='none', label='SV (클래스 +1)')
ax2.scatter(X[sv_neg, 0], X[sv_neg, 1], c='red', s=400, linewidths=2, 
           facecolors='none', label='SV (클래스 -1)')

ax2.set_xlabel('특성 1')
ax2.set_ylabel('특성 2')
ax2.set_title('SVM: Support Vectors (α > 0)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('/tmp/dual_interpretation.png', dpi=150, bbox_inches='tight')
print("\n✓ 그림자 가격 시각화 저장됨")

plt.show()
```

## 🔗 AI/ML 연결
- **특성 선택**: 각 특성의 쌍대 변수 = 그 특성의 중요도 (feature importance)
- **정규화 계수 튜닝**: λ(정규화)의 그림자 가격 = 모델 복잡도의 비용
- **리소스 할당**: 클라우드 컴퓨팅에서 각 작업의 쌍대 변수 = 자원의 한계 가격

## ⚖️ 가정과 한계
- 그림자 가격은 **국소적** 해석 (작은 RHS 변화에 대해서만 유효)
- 큰 RHS 변화에서는 최적 기저가 바뀌므로, 선형 근사 실패
- 최적해가 유일하지 않으면 쌍대 변수가 유일하지 않을 수 있음

## 📌 핵심 정리
1. **그림자 가격**: 쌍대 변수 = 제약 RHS 변화의 한계 효과
2. **활성 제약**: λ > 0, 그림자 가격 존재
3. **비활성 제약**: λ = 0, 그림자 가격 = 0
4. **경제 해석**: 자원 배분, 의사결정 지원
5. **ML 해석**: 특성 가중치, support vector, 정규화 계수

## 🤔 생각해볼 문제

**문제 1:** 공장 생산 문제에서 기계 시간의 그림자 가격이 물질의 그림자 가격보다 큰 이유를 경제학적으로 설명하세요.

<details>
<summary>힌트 및 해설</summary>

그림자 가격이 클수록 그 자원이 더 많이 제약합니다.

기계 시간의 그림자 가격이 크다는 것은:
- 기계가 완전히 활용 중 (binding constraint)
- 기계 시간을 1시간 더 늘리면 이윤이 크게 증가
- 기계 구매/임차의 우선순위가 높음

반대로 물질의 그림자 가격이 작다면:
- 물질이 남아 있음 (slack)
- 추가 물질은 이윤 개선에 도움 안 됨
- 현재의 병목은 물질이 아닌 다른 자원

</details>

**문제 2:** 포트폴리오 최적화에서 예산 제약의 그림자 가격을 0.02라고 하면, 이것이 무엇을 의미하는지 설명하세요.

<details>
<summary>힌트 및 해설</summary>

그림자 가격 = 0.02는:

"예산을 1단위 증가시키면 (또는 1달러 더 투자하면) 목적함수가 약 0.02 개선된다"는 의미입니다.

목적함수가 (기대 수익 - 0.5·위험) 형태이므로:

- 기대 수익이 2% 증가, 또는
- 위험이 4% 감소, 또는
- 둘의 조합

따라서 추가 자본의 이익이 연 2%라는 뜻이며, 차입 비용이 2% 이하라면 더 빌려서 투자할 가치가 있습니다.

</details>

**문제 3:** SVM에서 support vector의 쌍대 변수 αᵢ가 크다는 것이 분류기의 성능에 어떤 의미인지 설명하세요.

<details>
<summary>힌트 및 해설</summary>

SVM의 최적 분류기:
$$f(x) = \text{sign}\left(\sum_i \alpha_i^* y_i K(x_i, x) + b\right)$$

αᵢ가 크면:
- 표본 i가 결정 경계에 가까움
- 분류기 정의에 큰 기여
- 만약 αᵢ = C (최대값)이면, 그 표본은 마진 내에 있거나 오분류된 것

경제적 해석:
- αᵢ는 표본 i의 "영향력"
- αᵢ = 0이면 삭제 가능 (redundant)
- 대부분의 표본이 αᵢ = 0인 경우가 많음 (스파세성, 계산 효율)

</details>

<div align="center">

| [◀ 04. KKT 조건](./04-kkt-conditions.md) | [📚 README](../README.md) | [06. SVM의 쌍대 유도 ▶](./06-svm-dual-derivation.md) |

</div>
