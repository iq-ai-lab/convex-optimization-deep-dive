# 06. Stochastic 방법과 분산 감소

## 🎯 핵심 질문
- SGD는 왜 $O(1/\sqrt{k})$에서 수렴이 멈추는가?
- 분산이 수렴 속도를 제한하는 방법은?
- 분산 감소 기법 (SVRG, SAG, SAGA)으로 선형 수렴을 회복할 수 있는가?

## 🔍 왜 이 이론이 AI에서 중요한가
신경망 학습은 기본적으로 대량의 데이터로 구성된 손실함수 합의 최적화입니다. SGD는 메모리 효율적이지만, 고정된 분산으로 인해 높은 정확도에 수렴하지 못합니다. SVRG, SAGA 같은 분산 감소 기법은 이를 해결하여 강볼록 문제에서 선형 수렴을 달성합니다.

## 📐 수학적 선행 조건
- **합 최적화**: $\min_x \frac{1}{n}\sum_{i=1}^n f_i(x)$
- **조건부 기댓값**: $\mathbb{E}_i[\nabla f_i(x)] = \nabla f(x)$
- **분산**: $\sigma^2 = \mathbb{E}[\|\nabla f_i(x) - \nabla f(x)\|^2]$
- **강볼록성**: 모든 $f_i$가 μ-strongly convex

## 📖 직관적 이해

**SGD의 노이즈**: 매 iteration에서 하나의 함수 $f_i$만 샘플링하므로, 그래디언트가 "흔들립니다". 이 노이즈는 수렴의 하한을 만듭니다.

**분산의 영향**: 노이즈를 제어할 수 없으면 수렴 곡선이 $O(1/\sqrt{k})$에서 평탄해집니다.

**SVRG의 아이디어**: "Reference gradient" $\tilde{g} = \nabla f(\tilde{x})$를 주기적으로 계산하고, 상대적 그래디언트로 분산을 감소시킵니다:
$$\nabla f_i(x_k) - \nabla f_i(\tilde{x}) + \tilde{g}$$

$\mathbb{E}[\nabla f_i(x_k) - \nabla f_i(\tilde{x})] = \nabla f(x_k) - \nabla f(\tilde{x})$이므로, 분산이 남은 오차에 의존합니다.

**SAG/SAGA**: 각 함수의 과거 그래디언트를 메모리에 저장하여, 매 iteration마다 변화를 추적합니다.

## ✏️ 엄밀한 정의

**정의 (Stochastic gradient descent)**:
$$x_{k+1} = x_k - \eta \nabla f_{i_k}(x_k)$$

여기서 $i_k \sim \text{Uniform}(\{1,\ldots,n\})$ uniformly sampled.

**정의 (분산)**:
$$\sigma^2(x) = \mathbb{E}_{i}\left[\|\nabla f_i(x) - \nabla f(x)\|^2\right]$$

**정의 (SVRG - Stochastic Variance Reduced Gradient)**:
$$x_{k+1} = x_k - \eta_k \left(\nabla f_i(x_k) - \nabla f_i(\tilde{x}) + \nabla f(\tilde{x})\right)$$

주기적으로 (매 $m$ iteration마다) $\tilde{x}$를 현재값으로 업데이트, $\nabla f(\tilde{x})$ 재계산.

**정의 (SAG/SAGA - Stochastic Average Gradient)**:
메모리에 각 함수의 그래디언트 저장:
$$g_i^{(k)} = \nabla f_i(x_k) \quad \text{(sample } i_k \text{에 대해)}$$

업데이트:
$$x_{k+1} = x_k - \eta \sum_{i=1}^n \frac{g_i^{(k)}}{n}$$

## 🔬 정리와 증명

**정리 1 (SGD의 $O(1/\sqrt{k})$ 수렴 및 분산 하한)**

$f_i$가 L-smooth이고, 분산 $\sigma^2(x^*)$가 있을 때:
$$\mathbb{E}[f(x_k)] - f^* \le O\left(\frac{1}{\sqrt{k}}\right) + O(\sigma^2)$$

즉, SGD는 $O(\sigma^2)$ 오차로 수렴이 멈춥니다 (steady-state error).

*증명*:

**Step 1**: Descent lemma (SGD 버전):
$$\mathbb{E}[f(x_{k+1}) | x_k] \le f(x_k) - \eta \mathbb{E}[\|\nabla f(x_k)\|^2 | x_k] + \frac{L\eta^2}{2}\mathbb{E}[\|\nabla f_i(x_k)\|^2 | x_k]$$

여기서 $i_k$ random, 따라서:
$$\mathbb{E}[\nabla f_i(x_k)] = \nabla f(x_k)$$
$$\mathbb{E}[\|\nabla f_i(x_k)\|^2] = \mathbb{E}[\|\nabla f(x_k) + (\nabla f_i - \nabla f)\|^2]$$
$$= \|\nabla f(x_k)\|^2 + \sigma^2(x_k)$$

(직교성으로 교차항이 0)

**Step 2**: Descent lemma를 정리하면:
$$\mathbb{E}[f(x_{k+1}) | x_k] \le f(x_k) - \eta(1 - \frac{L\eta}{2})\|\nabla f(x_k)\|^2 + \frac{L\eta^2}{2}\sigma^2(x_k)$$

**Step 3**: $\eta = 1/L$을 택하면:
$$\mathbb{E}[f(x_{k+1})] \le \mathbb{E}[f(x_k)] - \frac{1}{2L}\mathbb{E}[\|\nabla f(x_k)\|^2] + \frac{1}{2L}\sigma^2(x_k)$$

**Step 4**: Telescoping sum:
$$f(x_k) - f^* \le \ldots + \frac{\sigma^2}{2L}$$

첫 번째 항은 $O(1/\sqrt{k})$, 두 번째 항은 분산으로 인한 고정 오차. □

**정리 2 (SVRG의 선형 수렴 - Strongly Convex)**

$f_i$가 L-smooth이고 μ-strongly convex일 때, SVRG는:
$$\mathbb{E}[\|x_k - x^*\|^2] \le \left(1 - \frac{\mu}{5L}\right)^k \|x_0 - x^*\|^2$$

즉, 결정론적 경사하강법처럼 선형 수렴합니다!

*증명 스케치*:

**Step 1**: SVRG 스텝:
$$x_{k+1} = x_k - \eta \left(\nabla f_i(x_k) - \nabla f_i(\tilde{x}) + \nabla f(\tilde{x})\right)$$

**Step 2**: 분산 감소를 보이기 위해, 보조 그래디언트:
$$v_k = \nabla f_i(x_k) - \nabla f_i(\tilde{x}) + \nabla f(\tilde{x})$$

기댓값:
$$\mathbb{E}_i[v_k] = \nabla f(x_k) - \nabla f(\tilde{x}) + \nabla f(\tilde{x}) = \nabla f(x_k)$$

좋음! 그리고 분산:
$$\mathbb{E}_i[\|v_k - \nabla f(x_k)\|^2] = \mathbb{E}[\|\nabla f_i - \nabla f\|^2 + \text{(correlation)}]$$

L-smooth 조건으로부터:
$$\|\nabla f_i(x) - \nabla f_i(y)\| \le L\|x - y\|$$

따라서:
$$\mathbb{E}[\|v_k - \nabla f(x_k)\|^2] \le 2L^2 \|x_k - \tilde{x}\|^2$$

**Step 3**: Strongly convex에서:
$$\|\nabla f(x) - \nabla f(x^*)\| \ge \mu \|x - x^*\|$$

따라서 $\|x_k - \tilde{x}\|$가 작으면 분산도 작습니다.

**Step 4**: 각 "epoch" (m iterations)에서 $\tilde{x}$를 업데이트하면, 누적 오차가 제어되어:
$$\mathbb{E}[\|x_{k+1} - x^*\|^2] \le (1 - c\mu/L) \mathbb{E}[\|x_k - x^*\|^2]$$

일정한 인수 $c > 0$로. □

**정리 3 (SAG/SAGA의 수렴)**

SAG: $\mathbb{E}[f(x_k)] - f^* \le O((1 - \mu/(16nL))^k)$ (선형)

SAGA: $\mathbb{E}[f(x_k)] - f^* \le O((1 - \mu/(16nL))^k)$ (선형)

*증명*:

메모리에 저장된 그래디언트:
$$\bar{g}^{(k)} = \frac{1}{n}\sum_{i=1}^n g_i^{(k)}$$

SAG: $g_i^{(k)}$ 중 일부만 업데이트 (표본 추출).

SAGA: 매 iteration마다 정확한 평균을 유지.

Both achieve linear convergence with memory $O(nd)$. □

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# 테스트 함수: strongly convex quadratic 합
# f(x) = (1/n) * sum_i f_i(x) where f_i(x) = (1/2)(x - c_i)^T A_i (x - c_i)

np.random.seed(42)
n_samples = 100
n_dim = 20

# 각 함수 f_i의 Hessian을 다르게 설정 (분산 생성)
A_list = []
c_list = []

base_A = np.eye(n_dim)
for i in range(n_samples):
    # 각 샘플마다 약간 다른 곡률
    A_i = base_A + 0.1 * np.random.randn(n_dim, n_dim)
    A_i = (A_i + A_i.T) / 2  # Symmetric
    A_list.append(A_i)
    c_list.append(np.random.randn(n_dim))

# 평균 함수
def f_avg(x):
    return np.mean([0.5 * (x - c)**2 @ A @ (x - c) for A, c in zip(A_list, c_list)])

def grad_f_avg(x):
    return np.mean([A @ (x - c) for A, c in zip(A_list, c_list)])

def grad_f_i(x, i):
    return A_list[i] @ (x - c_list[i])

# 최적해 (모든 f_i가 같은 최적해를 가짐, c=0일 때)
x_opt = np.zeros(n_dim)
f_opt = 0.0

# 강볼록성과 smoothness 파라미터
mu = np.min(eigvalsh(A_list[0]))
L = np.max(eigvalsh(A_list[-1]))

print(f"μ (strong convexity): {mu:.4f}")
print(f"L (smoothness): {L:.4f}")
print(f"Condition number: {L/mu:.2f}")

# 1. SGD
def sgd(x0, num_iters=1000, eta=1/L, verbose=False):
    x = x0.copy()
    losses = []
    
    for k in range(num_iters):
        i = np.random.randint(0, n_samples)
        grad = grad_f_i(x, i)
        x = x - eta * grad
        
        loss = f_avg(x) - f_opt
        losses.append(loss)
        
        if verbose and k % 100 == 0:
            print(f"SGD k={k:4d}: f(x)={loss:.2e}")
    
    return x, np.array(losses)

# 2. SVRG (Stochastic Variance Reduced Gradient)
def svrg(x0, num_iters=1000, m=50, eta=1/(10*L), verbose=False):
    """
    SVRG: Stochastic Variance Reduced Gradient
    
    Parameters:
    - m: epoch length (iterations per reference gradient update)
    """
    x = x0.copy()
    losses = []
    
    epoch = 0
    for k in range(num_iters):
        if k % m == 0:
            # 새로운 epoch: reference gradient 계산
            x_tilde = x.copy()
            g_tilde = grad_f_avg(x_tilde)
            epoch += 1
            
            if verbose:
                print(f"SVRG Epoch {epoch}: x computed")
        
        # SVRG 스텝: variance reduced gradient
        i = np.random.randint(0, n_samples)
        grad_fi_x = grad_f_i(x, i)
        grad_fi_tilde = grad_f_i(x_tilde, i)
        
        # Variance reduced update
        grad = grad_fi_x - grad_fi_tilde + g_tilde
        
        x = x - eta * grad
        
        loss = f_avg(x) - f_opt
        losses.append(loss)
        
        if verbose and (k+1) % (m*5) == 0:
            print(f"SVRG k={k+1:4d}: f(x)={loss:.2e}")
    
    return x, np.array(losses)

# 3. SAG (Stochastic Average Gradient)
def saga(x0, num_iters=1000, eta=1/(10*L), verbose=False):
    """
    SAGA: Stochastic Average Gradient with Averaging
    """
    x = x0.copy()
    losses = []
    
    # 모든 함수의 그래디언트 메모리
    grad_memory = [grad_f_i(x, i) for i in range(n_samples)]
    
    for k in range(num_iters):
        i = np.random.randint(0, n_samples)
        
        # 평균 그래디언트
        avg_grad = np.mean(grad_memory, axis=0)
        
        # 새로운 그래디언트 계산
        grad_fi_new = grad_f_i(x, i)
        
        # SAGA 업데이트
        grad = grad_fi_new - grad_memory[i] + avg_grad
        
        x = x - eta * grad
        
        # 메모리 업데이트
        grad_memory[i] = grad_fi_new
        
        loss = f_avg(x) - f_opt
        losses.append(loss)
        
        if verbose and (k+1) % 200 == 0:
            print(f"SAGA k={k+1:4d}: f(x)={loss:.2e}")
    
    return x, np.array(losses)

# 실행
x0 = np.random.randn(n_dim)
num_iters = 1000

print("\n" + "="*60)
print("Running SGD...")
x_sgd, losses_sgd = sgd(x0, num_iters=num_iters, verbose=False)

print("Running SVRG...")
x_svrg, losses_svrg = svrg(x0, num_iters=num_iters, m=20, verbose=False)

print("Running SAGA...")
x_saga, losses_saga = saga(x0, num_iters=num_iters, verbose=False)

print("="*60)

# 이론적 곡선
k_range = np.arange(1, num_iters + 1)

# SGD: O(1/sqrt(k))
sgd_theoretical = 1.0 / np.sqrt(k_range)

# GD (결정론적): O((1-mu/L)^k)
gd_theoretical = (1 - mu/L) ** k_range

# SVRG/SAGA: O((1-mu/(5L))^k) (실제로 더 나음)
svrg_theoretical = (1 - mu/(5*L)) ** k_range

# 시각화 1: 수렴 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 로그-로그 스케일
ax = axes[0, 0]
ax.loglog(k_range, losses_sgd, 'b-', label='SGD (실제)', linewidth=2)
ax.loglog(k_range, losses_svrg, 'r-', label='SVRG (실제)', linewidth=2)
ax.loglog(k_range, losses_saga, 'g-', label='SAGA (실제)', linewidth=2)
ax.loglog(k_range, sgd_theoretical, 'b--', label='SGD O(1/√k)', alpha=0.7, linewidth=1.5)
ax.loglog(k_range, svrg_theoretical, 'r--', label='SVRG O((1-μ/L)^k)', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('f(x_k) - f* (log)', fontsize=11)
ax.set_title('Convergence Comparison (Log-Log)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# (2) 반로그 스케일
ax = axes[0, 1]
ax.semilogy(losses_sgd, 'b-', label='SGD', linewidth=2)
ax.semilogy(losses_svrg, 'r-', label='SVRG', linewidth=2)
ax.semilogy(losses_saga, 'g-', label='SAGA', linewidth=2)
ax.semilogy(sgd_theoretical, 'b--', label='Theory O(1/√k)', alpha=0.7, linewidth=1.5)
ax.semilogy(svrg_theoretical, 'r--', label='Theory O(ρ^k)', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('f(x_k) - f* (log)', fontsize=11)
ax.set_title('Convergence Comparison (Semi-log)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (3) 처음 500 iteration 상세
ax = axes[1, 0]
k_detail = 500
ax.semilogy(losses_sgd[:k_detail], 'b-o', label='SGD', linewidth=2, markersize=2, markevery=50)
ax.semilogy(losses_svrg[:k_detail], 'r-s', label='SVRG', linewidth=2, markersize=2, markevery=50)
ax.semilogy(losses_saga[:k_detail], 'g-^', label='SAGA', linewidth=2, markersize=2, markevery=50)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('f(x_k) - f* (log)', fontsize=11)
ax.set_title('First 500 Iterations Detail', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (4) 분산 비교 (추정)
ax = axes[1, 1]
window_size = 50
sgd_var = np.convolve(np.diff(np.log(losses_sgd + 1e-15))**2, 
                      np.ones(window_size)/window_size, mode='valid')
svrg_var = np.convolve(np.diff(np.log(losses_svrg + 1e-15))**2, 
                       np.ones(window_size)/window_size, mode='valid')

ax.plot(sgd_var, 'b-', label='SGD variance (estimated)', linewidth=2)
ax.plot(svrg_var, 'r-', label='SVRG variance (estimated)', linewidth=2)
ax.set_xlabel('Iteration k (windowed)', fontsize=11)
ax.set_ylabel('Gradient variance (log scale)', fontsize=11)
ax.set_title('Variance Reduction Effect', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('variance_reduction_comparison.png', dpi=150)
plt.show()

# 정량적 분석
print("\n" + "="*60)
print("Quantitative Analysis")
print("="*60)

# 각 방법이 특정 정확도에 도달하는 데 걸리는 iteration 수
target_errors = [1e-2, 1e-4, 1e-6, 1e-8]

for target in target_errors:
    sgd_iters = np.where(losses_sgd < target)[0]
    svrg_iters = np.where(losses_svrg < target)[0]
    saga_iters = np.where(losses_saga < target)[0]
    
    sgd_count = sgd_iters[0] if len(sgd_iters) > 0 else num_iters
    svrg_count = svrg_iters[0] if len(svrg_iters) > 0 else num_iters
    saga_count = saga_iters[0] if len(saga_iters) > 0 else num_iters
    
    print(f"\nTarget error: {target:.0e}")
    print(f"  SGD:  {sgd_count:4d} iterations")
    print(f"  SVRG: {svrg_count:4d} iterations")
    print(f"  SAGA: {saga_count:4d} iterations")
    
    if svrg_count > 0:
        ratio = sgd_count / svrg_count
        print(f"  Speedup (SGD/SVRG): {ratio:.1f}x")

# 시각화 2: 분산 감소 메커니즘 설명
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 첫 100 iteration에서의 그래디언트 크기 분포
ax = axes[0]
x_test = x0.copy()

sgd_grads = []
svrg_grads = []

for _ in range(100):
    i = np.random.randint(0, n_samples)
    
    # SGD gradient
    sgd_grad = grad_f_i(x_test, i)
    sgd_grads.append(np.linalg.norm(sgd_grad))
    
    # SVRG-like: variance reduced
    x_tilde = x_test  # Simplified
    g_tilde = grad_f_avg(x_tilde)
    grad_fi_x = grad_f_i(x_test, i)
    grad_fi_tilde = grad_f_i(x_tilde, i)
    svrg_grad = grad_fi_x - grad_fi_tilde + g_tilde
    svrg_grads.append(np.linalg.norm(svrg_grad))

ax.hist(sgd_grads, bins=20, alpha=0.6, label='SGD gradients', color='blue')
ax.hist(svrg_grads, bins=20, alpha=0.6, label='SVRG gradients', color='red')
ax.set_xlabel('Gradient norm', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Gradient Magnitude Distribution', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 메모리 비용
ax = axes[1]
methods = ['SGD', 'SVRG\n(m=20)', 'SAGA']
memory_costs = [1, 1 + 20, n_samples]  # Relative memory

ax.bar(methods, memory_costs, color=['blue', 'red', 'green'], alpha=0.7)
ax.set_ylabel('Memory cost (relative)', fontsize=11)
ax.set_title('Memory Requirements', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

for i, cost in enumerate(memory_costs):
    ax.text(i, cost + 0.5, f'{cost}×', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('variance_reduction_analysis.png', dpi=150)
plt.show()
```

## 🔗 AI/ML 연결

**현대 딥러닝**: Momentum, Adam 등은 모두 SGD를 기반으로 하며, 분산의 영향을 받습니다.

**Batch effects**: 큰 배치 크기는 분산을 줄이지만 메모리를 증가시킵니다. SVRG는 이 트레이드오프를 개선합니다.

**강볼록성의 가정**: 신경망은 강볼록이 아니지만, 정규화나 특정 구간에서는 부분적으로 관찰됩니다.

## ⚖️ 가정과 한계

**가정 1**: 강볼록성을 가정합니다. 신경망은 대체로 비볼록입니다.

**가정 2**: 모든 $f_i$를 동일한 정확도로 계산할 수 있다고 가정합니다. 실제 데이터는 이질적입니다.

**SVRG 한계**: Epoch마다 전체 그래디언트 계산 필요 ($O(n)$ 비용).

**메모리**: SAGA는 $O(nd)$ 메모리 필요 (대규모에서 문제).

## 📌 핵심 정리

1. **SGD 분산**: $O(\sigma^2)$ steady-state error로 인해 $O(1/\sqrt{k})$ 수렴.
2. **분산 감소**: Reference gradient 또는 메모리로 분산 제어.
3. **SVRG 선형 수렴**: Strongly convex에서 $(1 - \mu/(5L))^k$ 달성.
4. **SAG/SAGA**: 메모리 기반, 정확한 평균 유지.
5. **트레이드오프**: 정확도 vs 계산/메모리 비용.

## 🤔 생각해볼 문제

**문제 1**: SVRG의 그래디언트 $v_k = \nabla f_i(x_k) - \nabla f_i(\tilde{x}) + \nabla f(\tilde{x})$가 정확한 그래디언트를 가진다는 것을 증명하시오.

<details>
<summary>힌트 및 해설</summary>

$$\mathbb{E}_i[v_k] = \mathbb{E}_i[\nabla f_i(x_k) - \nabla f_i(\tilde{x}) + \nabla f(\tilde{x})]$$
$$= \nabla f(x_k) - \nabla f(\tilde{x}) + \nabla f(\tilde{x})$$
$$= \nabla f(x_k)$$

즉, SVRG의 그래디언트는 정확한 그래디언트의 불편 추정량입니다.

</details>

**문제 2**: SGD의 분산 $\sigma^2 = \mathbb{E}[\|\nabla f_i - \nabla f\|^2]$가 어떤 상황에서 크고 작은가?

<details>
<summary>힌트 및 해설</summary>

분산이 클 때:
- 함수들이 매우 다름 (heterogeneous data)
- Label noise가 큼
- 배치 크기가 작음

분산이 작을 때:
- 모든 $f_i$가 비슷함 (homogeneous)
- Clean 데이터
- 배치 크기가 큼

따라서 데이터 특성과 배치 크기가 중요합니다.

</details>

**문제 3**: SAGA가 메모리를 $O(nd)$ 사용하는데, 이를 줄일 수 있는 방법이 있는가?

<details>
<summary>힌트 및 해설</summary>

메모리 효율적 변형들:
1. **스파시피케이션 (Sparsification)**: 자주 업데이트되는 벡터만 저장
2. **양자화 (Quantization)**: 그래디언트를 저저 정밀도로 저장
3. **샘플링 (Sampling)**: 모든 함수의 그래디언트가 아닌 일부만 저장

이들은 수렴 속도를 약간 해치지만 메모리를 크게 줄입니다.

</details>

<div align="center">
| [◀ 05. Interior Point Method](./05-interior-point.md) | [📚 README](../README.md) | [Ch6-01. Proximal Operator ▶](../ch6-proximal/01-proximal-operator.md) |
</div>
