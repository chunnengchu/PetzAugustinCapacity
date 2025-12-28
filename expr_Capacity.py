import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import qutip
import random
import time

# Fix random seeds
np.random.seed(1)
random.seed(1)
qutip.settings.rand_seed = 1


# Helper functions


def generate_psd(D):
    qobj_density_matrix = qutip.rand_dm(D)
    return qobj_density_matrix.full()

def ThompSonMetric(U,V):
    vals1 = linalg.eigh(U,V, eigvals_only=True)
    dT = np.log(max(np.max(vals1), 1/np.min(vals1)))
    return dT




def FGM(Wj_power, alpha, eps = 1e-9, T = 1000):
    # Universal Fast Gradient Method
    # Nesterov, Y. ``Universal gradient methods for convex optimization problems'' (2015)
    print("\n--- FGM for alpha =", alpha, "eps =", eps, "---")

    def Grad_tilde_g_alpha_R(Wj_power, p, alpha): # Gradient of the function \tilde{g}_{\alpha}^{\text{R}} in our paper.
        inner_matrix = p[0]*Wj_power[0]
        for j in range(1, len(Wj_power)):
            inner_matrix = inner_matrix + p[j] * Wj_power[j]
        inner_matrix = linalg.fractional_matrix_power(inner_matrix, 1 / alpha - 1)
        ans = np.zeros(len(p))
        for j in range(len(p)):
            ans[j] = np.real(np.tensordot(inner_matrix.T , Wj_power[j]))
        return ans / alpha, np.dot(ans, p)

    def tilde_g_alpha_R(Wj_power, p, alpha): # Function \tilde{g}_{\alpha}^{\text{R}} in our paper.
        inner = p[0] * Wj_power[0]
        for j in range(1, len(Wj_power)):
            inner += p[j] * Wj_power[j]
        inner = linalg.fractional_matrix_power(inner, 1 / alpha)
        return np.trace(inner)
    
    def RenyiInformation(Wj_power, p, alpha): # 
        return (alpha/(alpha-1)) * np.log(tilde_g_alpha_R(Wj_power, p, alpha))

    # [cite: 374] Initialization
    n = len(Wj_power)
    p1 = np.ones(n) / n
    pt = np.ones(n) / n
    tilde_p1 = np.ones(n) / n
    tilde_pt = np.ones(n) / n
    qt = np.ones(n) / n
    At = 0
    Lt = 1.0
    acc_grad = np.zeros(n) # Accumulated gradient (For \phi_t(p))
    
    time_list = [0.0]
    RenyiInformations = [np.real(RenyiInformation(Wj_power, pt, alpha))]
    
    start_time = time.time()
    for t in range(1, T + 1):
        # Step 1: v is already computed from the previous iteration [cite: 377]
        
        # Step 2: Line search for L_k [cite: 378, 382]
        i_t = 0
        while True:
            L_curr = (2 ** i_t) * Lt
            at_next = (1 + np.sqrt(1 + 4 * L_curr * At)) / (2 * L_curr)
            At_next = At + at_next
            tau_t_next = at_next / At_next # 
            tilde_pt_next = tau_t_next * qt + (1 - tau_t_next) * pt
            
            # Gradient calculation 
            grad_next, g_tilde_pt_next = Grad_tilde_g_alpha_R(Wj_power, tilde_pt_next, alpha)
            
            hat_pt_next = np.multiply(p1 , np.exp(-at_next * grad_next - acc_grad - np.max(-at_next * grad_next - acc_grad)))
            hat_pt_next /= np.sum(hat_pt_next)
            pt_next = tau_t_next * hat_pt_next + (1 - tau_t_next) * pt
            g_pt_next = tilde_g_alpha_R(Wj_power, pt_next, alpha)
            dist_term = (L_curr / 2.0) * (np.sum(np.abs(pt_next - tilde_pt_next))**2)
            # Note: Paper uses epsilon/2 * tau 
            if g_pt_next <= g_tilde_pt_next + np.dot(grad_next, pt_next - tilde_pt_next) + dist_term + (eps * tau_t_next / 2.0):
                break
            i_t += 1
        At = At_next
        Lt = L_curr / 2.0  # Optional 'restore' strategy mentioned in paper [cite: 383]
        pt = pt_next
        at = at_next
        # Update the estimating point v_k using the total gradient (Dual Averaging) 
        acc_grad += at * grad_next
        qt = np.multiply(p1 , np.exp(-acc_grad - np.max(-acc_grad)))
        qt /= np.sum(qt)
        time_list.append(time.time() - start_time + time_list[-1])
        RenyiInformations.append(np.real(RenyiInformation(Wj_power, pt, alpha)))
        print(f"iter {t}, Renyi Information = {RenyiInformations[-1]}")
        start_time = time.time()

    return RenyiInformations, time_list


def BlahutArimoto(Wj_power, alpha, eps = 1e-9, T = 1000):
    print("\n--- Blahut-Arimoto for alpha =", alpha, "---")
    n = len(Wj_power)
    d = Wj_power[0].shape[0]

    def SimpleIteration(Q_power, Wj_power, p, alpha): 
        # Compute the next iterate Q in the inner loop (Augustin information/mean computation) of BA algorithm
        ans = np.zeros((d, d))
        for j in range(n): 
            ans = ans + p[j] * Wj_power[j] / np.tensordot(Wj_power[j].T, Q_power) # O(D^2)
        ans = linalg.fractional_matrix_power(ans, (1-alpha) / alpha) #O(D^3)
        return ans
    
    def Grad_g_alpha_A(Q_power, Wj_power, p, alpha, eps = 1e-9): # Gradient of the function g_{\alpha}^{\text{A}} in our paper.
        while True:
            Q_power_prev = Q_power.copy()
            Q_power = SimpleIteration(Q_power, Wj_power, p, alpha)   
            # Check inner convergence
            if(ThompSonMetric(Q_power, Q_power_prev) < eps): # make sure inner loop is accurate enough
                break
        ans = np.zeros(len(p))
        for j in range(n):
            ans[j] = (1 / (1 - alpha)) * np.log(np.tensordot(Wj_power[j].T, Q_power))
        return ans, Q_power

    def AugustinInformation(Q_power, Wj_power, p, alpha, eps = 1e-9):
        grad, _ = Grad_g_alpha_A(Q_power, Wj_power, p, alpha, eps)
        return np.dot(p, -grad)
        
    
    EMD_STEP_SIZE = 1
    pt = np.ones(n) / n
    Q = np.identity(d) / d
    Q_power = Q ** (1 - alpha)
    AugustinInformations = [AugustinInformation(Q_power, Wj_power, pt, alpha, eps)] 
    time_list = [0.0]
    start_time = time.time()
    for t in range(1, T + 1): # Outer iterations for updating w
        
        grad, Q_power = Grad_g_alpha_A(Q_power, Wj_power, pt, alpha, eps)
        # Exponentiated gradient step
        pt = np.multiply(pt , np.exp(-EMD_STEP_SIZE * grad))
        pt /= np.sum(pt)
        # Compute Augustin information
        time_list.append(time.time() - start_time + time_list[-1]) # exclude the time for function value evalution
        AugustinInformations.append(np.real(AugustinInformation(Q_power, Wj_power, pt, alpha, eps)))
        print(f"iter {t}, Augustin Information = {AugustinInformations[-1]}")
        start_time = time.time()
    return AugustinInformations, time_list


# --- Plotting Function ---
def plot_comparison(fgm_balanced_informations, fgm_balanced_time,
                    fgm_small_informations, fgm_small_time,
                    ba_informations, ba_time,
                    alpha, d, n):
    """Draws a figure comparing the convergence of the two algorithms."""
    
    # Ensure errors are complex-free for plotting
    fgm_balanced_informations = np.real(fgm_balanced_informations)
    fgm_small_informations = np.real(fgm_small_informations)
    ba_informations = np.real(ba_informations)
    
    
    # Find the minimum (i.e., best/lowest) function value achieved by either algorithm
    max_information = np.max(fgm_small_informations)

    # Truncate the length of fgm_small_ to match others
    fgm_small_informations = fgm_small_informations[:len(fgm_balanced_informations)]
    fgm_small_time = fgm_small_time[:len(fgm_balanced_informations)]
    
    # Compute the approximated optimization error relative to the minimum
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-15 
    fgm_balanced_error = np.maximum(max_information-fgm_balanced_informations,epsilon)
    fgm_small_error = np.maximum(max_information-fgm_small_informations,epsilon)
    ba_error = np.maximum(max_information-ba_informations, epsilon)

    fgm_balanced_iters = np.arange(len(fgm_balanced_error))
    fgm_small_iters = np.arange(len(fgm_small_error))
    ba_iters = np.arange(len(ba_error))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot elapse time v.s. error

    # Plot fgm Balanced Method
    axs[1].plot(
        fgm_balanced_time, 
        fgm_balanced_error, 
        label='FGM--Balanced', 
        linestyle='-', 
        marker='o', 
        color='purple'
    )
    # Plot fgm small Method
    axs[1].plot(
        fgm_small_time, 
        fgm_small_error, 
        label='FGM--1e-9', 
        linestyle='-', 
        marker='>', 
        color='green'
    )
    # Plot BA Method
    axs[1].plot(
        ba_time, 
        ba_error, 
        label='Blahut-Arimoto', 
        linestyle='--', 
        marker='x', 
        color='red'
    )
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('Elapsed time (sec)', fontsize=16)
    axs[1].set_ylabel('Approx. optimization error', fontsize=16)
    # axs[1].set_title(f'α = {alpha}', fontsize=14)
    axs[1].set_xlim(-1e-3,max(fgm_small_time[-1], ba_time[-1], fgm_balanced_time[-1]))
    axs[1].set_ylim(1e-8, 1e-1)
    axs[1].grid(True)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(fontsize=14, loc='best')


    # Plot iter v.s. error
    # Plot Rodomanov Method
    # Plot fgm Balanced Method
    axs[0].plot(
        fgm_balanced_iters, 
        fgm_balanced_error, 
        label='FGM--Balanced', 
        linestyle='-', 
        marker='o', 
        color='purple'
    )
    # Plot fgm small Method
    axs[0].plot(
        fgm_small_iters, 
        fgm_small_error, 
        label='FGM--1e-9', 
        linestyle='-', 
        marker='>', 
        color='green'
    )
    # Plot BA Method
    axs[0].plot(
        ba_iters, 
        ba_error, 
        label='Blahut-Arimoto', 
        linestyle='--', 
        marker='x', 
        color='red'
    )
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Number of iterations', fontsize=16)
    axs[0].set_ylabel('Approx. optimization error', fontsize=16)
    # axs[0].set_title(f'α = {alpha}', fontsize=14)
    axs[0].set_xlim(0, len(fgm_balanced_error))
    axs[0].set_ylim(1e-8, 1e-1)
    axs[0].grid(True)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(fontsize=14, loc='best')

    print("Saving figure_",alpha,d,n)
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.12,
        top=0.95,
        wspace=0.35
    )
    plt.savefig(f'figure_{alpha}_{d}_{n}.png')
    # plt.show()


# Experiment parameters
d = 2 ** 5 # Quantum state dimension
n = 2 ** 7 # Number of quantum states (Cardinality of input alphabet)
alphas = [0.6, 0.9]
T = 1000

# Generate quantum states
Wj = []
for j in range(n):
    Wj.append(generate_psd(d))
    Wj[-1] /= np.trace(Wj[-1]) # Normalize to make it a density matrix


for alpha in alphas:
    eps_balanced = ((np.log(n))**(1 / (2*alpha))) * (T ** (1 - 1.5 / alpha)) 
    # Prepare quantum states raised to power alpha
    Wj_power = []
    for j in range(n):
        Wj_power.append(linalg.fractional_matrix_power(Wj[j], alpha))
    

    # Run FGM with balanced epsilon
    fgm_balanced_informations, fgm_balanced_time = FGM(Wj_power, alpha, eps = eps_balanced, T = T)

    # Run FGM with small epsilon
    fgm_small_informations, fgm_small_time = FGM(Wj_power, alpha, eps = 1e-9, T = T * 3) # Run longer for small epsilon to compute the approximate optimum
    
    # Run Blahut-Arimoto type algorithm
    ba_informations, ba_time = BlahutArimoto(Wj_power, alpha, eps = 1e-9, T = T)

    
    # --- Plotting Call ---

    plot_comparison(fgm_balanced_informations, fgm_balanced_time,
                    fgm_small_informations, fgm_small_time, 
                    ba_informations, ba_time,
                    alpha, d, n)
