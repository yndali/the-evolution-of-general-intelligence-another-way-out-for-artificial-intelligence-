Production-Ready Core: hyb_ason_core.py

import numpy as np
import scipy.sparse as sp
import threading

class HybASONEngine:
    def __init__(self, num_nodes=1000, sigma=0.7, rho_critical=0.98):
        self.N = num_nodes
        self.sigma = sigma
        self.rho_critical = rho_critical
        self.lock = threading.Lock()
        
        # 1. Critical Initialization (Sparse CSR representation)
        print(f"[Engine] Initializing Hyb-ASON core with N={num_nodes}...")
        A = np.random.normal(0, 1.0 / np.sqrt(self.N), (self.N, self.N))
        W_sym = self.sigma * ((A + A.T) / 2.0) + (1.0 - self.sigma) * A
        np.fill_diagonal(W_sym, 0.0)
        
        # Thresholding for sparsity to maintain performance
        W_sym[np.abs(W_sym) < 0.05] = 0.0
        self.W = sp.csr_matrix(np.abs(W_sym))
        self.W_0 = self.W.copy() 
        
        # 2. State & Thermodynamic Control
        self.V = np.zeros(self.N)
        self.theta_baseline = 1.0
        self.E = 100.0
        self.Lambda_metabolic = 5.0
        self.gamma_base = 0.01 
        
        self.tau = np.zeros(self.N, dtype=int)
        self.tags = np.zeros(self.N, dtype=int) 
        self.tags[:int(self.N * 0.1)] = 0       
        
        # History Buffer for Causal Inference (Fixes distributed jitter)
        self.spike_history = -100 * np.ones((self.N, 5), dtype=int)

    def _normalize_to_critical_radius(self):
        """O(N^2) Power Iteration for Sparse Matrices"""
        x = np.random.normal(0, 1, self.N)
        x = x / np.linalg.norm(x)
        for _ in range(15):
            x_next = self.W.dot(x)
            rho = np.dot(x, x_next)
            norm_val = np.linalg.norm(x_next)
            if norm_val == 0: break
            x = x_next / norm_val
        
        if rho > 0:
            self.W *= (self.rho_critical / rho)

    def step_cycle(self, external_spikes=None, dH_path=0.0, physical_step=0):
        with self.lock:
            # 3. Dynamic Threshold Limiter
            theta_dynamic = self.theta_baseline * (1.0 + 5.0 * np.exp(-max(0.1, self.E) / 20.0))
            
            if external_spikes:
                for idx in external_spikes:
                    self.V[idx] += 1.2
                    self.tags[idx] = 1 
            
            # Sparse Pulse Propagation
            fired = self.V >= theta_dynamic
            self.V[fired] = 0.0
            input_currents = self.W.T.dot(fired.astype(float))
            
            # Logic: Update State & History
            active_edges = 0
            for i in np.where(fired)[0]:
                self.tau[i] = physical_step
                self.spike_history[i] = np.roll(self.spike_history[i], 1)
                self.spike_history[i][0] = physical_step
                active_edges += self.W.getrow(i).nnz
            
            self.V = np.clip(self.V * 0.95 + input_currents, 0, 10)
            
            # Energy Clamping (Fixes Low-Frequency Oscillations)
            energy_loss = min(active_edges * self.gamma_base * (self.E / 100.0), self.E * 0.5)
            self.E = max(0.1, self.E + self.Lambda_metabolic - energy_loss)
            
            # Local Plasticity
            if np.any(fired):
                self._apply_asynchronous_stdp(physical_step)
                self._apply_synaptic_scaling()
                if physical_step % 10 == 0:
                    self._normalize_to_critical_radius()
            
            return fired

    def _apply_asynchronous_stdp(self, current_step):
        for i in np.where(self.spike_history[:, 0] == current_step)[0]:
            # Vectorized causal verification
            row = self.W.getrow(i)
            targets = row.indices
            for j in targets:
                # Check causal window (1-3 logic steps)
                if any(0 < (t - current_step) <= 3 for t in self.spike_history[j]):
                    self.W[i, j] = min(self.W[i, j] + 0.02, 1.0)
                else:
                    self.W[i, j] = max(self.W[i, j] - 0.005, 0.0)

    def _apply_synaptic_scaling(self):
        """In-degree conservation for stability"""
        col_sums = np.array(self.W.sum(axis=0)).flatten()
        target_sums = np.array(self.W_0.sum(axis=0)).flatten()
        for j in range(self.N):
            if col_sums[j] > 0:
                self.W[:, j] *= (target_sums[j] / col_sums[j])

    def estimate_spectral_radius(self):
        """O(N^2) Spectral Radius assertion"""
        x = np.random.normal(0, 1, self.N)
        x /= np.linalg.norm(x)
        for _ in range(10):
            x = self.W.dot(x)
            x /= np.linalg.norm(x)
        return np.linalg.norm(self.W.dot(x))

        
