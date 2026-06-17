# models/ahah/core.py
import numpy as np
from typing import List, Optional

class MemristiveSynapse:
    """
    Componente físico básico. Armazena apenas o peso (condutância).
    """
    def __init__(self, initial_w: float = 0.0):
        self.w = float(initial_w)

    def apply_delta(self, delta: float, w_min: float, w_max: float) -> None:
        """Aplica variação com saturação física (Clamping)."""
        self.w = float(np.clip(self.w + delta, w_min, w_max))

class AHAHNode:
    """
    Nó AHAH Genérico.
    Não armazena hiperparâmetros. Eles são passados no update.
    """
    def __init__(self, input_dim: int, y_max: float = 1.0):
        self.input_dim = int(input_dim)
        self.y_max = float(y_max)
        
        self.synapses: List[MemristiveSynapse] = []
        for _ in range(self.input_dim):
            w_init = np.random.normal(loc=0.0, scale=0.05)
            self.synapses.append(MemristiveSynapse(w_init))

    def potential(self, x: np.ndarray) -> float:
        w = np.array([s.w for s in self.synapses], dtype=np.float32)
        y = float(np.dot(w, x))
        return float(np.clip(y, -self.y_max, self.y_max))

    def update_weights(self, active_idx: np.ndarray, 
                       hebb: float, anti_hebb: float, 
                       decay: float, w_min: float, w_max: float,
                       noise_mean: float = 0.0, noise_std: Optional[float] = None,
                       bias_start_idx: int = -1, bias_mode: str = 'C') -> None:
        """
        Atualiza pesos. 
        """
        if len(active_idx) == 0: return

        for i in active_idx:
            syn = self.synapses[i]
            
            # Lógica de Bias
            is_bias = (i >= bias_start_idx) and (bias_start_idx >= 0)
            
            if is_bias:
                if bias_mode == 'C': delta = hebb + anti_hebb
                elif bias_mode == 'B': delta = anti_hebb
                elif bias_mode == 'A': delta = -hebb - anti_hebb
                else: delta = hebb + anti_hebb
            else:
                delta = hebb + anti_hebb

            # Aplica Decay
            delta -= (decay * syn.w)
            
            # Aplica Ruído
            if noise_std is not None and noise_std > 0.0:
                delta += np.random.normal(loc=noise_mean, scale=noise_std)
                
            # Atualiza Memristor
            syn.apply_delta(delta, w_min, w_max)

class AnalogAHAHNode:
    """Nó analógico para o Encoder (SAOL)."""
    def __init__(self, beta: float, w_min: float, w_max: float):
        # Aqui o beta é intrínseco à construção do encoder
        self.beta = beta
        self.w_min = w_min
        self.w_max = w_max
        self.synapse_val = MemristiveSynapse(0.0)
        self.synapse_bias = MemristiveSynapse(0.0)

    def activate(self, value: float) -> float:
        return (self.synapse_val.w * value) + (self.synapse_bias.w * 1.0)

    def update(self, value: float, activation: float):
        d_val = -self.beta * activation * value
        d_bias = -self.beta * activation * 1.0
        self.synapse_val.apply_delta(d_val, self.w_min, self.w_max)
        self.synapse_bias.apply_delta(d_bias, self.w_min, self.w_max)

class AHAHLayer:
    """Container de nós."""
    def __init__(self, obs_dim: int, nodes_per_layer: int, y_max: float = 1.0):
        self.obs_dim = int(obs_dim)
        self.nodes_total = int(nodes_per_layer)
        self.nodes: List[AHAHNode] = [
            AHAHNode(input_dim=obs_dim, y_max=y_max)
            for _ in range(self.nodes_total)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.array([node.potential(x) for node in self.nodes], dtype=np.float32)

    def get_node(self, index: int) -> AHAHNode:
        return self.nodes[index]