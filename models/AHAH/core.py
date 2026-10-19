# models/ahah/core.py
import numpy as np
from typing import List, Optional

class MemristiveSynapse: # Código Legacy para compatibilidade
    """
    Componente físico básico. Armazena apenas o peso.
    """
    def __init__(self, initial_w: float = 0.0):
        self.w = float(initial_w)

    def apply_delta(self, delta: float, w_min: float, w_max: float) -> None:
        """Aplica variação com saturação física (Clamping)."""
        # Matemática nativa sem overhead de C-API do NumPy para escalares:
        # np.clip(self.w + delta, w_min, w_max) só vale a pena para matrizes grandes.
        novo_w = self.w + delta
        if novo_w > w_max:
            self.w = w_max
        elif novo_w < w_min:
            self.w = w_min
        else:
            self.w = novo_w

class AHAHNode: # Código Legacy para compatibilidade
    """
    Nó AHAH Genérico (Legacy).
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

class AnalogAHAHNode: # Código Legacy para compatibilidade
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
    """
    Container vetorizado de nós AHaH.
    Substitui a iteração objeto-por-objeto por matrizes 2D em NumPy (C-BLAS),
    acelerando a inferência e atualização em mais de 10x sem alterar as equações físicas.
    """
    def __init__(self, obs_dim: int, nodes_per_layer: int, y_max: float = 1.0):
        self.obs_dim = int(obs_dim)
        self.nodes_total = int(nodes_per_layer)
        self.y_max = float(y_max)
        
        # Matriz de pesos 2D: [número de fibras, número de spikes de entrada]
        # Inicialização idêntica à original: Gaussiana com média 0 e desvio 0.05
        self.W = np.random.normal(loc=0.0, scale=0.05, size=(self.nodes_total, self.obs_dim)).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula o potencial de todas as fibras em paralelo via produto escalar matricial.
        y = clip(W @ x, -y_max, y_max)
        """
        y = np.dot(self.W, x)
        return np.clip(y, -self.y_max, self.y_max).astype(np.float32)

    def update_matrix(self, active_idx: np.ndarray,
                      hebb_vec: np.ndarray, anti_vec: np.ndarray,
                      decay: float, w_min: float, w_max: float,
                      noise_mean: float = 0.0, noise_std: Optional[float] = None,
                      bias_start_idx: int = -1, bias_mode: str = 'B') -> None:
        """
        Atualiza as sinapses ativas de TODAS as fibras simultaneamente via broadcasting.
        """
        if len(active_idx) == 0:
            return

        # 1. Delta básico: [nodes_total, 1]
        # Para entradas normais (não-bias): delta = hebb + anti_hebb
        delta = hebb_vec + anti_vec # Shape: (nodes_total, 1)

        # 2. Tratamento das sinapses de Bias (se houver e estiverem ativas)
        # Cria uma matriz 2D de delta para as sinapses ativas: [nodes_total, len(active_idx)]
        delta_matrix = np.repeat(delta, len(active_idx), axis=1)

        if bias_start_idx >= 0:
            # Identifica quais das colunas ativas pertencem ao bias
            is_bias = (active_idx >= bias_start_idx)
            if np.any(is_bias):
                if bias_mode == 'B':
                    delta_matrix[:, is_bias] = anti_vec
                elif bias_mode == 'A':
                    delta_matrix[:, is_bias] = -hebb_vec - anti_vec
                # Modo 'C' já é hebb + anti_hebb por padrão

        # 3. Aplicação do Decaimento Físico nas sinapses ativas
        W_sub = self.W[:, active_idx]
        delta_matrix -= (decay * W_sub)

        # 4. Aplicação do Ruído Térmico/Estocástico
        if noise_std is not None and noise_std > 0.0:
            delta_matrix += np.random.normal(loc=noise_mean, scale=noise_std, size=delta_matrix.shape)

        # 5. Aplicação direta e saturação física [w_min, w_max]
        self.W[:, active_idx] = np.clip(W_sub + delta_matrix, w_min, w_max)