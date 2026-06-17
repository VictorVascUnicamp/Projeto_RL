# models/AHAH/encoders.py
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import deque

class BaseEncoder:
    """
    Interface base para todos os encoders sensoriais.
    Define o contrato que o Controller espera: encode, save/load state e reset.
    """
    def __init__(self):
        self.total_spike_dim = 0
        self.sparse_mode = False

    def encode(self, obs: np.ndarray, learn: bool = False) -> Tuple[np.ndarray, int]:
        """
        Converte observação contínua em vetor de spikes binários.
        Args:
            obs: Vetor de observação do ambiente.
            learn: Se True, o encoder pode adaptar seus parâmetros internos (apenas Adaptive).
        Returns:
            (spikes, dimensão_total)
        """
        raise NotImplementedError
    
    def save_state(self) -> Dict[str, Any]: 
        """Retorna estado interno (pesos) para salvamento."""
        return {}
    
    def load_state(self, state: Dict[str, Any]): 
        """Carrega estado interno."""
        pass
    
    def reset(self): 
        """Reseta memória de curto prazo (usado no TemporalEncoder)."""
        pass

class GeneralSpikeEncoder(BaseEncoder):
    """
    Encoder Fixo Otimizado (com Lookup Tables).
    
    Funcionamento:
    Divide o espaço de entrada [min, max] em 'bins' fixos.
    
    Otimização:
    Usa Tabelas de Consulta (LUTs) pré-calculadas para converter índices inteiros
    """
    def __init__(self, specs: List[Dict[str, Any]], bias_per_node: int = 1):
        super().__init__()
        self.specs = specs
        self.bias_per_node = int(bias_per_node)
        
        # Calcula tamanho total da saída (soma dos bits de todos os sensores)
        self.input_bits_total = sum(int(s["bits"]) for s in specs)
        self.total_spike_dim = (self.input_bits_total * 2) + self.bias_per_node
        
        # --- OTIMIZAÇÃO: Pré-cálculo de Lookup Tables (LUT) ---
        # Cria uma matriz de bits para cada resolução usada (ex: uma tabela para 8 bits, outra para 12).
        self.luts = {}
        unique_bits = set(int(s["bits"]) for s in specs)
        
        for b in unique_bits:
            num_bins = 1 << b # 2^bits
            # Matriz [num_bins, 2*b]
            lut = np.zeros((num_bins, 2 * b), dtype=np.uint8)
            
            # Geração vetorizada dos bits
            indices = np.arange(num_bins)
            powers = 1 << np.arange(b - 1, -1, -1)
            # Broadcasting para criar matriz de bits (True/False)
            bit_vals = (indices[:, None] & powers[None, :]) > 0
            
            # Codificação One-Hot [Not, Yes]
            lut[:, 0::2] = 1 - bit_vals.astype(np.uint8) # Colunas pares
            lut[:, 1::2] = bit_vals.astype(np.uint8)     # Colunas ímpares
            
            self.luts[b] = lut

        # --- OTIMIZAÇÃO: Pré-cálculo de vetores de Normalização ---
        # Evita acessar dicionários e fazer contas repetidas dentro do loop de encode
        self.mins = np.array([float(s["min"]) for s in specs], dtype=np.float32)
        self.maxs = np.array([float(s["max"]) for s in specs], dtype=np.float32)
        # Evita divisão por zero com 1e-9
        self.ranges = np.where((self.maxs - self.mins) > 1e-9, (self.maxs - self.mins), 1e-9)

    def encode(self, obs: np.ndarray, learn: bool = False) -> Tuple[np.ndarray, int]:
        # Validação de dimensão
        if len(obs) != len(self.specs): 
            raise ValueError(f"Dimensão incorreta. Esperado {len(self.specs)}, recebido {len(obs)}")

        spike_list = []
        
        # Iteração sobre os sensores (ainda necessária pois tipos podem variar entre linear/circular)
        for i, val in enumerate(obs):
            spec = self.specs[i]
            bits = int(spec["bits"])
            stype = spec.get("type", "linear")
            
            # Normalização Rápida (usando arrays pré-calculados)
            min_v = self.mins[i]
            # Garante que o valor esteja dentro dos limites físicos definidos no config
            val_clamped = np.clip(val, min_v, self.maxs[i])
            
            idx = 0
            if stype == "circular":
                # Lógica para ângulos que dão a volta (wrap-around)
                pos = (val_clamped - min_v) / self.ranges[i]
                nb = 1 << bits
                idx = int(np.clip(pos * (nb - 1), 0, nb - 1))
            else:
                # Lógica Linear Padrão
                norm = (val_clamped - min_v) / self.ranges[i]
                levels = (1 << bits) - 1
                idx = int(round(norm * levels))
            
            # Consulta na Tabela (Instantâneo)
            spike_list.append(self.luts[bits][idx])

        # Concatena todos os vetores de bits em um único array
        spikes_obs = np.concatenate(spike_list)

        # Adiciona o Bias no final
        if self.bias_per_node > 0:
            spikes = np.concatenate([spikes_obs, np.ones(self.bias_per_node, dtype=np.uint8)])
        else:
            spikes = spikes_obs
            
        return spikes, int(spikes.shape[0])


class AdaptiveSpikeEncoder(BaseEncoder):
    """
    Encoder AHAH Adaptativo.
    
    Emula uma população de neurônios memristivos competindo para representar a entrada.
    Usa a regra 'Synaptic Anti-Hebbian Online Learning' (SAOL).
    
    Se um neurônio vence (dispara), seus pesos são ajustados para suprimir sua ativação
    futura para aquele mesmo padrão (Anti-Hebbiano). Isso força os neurônios a se 
    espalharem e cobrirem a distribuição de probabilidade dos dados reais.
    
    Otimização:
    Utiliza operações matriciais (NumPy) para calcular a ativação de milhares de nós
    simultaneamente, sem instanciar objetos Python individuais.
    """
    def __init__(self, specs: List[Dict[str, Any]], bias_per_node: int = 1, 
                 encoder_beta: float = 0.05, w_min: float = -1.0, w_max: float = 1.0):
        super().__init__()
        self.specs = specs
        self.bias_per_node = int(bias_per_node)
        self.beta = encoder_beta # Taxa de aprendizado/supressão
        
        # Limites físicos dos pesos
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        
        self.input_bits_total = sum(int(s["bits"]) for s in specs)
        self.total_spike_dim = (self.input_bits_total * 2) + self.bias_per_node
        
        # Verifica se todos os sensores têm o mesmo número de bits
        # Se sim, ativamos o modo matricial (Turbo)
        first_bits = int(specs[0]["bits"])
        self.is_homogeneous = all(int(s["bits"]) == first_bits for s in specs)
        self.bits_common = first_bits

        # --- Preparação das LUTs (Igual ao General) ---
        self.luts = {}
        unique_bits = set(int(s["bits"]) for s in specs)
        for b in unique_bits:
            num_bins = 1 << b
            lut = np.zeros((num_bins, 2 * b), dtype=np.uint8)
            indices = np.arange(num_bins)
            powers = 1 << np.arange(b - 1, -1, -1)
            bit_vals = (indices[:, None] & powers[None, :]) > 0
            lut[:, 0::2] = 1 - bit_vals.astype(np.uint8)
            lut[:, 1::2] = bit_vals.astype(np.uint8)
            self.luts[b] = lut

        # --- Inicialização dos Pesos (Matrizes) ---
        # W_val: Pesos conectados ao sinal de entrada
        # W_bias: Pesos conectados ao bias interno (limiar)
        self.W_val_list = []
        self.W_bias_list = []
        
        for spec in specs:
            bits = int(spec["bits"])
            num_bins = 1 << bits
            
            # Inicializa pesos para criar hiperplanos distribuídos uniformemente
            # w_val * x + w_bias = 0  -> Equação da fronteira
            
            # Inicia condutância alta (dentro dos limites físicos)
            start_val = min(1.0, self.w_max)
            w_vals = np.full(num_bins, start_val, dtype=np.float32)
            
            # Distribui os bias para cobrir o espaço [-1, 1]
            centers = np.linspace(-1.0, 1.0 - (2.0/num_bins), num_bins, dtype=np.float32)
            w_biases = -centers
            
            # Garante limites físicos
            w_biases = np.clip(w_biases, self.w_min, self.w_max)
            
            self.W_val_list.append(w_vals)
            self.W_bias_list.append(w_biases)
            
        # Se for homogêneo, converte listas em Matrizes 2D para cálculo em bloco
        if self.is_homogeneous:
            self.W_val_matrix = np.array(self.W_val_list, dtype=np.float32)   # [NumSensores, NumBins]
            self.W_bias_matrix = np.array(self.W_bias_list, dtype=np.float32) # [NumSensores, NumBins]

        # Pré-cálculo de normalização
        self.mins = np.array([float(s["min"]) for s in specs], dtype=np.float32)
        self.maxs = np.array([float(s["max"]) for s in specs], dtype=np.float32)
        self.ranges = np.where((self.maxs - self.mins) > 1e-9, (self.maxs - self.mins), 1e-9)

    def encode(self, obs: np.ndarray, learn: bool = True) -> Tuple[np.ndarray, int]:
        # Seleciona a implementação mais rápida
        if self.is_homogeneous:
            return self._encode_matrix(obs, learn)
        else:
            return self._encode_loop(obs, learn)

    def _encode_matrix(self, obs: np.ndarray, learn: bool) -> Tuple[np.ndarray, int]:
        """Caminho Otimizado (Matricial). Processa todos os sensores juntos."""
        
        # 1. Normalização Vetorizada (Mapeia [Min, Max] -> [-1.0, 1.0])
        # Crucial para garantir simetria na regra de aprendizado.
        norm_vals = (obs - self.mins) / self.ranges
        inputs = (norm_vals * 2.0) - 1.0 
        
        # 2. Forward (Broadcasting)
        # Calcula y = Wx + b para todos os bins de todos os sensores
        # inputs[:, None] expande dimensão para multiplicar corretamente
        activations = (self.W_val_matrix * inputs[:, None]) + self.W_bias_matrix
        
        # 3. Winner-Take-All
        # Encontra o índice do neurônio com maior ativação em cada sensor
        winners = np.argmax(activations, axis=1)
        
        # 4. Aprendizado SAOL (Apenas se learn=True)
        if learn:
            # Seleciona as ativações vencedoras
            rows = np.arange(len(obs))
            winner_acts = activations[rows, winners]
            
            # Regra Anti-Hebbiana: dW = -Beta * y * x
            d_val = -self.beta * winner_acts * inputs
            d_bias = -self.beta * winner_acts * 1.0
            
            # Atualiza pesos
            self.W_val_matrix[rows, winners] += d_val
            self.W_bias_matrix[rows, winners] += d_bias
            
            # Clamping 
            self.W_val_matrix[rows, winners] = np.clip(self.W_val_matrix[rows, winners], self.w_min, self.w_max)
            self.W_bias_matrix[rows, winners] = np.clip(self.W_bias_matrix[rows, winners], self.w_min, self.w_max)

        # 5. Geração de Spikes
        # Consulta a LUT para converter índices em bits
        spikes_obs = self.luts[self.bits_common][winners].flatten()
        
        if self.bias_per_node > 0:
            spikes = np.concatenate([spikes_obs, np.ones(self.bias_per_node, dtype=np.uint8)])
        else:
            spikes = spikes_obs
            
        return spikes, int(spikes.shape[0])

    def _encode_loop(self, obs: np.ndarray, learn: bool) -> Tuple[np.ndarray, int]:
        """Caminho Flexível (Loop). Usado se sensores tiverem bits diferentes."""
        spike_list = []
        for i, raw_val in enumerate(obs):
            bits = int(self.specs[i]["bits"])
            
            # Normalização Bipolar [-1, 1]
            norm = (float(raw_val) - self.mins[i]) / self.ranges[i]
            inp = (norm * 2.0) - 1.0
            
            # Recupera vetores de pesos da lista
            w_v = self.W_val_list[i]
            w_b = self.W_bias_list[i]
            
            # Forward
            acts = w_v * inp + w_b
            win = int(np.argmax(acts))
            
            # Aprendizado
            if learn:
                act = acts[win]
                # Atualização escalar
                d_val = -self.beta * act * inp
                d_bias = -self.beta * act
                
                w_v[win] = np.clip(w_v[win] + d_val, self.w_min, self.w_max)
                w_b[win] = np.clip(w_b[win] + d_bias, self.w_min, self.w_max)
            
            # Lookup
            spike_list.append(self.luts[bits][win])
            
        spikes_obs = np.concatenate(spike_list)
        
        if self.bias_per_node > 0:
            spikes = np.concatenate([spikes_obs, np.ones(self.bias_per_node, dtype=np.uint8)])
        else:
            spikes = spikes_obs
            
        return spikes, int(spikes.shape[0])

    def save_state(self) -> Dict[str, Any]:
        """Salva pesos. Converte matrizes para listas para JSON."""
        if self.is_homogeneous:
            return {
                "W_val": [row.tolist() for row in self.W_val_matrix], 
                "W_bias": [row.tolist() for row in self.W_bias_matrix]
            }
        else:
            return {
                "W_val": [w.tolist() for w in self.W_val_list], 
                "W_bias": [w.tolist() for w in self.W_bias_list]
            }

    def load_state(self, state: Dict[str, Any]):
        """Carrega pesos salvos."""
        if "W_val" in state:
            w_val_data = [np.array(w, dtype=np.float32) for w in state["W_val"]]
            w_bias_data = [np.array(w, dtype=np.float32) for w in state["W_bias"]]
            
            if self.is_homogeneous:
                self.W_val_matrix = np.array(w_val_data, dtype=np.float32)
                self.W_bias_matrix = np.array(w_bias_data, dtype=np.float32)
                # Sincroniza listas para segurança
                self.W_val_list = list(self.W_val_matrix)
                self.W_bias_list = list(self.W_bias_matrix)
            else:
                self.W_val_list = w_val_data
                self.W_bias_list = w_bias_data

class TemporalEncoderWrapper(BaseEncoder):
    """
    Time Embedding (Tap Delay Line).
    
    Funcionalidade:
    Mantém um buffer circular (deque) dos últimos N padrões de bits.
    Concatena tudo em um vetor único. Isso permite que o AHAH 
    detecte padrões temporais (movimento, direção) olhando para a "foto" do histórico.
    
    Exemplo (Memória=3):
    Input T: [1, 0]
    Histórico: [[1, 0], [0, 0], [0, 1]]
    Saída: [1, 0, 0, 0, 0, 1]
    """
    def __init__(self, base_encoder: BaseEncoder, memory_size: int = 1):
        super().__init__()
        self.base = base_encoder
        self.memory_size = int(memory_size)
        
        # A dimensão aumenta linearmente com a memória
        self.total_spike_dim = self.base.total_spike_dim * self.memory_size
        
        # Buffer
        self.history = deque(maxlen=self.memory_size)

    def reset(self):
        """Limpa a memória (ex: no início de um episódio)."""
        self.history.clear()
        self.base.reset() # Propaga reset

    def encode(self, obs: np.ndarray, learn: bool = False) -> Tuple[np.ndarray, int]:
        # 1. Obtém spikes do momento atual
        current_spikes, _ = self.base.encode(obs, learn)
        
        # 2. Preenche histórico
        if len(self.history) == 0:
            # Se vazio, repete o atual para encher (Cold Start)
            for _ in range(self.memory_size):
                self.history.append(current_spikes)
        else:
            # Adiciona novo, descarta mais antigo
            self.history.append(current_spikes)
        
        # 3. Concatena
        full_spikes = np.concatenate(list(self.history))
        
        return full_spikes, self.total_spike_dim

    def save_state(self) -> Dict[str, Any]:
        # Salva os pesos do encoder base
        state = self.base.save_state()
        # Converte a fila (deque) em uma matriz numpy para salvar com segurança
        if len(self.history) > 0:
            state["history"] = np.array(list(self.history), dtype=np.uint8)
        else:
            state["history"] = np.array([], dtype=np.uint8)
        return state

    def load_state(self, state: Dict[str, Any]):
        # Carrega os pesos base
        self.base.load_state(state)
        self.history.clear()
        
        # Restaura a memória temporal
        if "history" in state and len(state["history"]) > 0:
            for item in state["history"]:
                self.history.append(item)