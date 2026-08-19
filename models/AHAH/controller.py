# models/ahah/controller.py
import numpy as np
from typing import Optional, Any, Dict, List, Tuple, Union, cast
import io

from models.AHAH.core import AHAHLayer
# Importa todos os componentes de encoder necessários
from models.AHAH.encoders import GeneralSpikeEncoder, AdaptiveSpikeEncoder, TemporalEncoderWrapper

from models.base import BaseAgent

class AHAHController(BaseAgent):
    """
    Controlador motor neuromórfico AHAH por meio de competição entre juntas.
    Usado em problema de aprendizado pro reforço.
    """
    def __init__(self, config: Dict[str, Any], is_eval: bool = False):
        self.config = config
        self.is_eval = is_eval
        ahah_conf = config["AHAH"]
        env_conf = config["environment"]
        
        # --- Parâmetros Físicos  ---
        self.alpha = float(ahah_conf["alpha"])   # Taxa Hebbiana 
        self.beta = float(ahah_conf["beta"])     # Taxa Anti-Hebbiana 
        self.w_min = float(ahah_conf["w_min"])   # Peso Mínimo
        self.w_max = float(ahah_conf["w_max"])   # Peso Máximo
        self.decay = float(ahah_conf["decay"]) # Decaimento L1
        
        # Ruído térmico/analógico
        self.noise_mean = float(ahah_conf.get("noise_mean", 0.0))
        std = ahah_conf.get("noise_std", 0.0)
        self.noise_std = float(std) if std > 0 else None

        # --- Arquitetura de Decisão ---
        self.tie_eps = 1e-4 # Margem de empate
        self.y_max = 1.0    # Tensão máxima
        
        # Configuração do Viés (Bias)
        self.bias_update_mode = ahah_conf.get("bias_update_mode", "B")
        self.bias_per_node = int(ahah_conf.get("bias_per_node", 1))
        
        # Tipo de Ação (Definido pelo Ambiente)
        self.action_type = env_conf["action_type"]
        self.fibers_per_unit = int(ahah_conf["fibers_per_joint"]) 
        
        if self.fibers_per_unit % 2 != 0:
            raise ValueError("Fibras devem ser pares (Agonista/Antagonista).")

        if "num_joints" in env_conf:
            self.num_units = int(env_conf["num_joints"])
        else:
            self.num_units = 1 

        self.fibers_total = self.num_units * self.fibers_per_unit

        # --- Configuração do Encoder ---
        specs = env_conf["observation_spec"]
        # O bias do encoder é o mesmo do controller neste config, mas poderia ser separado
        enc_bias = self.bias_per_node 
        
        # 1. Escolha do Encoder Base
        self.encoder_type = ahah_conf["encoder_type"]
        
        # Inicializa variável para evitar erro de 'possibly unbound' do Pylance
        base_encoder = None 
        
        if self.encoder_type == "adaptive":
            enc_beta = float(ahah_conf.get("encoder_beta", 0.05))
            base_encoder = AdaptiveSpikeEncoder(
                specs, 
                bias_per_node=enc_bias, 
                encoder_beta=enc_beta,
                w_min=self.w_min,
                w_max=self.w_max
            )
        else:
            base_encoder = GeneralSpikeEncoder(specs, bias_per_node=enc_bias)
            
        # 2. Aplicação do Wrapper Temporal (Memória de Curto Prazo)
        use_temporal = ahah_conf.get("use_temporal_encoder", False)
        
        if use_temporal:
            # Memória definida em passos (inteiro)
            mem_size = int(ahah_conf.get("temporal_memory", 3))
            self.encoder = TemporalEncoderWrapper(base_encoder, memory_size=mem_size)
        else:
            self.encoder = base_encoder 
        
        # Camada Neural (Lazy Init: criada no primeiro uso para pegar dimensão correta)
        self.layer: Optional[AHAHLayer] = None
        
        # Cache de Spikes (Para não recalcular encoder na mesma iteração)
        self.last_obs_bytes = None
        self.last_spikes = None
        self.prev_reward = 0.0 

        # --- Variáveis do Monte Carlo ---
        self.use_mc = ahah_conf.get("use_monte_carlo", False)
        self.mc_buffer_size = int(ahah_conf.get("mc_buffer_size", 500))
        self.trajectory_buffer = []  # O Traço de Elegibilidade
        self.ep_reward_sum = 0.0     # Recompensa Acumulada
        self.baseline = 0.0          # Régua Histórica
        self.baseline_init = False   # Controle da primeira jogada

    def _ensure_layer(self, spike_dim: int):
        """Inicializa a camada física se ainda não existir."""
        if self.layer is None:
            self.layer = AHAHLayer(
                obs_dim=spike_dim,
                nodes_per_layer=self.fibers_total,
                y_max=self.y_max,
            )

    def _get_spikes_cached(self, obs: np.ndarray, learn: bool) -> np.ndarray:
        """
        Codifica a observação ou recupera do cache.
        Se learn=True, o encoder adaptativo atualiza seus pesos internos.
        """
        obs_bytes = obs.tobytes()
        # Se não estamos aprendendo e a obs é idêntica, usa cache
        if not learn and self.last_obs_bytes == obs_bytes and self.last_spikes is not None:
            return self.last_spikes
            
        # Caso contrário, processa
        spikes, _ = self.encoder.encode(obs, learn=learn)
        
        self.last_obs_bytes = obs_bytes
        self.last_spikes = spikes
        return spikes

    def get_action(self, obs: np.ndarray) -> Union[np.ndarray, int]:
        """
        Ciclo de Decisão (Inferência).
        Não permite aprendizado no encoder aqui (learn=False).
        """
        spikes = self._get_spikes_cached(obs, learn=False)
        
        # Garante que a camada existe com o tamanho correto de spikes
        self._ensure_layer(int(spikes.shape[0]))
        assert self.layer is not None
        
        # Calcula ativação
        scores = self.layer.forward(spikes)
        
        if self.action_type == "multi_joint_3state":
            # Reacher: Decisão por junta
            actions = []
            for i in range(self.num_units):
                s_idx = i * self.fibers_per_unit
                e_idx = s_idx + self.fibers_per_unit
                y_local = scores[s_idx:e_idx]
                
                direction = self._decide_direction(y_local)
                actions.append(direction + 1)
            return np.array(actions, dtype=np.int64)
        
        elif self.action_type == "analog_muscle":
            # BipedalWalker: Controle de Força Analógica por Recrutamento de Fibras
            actions =[]
            for i in range(self.num_units):
                s_idx = i * self.fibers_per_unit
                e_idx = s_idx + self.fibers_per_unit
                y_local = scores[s_idx:e_idx]
                
                half = self.fibers_per_unit // 2
                
                # Conta quantas fibras romperam a inércia (y > 0)
                agon_ativas = np.count_nonzero(y_local[:half] > 0)
                anta_ativas = np.count_nonzero(y_local[half:] > 0)
                
                # O Torque é a força líquida dividida pela força máxima possível.
                # Resultado garantido entre [-1.0, 1.0] em degraus suaves!
                torque = (agon_ativas - anta_ativas) / half
                actions.append(torque)
                
            return np.array(actions, dtype=np.float32)
        
        elif self.action_type == "discrete_2state":
            # CartPole: Decisão global
            direction = self._decide_direction(scores) 
            return 1 if direction > 0 else 0
        else:
            raise ValueError(f"Tipo de ação desconhecido: {self.action_type}")

    def _decide_direction(self, y_fiber_subset: np.ndarray) -> int:
        """
        Competição Agonista vs Antagonista.
        Retorna: 1, -1 ou 0 (Empate: não se move).
        """
        half = self.fibers_per_unit // 2
        y_agon = float(np.sum(y_fiber_subset[:half]))
        y_anta = float(np.sum(y_fiber_subset[half:]))
        
        if y_agon - y_anta > self.tie_eps: return 1
        elif y_anta - y_agon > self.tie_eps: return -1
        else: return 0

    
    def update(self, obs: np.ndarray, action: Any, reward: float, 
               next_obs: Any = None, done: bool = False) -> None:
        """
        Ciclo de Aprendizado.
        Lógica de Musculos Agonista/Antagonista
        Suporta lógica instantânea (Greedy) e Retornos em Bloco (Monte Carlo).
        """
        spikes = self._get_spikes_cached(obs, learn=True)
        assert self.layer is not None
        layer_ref = self.layer 

        if self.use_mc:
            # 1. Armazena o evento na memória de curto prazo (Traço de Elegibilidade)
            self.trajectory_buffer.append((spikes, action))
            self.ep_reward_sum += float(reward)
            
            # 2. Avalia se o bloco temporal terminou
            if done or len(self.trajectory_buffer) >= self.mc_buffer_size:
                
                # Inicializa o Baseline na primeira rodada
                if not self.baseline_init:
                    self.baseline = self.ep_reward_sum
                    self.baseline_init = True
                
                # Avalia o Desempenho com Margem de Segurança (Deadzone)
                delta_r = self.ep_reward_sum - self.baseline
                
                # Cria uma tolerância dinâmica
                margin = abs(self.baseline * 0.05) 
                
                # A margem de tolerância perdoa erros pequenos
                if delta_r < -margin:
                    sgn_r = -1.0 
                else:
                    sgn_r = 1.0 
                    
                bias_start = int(spikes.shape[0]) - self.bias_per_node
                
                # 3. Retrospectiva: Aplica a atualização física nas memórias do buffer
                for past_spikes, past_action in self.trajectory_buffer:
                    active_idx = np.nonzero(past_spikes)[0]
                    
                    # Sub-função física para aplicar correntes (Memristor)
                    def update_unit_mc(unit_idx, direction_taken):
                        start = unit_idx * self.fibers_per_unit
                        end = start + self.fibers_per_unit
                        half = self.fibers_per_unit // 2
                        
                        for k in range(start, end):
                            node = layer_ref.get_node(k) 
                            yk = node.potential(past_spikes)
                            
                            anti = -self.beta * yk
                            hebb = 0.0
                            rel_k = k - start
                            is_agonist = (rel_k < half)
                            
                            if (direction_taken > 0 and is_agonist) or \
                               (direction_taken < 0 and not is_agonist):
                                hebb = self.alpha * sgn_r
                            
                            node.update_weights(
                                active_idx=active_idx,
                                hebb=hebb,
                                anti_hebb=anti,
                                decay=self.decay,     
                                w_min=self.w_min,     
                                w_max=self.w_max,     
                                noise_mean=self.noise_mean,
                                noise_std=self.noise_std,
                                bias_start_idx=bias_start,
                                bias_mode=self.bias_update_mode
                            )

                    # Despacha a atualização das fibras conforme a ação gravada
                    if self.action_type == "multi_joint_3state":
                        acts = np.array(past_action).flatten()
                        for i in range(self.num_units):
                            update_unit_mc(i, int(acts[i]) - 1)

                    elif self.action_type == "analog_muscle":
                        directions = np.sign(past_action)
                        for i in range(self.num_units):
                            update_unit_mc(i, int(directions[i]))
                            
                    elif self.action_type == "discrete_2state":
                        direction = 1 if int(past_action) == 1 else -1
                        update_unit_mc(0, direction)

                # 4. Ajuste do Baseline (APENAS se não for Validação)
                if not self.is_eval:
                    # Média Móvel Exponencial (EMA) para suavizar a régua
                    self.baseline = 0.9 * self.baseline + 0.1 * self.ep_reward_sum
                
                # 5. Limpa a memória para o próximo bloco temporal
                self.trajectory_buffer.clear()
                self.ep_reward_sum = 0.0

        else:
            # =========================================================
            # Lógica original gulosa (recompensa por passo)
            # =========================================================
            active_idx = np.nonzero(spikes)[0]
            bias_start = int(spikes.shape[0]) - self.bias_per_node
            
            # Calcula a derivada local da recompensa
            r_local = float(reward)
            delta_r = r_local - self.prev_reward
            self.prev_reward = r_local
            
            # Reforço dependente da derivada
            if delta_r > 0: sgn_r = 1.0
            elif delta_r < 0: sgn_r = -1.0 
            else: sgn_r = 0.0
            
            # Função auxiliar para atualizar um grupo de fibras (uma junta ou motor)
            def update_unit(unit_idx, direction_taken):
                start = unit_idx * self.fibers_per_unit
                end = start + self.fibers_per_unit
                half = self.fibers_per_unit // 2
                
                for k in range(start, end):
                    node = layer_ref.get_node(k) 
                    yk = node.potential(spikes)
                    
                    # 1. Anti-Hebbiano: todos os nós ativos
                    anti = -self.beta * yk
                    
                    # 2. Hebbiano: apenas nós das fibras vencedoras
                    hebb = 0.0
                    rel_k = k - start
                    is_agonist = (rel_k < half)
                    if (direction_taken > 0 and is_agonist) or \
                    (direction_taken < 0 and not is_agonist):
                        hebb = self.alpha * sgn_r
                    
                    # Injeta todos os parâmetros físicos no nó
                    node.update_weights(
                        active_idx=active_idx,
                        hebb=hebb,
                        anti_hebb=anti,
                        decay=self.decay,     
                        w_min=self.w_min,     
                        w_max=self.w_max,     
                        noise_mean=self.noise_mean,
                        noise_std=self.noise_std,
                        bias_start_idx=bias_start,
                        bias_mode=self.bias_update_mode
                    )

            # Despacha atualização para as unidades
            if self.action_type == "multi_joint_3state":
                actions = np.array(action).flatten()
                for i in range(self.num_units):
                    # action[i] é {0,1,2}, direction é {-1,0,1}
                    update_unit(i, int(actions[i]) - 1)

            elif self.action_type == "analog_muscle":
                # Extrai o sinal da ação (-1, 0, 1) para saber qual fibra reforçar
                directions = np.sign(action)
                for i in range(self.num_units):
                    update_unit(i, int(directions[i]))
                    
            elif self.action_type == "discrete_2state":
                # CartPole: 0 ou 1 -> -1 ou 1
                direction = 1 if int(action) == 1 else -1
                update_unit(0, direction)

    def get_state(self) -> bytes:
        """
        Gera o arquivo .npz diretamente na memória RAM.
        """
        if self.layer is None:
            return b""
            
        # 1. Coleta os pesos
        W = []
        for node in self.layer.nodes:
            W.append([s.w for s in node.synapses])
        
        # 2. Monta o dicionário de dados
        save_dict = {
            "W": np.array(W, dtype=np.float32),
            "encoder_state": self.encoder.save_state()
        }
        
        # 3. Salva em Buffer de Memória (BytesIO)
        memory_file = io.BytesIO()
        
        # O numpy escreve o ZIP dentro dessa variável 'memory_file'
        np.savez(memory_file, **save_dict)
        
        # 4. Retorna o conteúdo binário
        return memory_file.getvalue()
    
    def get_architecture_info(self) -> Dict[str, Any]:
        info = {
            "input_dim": len(self.config["environment"]["observation_spec"]),
            "spike_dim": self.encoder.total_spike_dim,
            "output_fibers": self.fibers_total,
            "layers": 2 if self.encoder_type == "adaptive" else 1,
            "architecture": f"({len(self.config['environment']['observation_spec'])} -> {self.encoder.total_spike_dim} -> {self.fibers_total})"
        }
        
        # Parâmetros da Camada Motora (Memristores)
        controller_params = self.encoder.total_spike_dim * self.fibers_total
        
        # Parâmetros do Encoder (Se for adaptativo, usa memristores reais. Se fixo, são LUTs virtuais)
        if self.encoder_type == "adaptive":
            encoder_params = info["input_dim"] * (self.encoder.total_spike_dim // info["input_dim"]) * 2
            info["total_parameters"] = controller_params + encoder_params
        else:
            info["total_parameters"] = controller_params
            
        return info

    def reset(self):
        """Reseta a memória temporal (se houver)."""
        if hasattr(self.encoder, "reset"):
            self.encoder.reset()
        self.prev_reward = 0.0 
        self.trajectory_buffer.clear()
        self.ep_reward_sum = 0.0

    def save(self, path: str):
        """Salva pesos e estado do encoder."""
        if self.layer is None: return
        W = []
        for node in self.layer.nodes:
            W.append([s.w for s in node.synapses])
        
        save_dict = {
            "W": np.array(W, dtype=np.float32),
            "encoder_state": self.encoder.save_state()
        }
        np.savez(path, **save_dict)

    def load(self, path: str):
        """Carrega pesos e estado do encoder."""
        data = np.load(path, allow_pickle=True)
        W = data["W"]
        self._ensure_layer(W.shape[1])
        if self.layer is None: return 
        
        idx = 0
        for node in self.layer.nodes:
            for i, s in enumerate(node.synapses):
                s.w = float(W[idx, i])
            idx += 1
            
        if "encoder_state" in data:
            self.encoder.load_state(data["encoder_state"].item())
        
        self.last_obs_bytes = None