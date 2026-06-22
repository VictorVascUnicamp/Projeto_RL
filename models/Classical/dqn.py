# models/Classical/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Any, Union, List, Optional

from models.base import BaseAgent

# --- A Rede Neural ---
class QNetwork(nn.Module):
    """
    Aproximador de Função Universal (MLP).
    Objetivo: Receber o Estado (sensores) e prever o 'Valor Q' (qualidade) 
    de cada ação possível.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = [128, 128]):
        super(QNetwork, self).__init__()
        layers = []
        in_dim = input_dim
        
        # Constrói as camadas ocultas dinamicamente baseadas na lista do config
        # Ex: [256, 256] cria duas camadas densas com 256 neurônios cada.
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            # ReLU (Rectified Linear Unit): Zera valores negativos. 
            layers.append(nn.ReLU()) 
            in_dim = h_dim
            
        # Camada de Saída: Conecta a última camada oculta ao número de ações.
        # Não tem ativação pois Q-values podem ser qualquer número real (+/-).
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Passagem direta (Inferência)."""
        return self.net(x)

# --- O Agente ---
class DQNAgent(BaseAgent):
    """
    Implementação de DQN (Deep Q-Network).
    - Suporte a ações Multi-Discretas via achatamento (Combinatorial).
    - Decaimento linear de exploração (Epsilon).
    - Double Network (Policy + Target) para estabilidade.
    """
    def __init__(self, config: Dict[str, Any], max_decay_calls: int): 
        self.config = config
        env_conf = config["environment"]
        dqn_conf = config["DQN"]
        
        # 1. Dimensões de Entrada (Sensores)
        self.obs_dim = len(env_conf["observation_spec"]) 
        self.action_type = env_conf["action_type"]

        # Extrai limites para normalização estática Min-Max [-1, 1]
        specs = env_conf.get("observation_spec", [])
        if specs:
            self.obs_mins = np.array([float(s["min"]) for s in specs], dtype=np.float32)
            self.obs_maxs = np.array([float(s["max"]) for s in specs], dtype=np.float32)
            self.obs_ranges = np.where((self.obs_maxs - self.obs_mins) > 1e-9, (self.obs_maxs - self.obs_mins), 1e-9)
        else:
            self.obs_mins, self.obs_ranges = None, None

        # 2. Configuração do Espaço de Ação (Achatamento 3^N)
        # O DQN original só entende escolhas únicas (botão A ou botão B).
        # O Reacher tem juntas simultâneas.
        # Solução: Criamos uma "Meta-Ação" para cada combinação possível.
        if self.action_type == "multi_joint_3state":
            self.num_joints = env_conf.get("num_joints", 2)
            self.actions_per_joint = 3 # {-1, 0, +1}
            self.independent_joints = dqn_conf.get("independent_joints", False)
            
            if self.independent_joints:
                # (Legado) Trata cada junta separada. Não recomendado.
                self.act_dim = self.num_joints * self.actions_per_joint
            else:
                # (Recomendado) Espaço Combinatorial: 3^N ações.
                # Permite aprender correlação (mover ombro E cotovelo juntos).
                self.act_dim = self.actions_per_joint ** self.num_joints
        else:
            # CartPole: Simples, 2 ações.
            self.act_dim = 2

        # 3. Hiperparâmetros de Aprendizado
        self.gamma = float(dqn_conf.get("gamma", 0.99)) # Importância do futuro
        self.batch_size = int(dqn_conf.get("batch_size", 64))
        self.lr = float(dqn_conf.get("learning_rate", 1e-3))
        self.target_update_freq = int(dqn_conf.get("target_update_frequency", 100))
        
        # Leitura da Arquitetura Neural
        use_res = env_conf.get("use_reservoir", False) or config["agent"]["type"] == "ESN"
        # Reservatório não usa camadas encondidas
        self.hidden_layers = dqn_conf.get("hidden_layers", [128, 128]) if not use_res else []

        # 4. Exploração (Epsilon-Greedy)
        self.epsilon = float(dqn_conf.get("epsilon_start", 1.0))
        self.epsilon_min = float(dqn_conf.get("epsilon_end", 0.05))
        self.ratio = float(dqn_conf.get("epsilon_decay_ratio", 0.5))
        
        # Calcula passo de decaimento para durar X% do treino total
        decay_duration = int(max_decay_calls * self.ratio)
        decay_duration = max(1, decay_duration)
        self.epsilon_decrement = (self.epsilon - self.epsilon_min) / decay_duration
        
        # 5. Inicialização de Redes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Rede Principal: Treina a cada passo.
        self.q_net: QNetwork = QNetwork(self.obs_dim, self.act_dim, self.hidden_layers).to(self.device)
        
        # Rede Alvo: Cópia estável para calcular o erro (Ground Truth temporário).
        self.target_net: QNetwork = QNetwork(self.obs_dim, self.act_dim, self.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=int(dqn_conf.get("buffer_capacity", 10000)))
        self.steps = 0

    def _index_to_action(self, index: int) -> Union[int, np.ndarray]:
        """
        Tradutor: Converte a decisão abstrata da rede (Inteiro) 
        para o comando físico dos motores (Vetor).
        Ex: Índice 5 -> Base 3 -> [1, 2] -> Motores [0, +1]
        """
        if not self.action_type == "multi_joint_3state": return index
        
        # Conversão de base 10 para base 3
        actions = []
        idx = index
        for _ in range(self.num_joints):
            actions.append(idx % self.actions_per_joint)
            idx //= self.actions_per_joint
        # Inverte para ficar na ordem [junta1, junta2,...]
        return np.array(actions[::-1], dtype=np.int64)

    def get_action(self, obs: np.ndarray) -> Union[int, np.ndarray]:
        # Normalização
        obs = self._normalize_obs(obs)

        # 1. Exploração
        if random.random() < self.epsilon:
            if self.action_type == "multi_joint_3state":
                # Retorna vetor aleatório
                return np.random.randint(0, 3, size=self.num_joints)
            else:
                return random.randint(0, self.act_dim - 1)

        # 2. Exploitação (Rede)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        
        # Lógica de Decisão
        if self.action_type == "multi_joint_3state":
            if self.independent_joints:
                # A rede retorna [batch, num_joints * 3]
                # Nós remodelamos para [num_joints, 3] e pegamos o maior de cada trio.
                q_reshaped = q_values.view(self.num_joints, 3)
                actions = torch.argmax(q_reshaped, dim=1).cpu().numpy()
                return actions
            else:
                # Modo Combinatorial 
                # A rede retorna [batch, 3^num_joints]
                # Pegamos o índice único vencedor
                action_idx = int(torch.argmax(q_values).item())
                return self._index_to_action(action_idx)
        else:
            return int(torch.argmax(q_values).item())

    def update(self, obs: np.ndarray, action: Any, reward: float, 
               next_obs: Optional[np.ndarray] = None, done: bool = False):
        """Ciclo de Aprendizado (Replay + Backprop)."""

        # Normalização
        obs = self._normalize_obs(obs)  
        if next_obs is not None:
            next_obs = self._normalize_obs(next_obs)
        
        # 1. Preparação: Se ação for vetor, converte de volta para índice
        # para economizar memória no buffer (guarda 1 int em vez de N ints).
        store_action = action
        if self.action_type == "multi_joint_3state" and not self.independent_joints:
            action_idx = 0
            for a in action:
                action_idx = action_idx * self.actions_per_joint + a
            store_action = action_idx

        # 2. Armazenamento na Memória de Longo Prazo
        self.memory.append((obs, store_action, reward, next_obs, done))
        self.steps += 1

        # Só treina se tiver dados suficientes para um lote estatístico
        if len(self.memory) < self.batch_size:
            return

        # 3. Sonho (Experience Replay): Amostra memórias aleatórias
        batch = random.sample(self.memory, self.batch_size)
        obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)
        
        # Move para GPU
        obs_b = torch.FloatTensor(np.array(obs_b)).to(self.device)
        next_obs_b = torch.FloatTensor(np.array(next_obs_b)).to(self.device)
        rew_b = torch.FloatTensor(rew_b).unsqueeze(1).to(self.device)
        done_b = torch.FloatTensor(np.array(done_b, dtype=np.float32)).unsqueeze(1).to(self.device)
        act_b = torch.LongTensor(act_b).unsqueeze(1).to(self.device)
        
        # 4. Cálculo do Erro TD (Temporal Difference)
        # O que a rede acha que vale (Q_pred)
        q_curr = self.q_net(obs_b).gather(1, act_b)
        
        # O que realmente vale (R + gamma * Melhor_Futuro)
        with torch.no_grad():
            q_next_max = self.target_net(next_obs_b).max(1)[0].unsqueeze(1)
            target = rew_b + (1 - done_b) * self.gamma * q_next_max
        
        # Erro quadrático médio
        loss = nn.MSELoss()(q_curr, target)

        # 5. Ajuste dos Pesos (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 6. Sincronização Lenta da Rede Alvo
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def get_architecture_info(self) -> Dict[str, Any]:
        # Soma todos os pesos e viés (bias) da rede
        total_params = sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)
        
        return {
            "input_dim": self.obs_dim,
            "output_dim": self.act_dim,
            "layers": len(self.hidden_layers) + 1,
            "architecture": f"({self.obs_dim} -> {self.hidden_layers} -> {self.act_dim})",
            "total_parameters": total_params
        }

    def decay_epsilon(self):
        """Reduz a curiosidade do agente ao longo do tempo."""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mins is not None and self.obs_ranges is not None:
            norm = (obs - self.obs_mins) / self.obs_ranges
            return (norm * 2.0) - 1.0 # Mapeia para [-1, 1]
        return obs

    def get_state(self):
        return self.q_net.state_dict()
    
    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
