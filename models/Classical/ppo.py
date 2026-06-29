# models/Classical/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from typing import Dict, Any, Union, Optional, List, Tuple

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box

class ActorCritic(nn.Module):
    """
    A 'Cabeça' e o 'Corpo' do Agente.
    Herda de nn.Module, o que dá superpoderes como .to(device) e .parameters().
    """
    def __init__(self, input_dim: int, action_dim: int, action_type = "analog_muscle", num_joints: int = 1,
                 hidden_layers: list = [64, 64]):
        super().__init__()
        self.action_type = action_type
        
        # --- Construção da Rede Neural (Feature Extractor) ---
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim)) 
            layers.append(nn.Tanh())                
            in_dim = h_dim
        
        # nn.Sequential: Empacota as camadas.
        self.network = nn.Sequential(*layers)

        # --- Crítico (Avaliador) ---
        # Estima o Valor V(s), qualidade do estado.
        self.critic = nn.Linear(in_dim, 1)

        # --- Ator (Decisão) ---
        # Estima a probabilidade de cada ação.
        if self.action_type == "multi_joint_3state":
            # Cria uma cabeça de decisão independente para cada junta do robô.
            self.actor_heads = nn.ModuleList([
                nn.Linear(in_dim, action_dim) for _ in range(num_joints)
            ])
        elif self.action_type == "analog_muscle":
            # Modo Contínuo
            self.actor_mean = nn.Linear(in_dim, action_dim)
            # Desvio padrão logarítmico independente (Parâmetro treinável)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = nn.Linear(in_dim, action_dim)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Passa o input pela rede e depois pelo crítico."""
        return self.critic(self.network(x))

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
        """
        Calcula tudo o que o PPO precisa num único passo.
        x: Estado (Tensor).
        action: Ação tomada anteriormente.
        """
        # 1. Processa a visão do mundo (comum a ator e crítico)
        hidden = self.network(x)
        
        # 2. Gera Distribuições de Probabilidade
        if self.action_type == "multi_joint_3state":
            # Calcula os 'logits' (pontos brutos) para cada junta
            logits_list = [head(hidden) for head in self.actor_heads]
            # Categorical aplica Softmax: converte pontos em % (ex: 0.1, 0.9)
            probs_list = [Categorical(logits=l) for l in logits_list]
            
            if action is None:
                # Inferência: Sorteia uma ação para cada junta baseada nas %
                action_list = [probs.sample() for probs in probs_list]
                # Empilha: [Junta1, Junta2]
                action = torch.stack(action_list, dim=1)
            
            # Calcula a probabilidade logarítmica da ação escolhida
            # (Soma os logs = Multiplica as probabilidades)
            log_prob = torch.stack([probs.log_prob(action[:, i]) for i, probs in enumerate(probs_list)], 
                                   dim=1).sum(1)
            entropy = torch.stack([probs.entropy() for probs in probs_list], dim=1).sum(1)
        elif self.action_type == "analog_muscle":
            action_mean = self.actor_mean(hidden)
            
            # [CORREÇÃO DE ESTABILIDADE FÍSICA] Evita que a variância convirja para 0 ou Infinito.
            # -20 evita divisão por zero; 2 evita explosão numérica.
            clamped_logstd = torch.clamp(self.actor_logstd, min=-20.0, max=2.0)
            action_logstd = clamped_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            
            # Soma as probabilidades e entropias das 4 juntas
            log_prob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)
        else:
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action)
            entropy = probs.entropy()

        # Retorna a tupla completa: Ação, Probabilidade da Ação, Incerteza, Valor do Estado
        return action, log_prob, entropy, self.critic(hidden)

from models.base import BaseAgent

class PPOAgent(BaseAgent):
    """
    Implementação do agente para Proximal Policy Optimization (PPO).
    Algoritmo On-Policy: Aprende, descarta os dados e coleta novos.
    """
    def __init__(self, config: Dict[str, Any], env: Optional[gym.Env] = None):
        self.config = config
        env_conf = config["environment"]
        ppo_conf = config["PPO"]
        
        # Extrai limites para normalização estática Min-Max [-1, 1]
        specs = env_conf["observation_spec"]
        if specs:
            self.obs_mins = np.array([float(s["min"]) for s in specs], dtype=np.float32)
            self.obs_maxs = np.array([float(s["max"]) for s in specs], dtype=np.float32)
            self.obs_ranges = np.where((self.obs_maxs - self.obs_mins) > 1e-9, (self.obs_maxs - self.obs_mins), 1e-9)
        else:
            self.obs_mins, self.obs_ranges = None, None

        self.action_type = env_conf["action_type"]

        # Configuração de Dimensões
        if env is not None:
            obs_space = env.observation_space
            act_space = env.action_space

            if obs_space is None:
                raise ValueError("O ambiente não possui observation_space definido.")
            
            self.obs_dim = int(obs_space.shape[0]) # type: ignore
            
            # Pergunta se o espaço de ações é vetorial (nvec) ou escalar (n)
            if isinstance(act_space, MultiDiscrete):
                
                # Tamanho do vetor nvec (ex: len([3, 3]) = 2)
                self.num_joints = len(act_space.nvec)
                
                # Pega o primeiro valor (ex: 3)
                self.act_dim = int(act_space.nvec[0])

            elif isinstance(act_space, Discrete):
                self.num_joints = 1
                
                # Pega o número total de ações
                self.act_dim = int(act_space.n)
            
            elif self.action_type == "analog_muscle":
                self.num_joints = int(env_conf.get("num_joints", 4))
                self.act_dim = self.num_joints

            else:
                raise ValueError(f"Espaço de ação não suportado: {type(act_space)}")
        else:
            self.obs_dim = len(env_conf["observation_spec"])
            if self.action_type == "multi_joint_3state":
                self.num_joints = int(env_conf.get("num_joints", 2))
                self.act_dim = 3
            elif self.action_type == "analog_muscle":
                self.num_joints = int(env_conf.get("num_joints", 4))
                self.act_dim = self.num_joints
            else:
                self.num_joints = 1
                self.act_dim = 2

        use_res = ppo_conf.get("use_reservoir", False) or config["agent"]["type"] == "ESN"

        # Hiperparâmetros
        self.lr = float(ppo_conf.get("learning_rate", 3e-4))
        self.gamma = float(ppo_conf.get("gamma", 0.99))
        self.gae_lambda = float(ppo_conf.get("gae_lambda", 0.95))
        self.clip_coef = float(ppo_conf.get("clip_coef", 0.2))
        self.ent_coef = float(ppo_conf.get("ent_coef", 0.01))
        self.vf_coef = float(ppo_conf.get("vf_coef", 0.5))
        self.update_epochs = int(ppo_conf.get("update_epochs", 10))
        self.batch_size = int(ppo_conf.get("batch_size", 64))
        self.update_horizon = int(ppo_conf.get("update_horizon", 2048))
        # Se usar reservatório, a rede vira um Readout linear direto (lista vazia)
        self.hidden_layers = ppo_conf.get("hidden_layers", [64, 64]) if not use_res else []

        self._clear_buffer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- DICA DE TIPO ---
        # Dizemos explicitamente que self.network é da classe ActorCritic.
        # Agora o Pylance sabe quais métodos ela tem.
        self.network: ActorCritic = ActorCritic(
            self.obs_dim, self.act_dim, self.action_type, self.num_joints, self.hidden_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)

        self.last_log_prob = None
        self.last_value = None

    def reset(self): pass

    def decay_epsilon(self): pass

    def get_action(self, obs: np.ndarray) -> Union[int, np.ndarray]:
        """
        Recebe observação (Numpy) -> Retorna Ação (Numpy/Int).
        """
        # Normalização
        obs = self._normalize_obs(obs)

        # 1. Prepara o dado: Transforma array numpy em Tensor e adiciona dimensão de lote [1, N]
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # 2. Roda a rede sem calcular gradientes
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(obs_t)
            
        # 3. Guarda metadados na memória RAM (necessário para o treino depois)
        self.last_log_prob = log_prob
        self.last_value = value
        
        # 4. Converte a ação de volta
        if self.action_type == "multi_joint_3state":
            # [0]: Tira do lote de tamanho 1
            return action.cpu().numpy()[0].astype(np.int64)
        elif self.action_type == "analog_muscle":
            return action.cpu().numpy()[0].astype(np.float32)
        else:
            return int(action.item())

    def update(self, obs, action, reward, next_obs=None, done=False):
        """Acumula dados. Se encher o buffer, treina."""

        # Normalização
        obs = self._normalize_obs(obs)             
        if next_obs is not None:
            next_obs = self._normalize_obs(next_obs)

        self.buffer_obs.append(obs)
        self.buffer_actions.append(action)
        self.buffer_logprobs.append(self.last_log_prob)
        self.buffer_values.append(self.last_value)
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)

        if len(self.buffer_obs) >= self.update_horizon:
            self._train(next_obs, done)
            self._clear_buffer()
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mins is not None and self.obs_ranges is not None:
            norm = (obs - self.obs_mins) / self.obs_ranges
            return (norm * 2.0) - 1.0 # Mapeia para [-1, 1]
        return obs

    def _clear_buffer(self):
        self.buffer_obs = []; self.buffer_actions = []; self.buffer_logprobs = []
        self.buffer_rewards = []; self.buffer_dones = []; self.buffer_values = []

    def _train(self, next_obs, done):
        if len(self.buffer_obs) == 0: return

        # Transforma listas Python em Tensores PyTorch grandes
        b_obs = torch.tensor(np.array(self.buffer_obs), dtype=torch.float32).to(self.device)
        b_actions = torch.tensor(np.array(self.buffer_actions)).to(self.device)
        b_logprobs = torch.cat(self.buffer_logprobs)
        b_values = torch.cat(self.buffer_values).flatten()
        b_rewards = torch.tensor(self.buffer_rewards, dtype=torch.float32).to(self.device)
        b_dones = torch.tensor(self.buffer_dones, dtype=torch.float32).to(self.device)

        # GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            next_value = self.network.get_value(next_obs_t).reshape(1, -1)

            # 1. Preparação
            # advantages vai guardar os A_t calculados
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0

            # 2. Loop Reverso (Do futuro para o passado)
            # t vai de (T-1) até 0.
            for t in reversed(range(len(b_rewards))):
                if t == len(b_rewards) - 1:
                    nextnonterminal = 1.0 - int(done)
                    nextvalues = next_value[0, 0]
                else:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = b_values[t + 1]

                # 3. Cálculo do Erro TD (delta)
                # delta = reward + gamma * V(next) - V(current)
                delta = b_rewards[t] + self.gamma * nextvalues * nextnonterminal - b_values[t]

                # 4. Cálculo do GAE (Recursivo)
                # A_t = delta + (gamma * lambda * A_t+1)
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            # 5. O Retorno (Target para o Crítico)
            # Como A = R - V, então R = A + V
            returns = advantages + b_values

        # Treino em Lotes (Epochs)
        b_inds = np.arange(len(b_obs))
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]
                
                # Re-avalia as ações passadas com a rede atual
                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                # Ratio: Probabilidade Nova / Probabilidade Velha
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Loss PPO (Clipada para segurança)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                
                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss

                # Atualiza Pesos
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

    def get_state(self) -> Any:
        return self.network.state_dict()

    def get_architecture_info(self) -> Dict[str, Any]:
        # Soma todos os pesos e viés (bias) da rede
        total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        return {
            "input_dim": self.obs_dim,
            "output_dim": self.act_dim,
            "layers": len(self.hidden_layers) + 1,
            "architecture": f"({self.obs_dim} -> {self.hidden_layers} -> {self.act_dim})",
            "total_parameters": total_params
        }

    def save(self, path: str): torch.save(self.network.state_dict(), path)

    def load(self, path: str): self.network.load_state_dict(torch.load(path))