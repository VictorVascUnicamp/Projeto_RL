# envs/reacher_2d.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Optional, Tuple, Any, Dict

class Reacher2DEnv(gym.Env):
    """
    Ambiente Reacher 2D
    
    Simula um braço robótico planar com N juntas.
    Objetivo: Levar a ponta do braço (End-Effector) até um alvo.
    
    - Modos de Observação:
        - 'geometric': Seno/Cosseno + Vetor Distância.
        - 'physical': Seno/Cosseno + Distância de 2 Sensores na cabeça.
    - Modos de Aprendizado:
        - 'episodic': Reseta a cada N passos ou sucesso.
        - 'continuous': Nunca reseta o braço, apenas muda o alvo (Lifelong Learning).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, num_joints: int = 2, 
            render_mode: Optional[str] = None, 
            delta: float = np.pi / 32, 
            min_angle: float = -np.pi,
            max_angle: float = np.pi,
            max_steps: int = 250,
            learning_mode: str = "episodic",
            target_count: int = 100,
            obs_mode: str = "geometric"):
        
        super().__init__()
        
        # --- Parâmetros Físicos do Robô ---
        self.num_joints = int(num_joints)
        self.link_lengths = np.ones(self.num_joints) * 0.5 # Comprimento fixo dos elos
        self.head_width = 0.2 # Distância entre os olhos (sensores) na cabeça
        
        # Limites de Atuação
        self.max_angle = float(max_angle)
        self.min_angle = float(min_angle)
        self.delta = float(delta) # Velocidade angular por passo
        
        # --- Parâmetros da Tarefa ---
        self.max_steps = int(max_steps)
        self.target_radius = 0.05 # Tamanho do alvo
        self.ee_radius = 0.05     # Tamanho da ponta
        
        # Configuração de Aprendizado
        self.learning_mode = learning_mode
        self.success_threshold = int(target_count) # Meta para o modo contínuo
        self.targets_hit_session = 0 # Contador de sucessos
        
        self.obs_mode = obs_mode # 'geometric' ou 'physical'

        # Configuração de Renderização
        self.render_mode = render_mode
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

        # Espaço de Ação: [0, 1, 2] para cada junta (MultiDiscrete)
        # 0: -delta, 1: 0, 2: +delta
        self.action_space = spaces.MultiDiscrete([3] * self.num_joints)
        
        # Espaço de Observação:
        # Geometric: 2*N (sin/cos) + 2 (vetor)
        # Physical:  2*N (sin/cos) + 2 (olhos)
        # Ambos têm o mesmo tamanho, mas semântica diferente.
        obs_dim = (2 * self.num_joints) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Cache de Cinemática:
        # Armazena a posição da ponta (ee_pos) e dos olhos para não recalcular
        # várias vezes no mesmo passo (uma vez para reward, outra para obs).
        self.cache_kinematics = (np.zeros(2), None, None)

        self.reset()

    def _calculate_value(self, dist: float) -> float:
        """
        Função de Potencial: Transforma distância [0, inf] em Valor [1, 0].
        """
        return 1.0 / (dist + 1.0)

    def _update_kinematics_cache(self):
        """
        Atualiza o cache de posições.
        Só calcula a posição dos olhos se o modo de observação exigir ('physical').
        """
        need_eyes = (self.obs_mode == "physical")
        self.cache_kinematics = self._forward_kinematics(calc_eyes=need_eyes)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reinicia o episódio.
        Posiciona o braço aleatoriamente e gera um novo alvo.
        """
        super().reset(seed=seed)
        self.angles = np.random.uniform(self.min_angle, self.max_angle, size=self.num_joints)
        self.target = self._sample_target()
        self.steps = 0
        self.targets_hit_session = 0
        
        # Atualiza a física inicial
        self._update_kinematics_cache()
        ee_pos = self.cache_kinematics[0]
        
        # Inicializa o valor anterior para o reward começar correto
        dist = np.linalg.norm(ee_pos - self.target)
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executa um passo de simulação.
        1. Aplica ação (atualiza ângulos, posição).
        2. Calcula colisão e recompensas.
        3. Gerencia fim de episódio (ou troca de alvo no contínuo).
        """
        # 1. Movimento (Vetorizado)
        # Converte [0, 1, 2] -> [-1, 0, 1]
        moves = action.astype(np.int8) - 1
        self.angles += moves * self.delta
        self.angles = np.clip(self.angles, self.min_angle, self.max_angle)
        self.steps += 1
        
        # 2. Física (Atualiza Cache)
        self._update_kinematics_cache()
        ee_pos = self.cache_kinematics[0]
        
        # Distância ao alvo
        dist = np.linalg.norm(ee_pos - self.target)
        hit_target = bool(dist < (self.target_radius + self.ee_radius))

        # --- CÁLCULO DE RECOMPENSA ---
    
        reward = self._calculate_value(float(dist))
        if hit_target:
            reward += 100.0

        terminated = False
        truncated = False

        # 3. Lógica de Sucesso
        if hit_target:
            if self.learning_mode == "continuous":
                # Modo Contínuo: Não reseta o braço. Apenas move o alvo.
                self.targets_hit_session += 1
                if self.targets_hit_session >= self.success_threshold:
                    terminated = True # Fim da sessão de treino
                else:
                    self.target = self._sample_target()
            else:
                # Modo Episódico: Acabou.
                terminated = True

        # Limite de tempo
        if self.steps >= self.max_steps:
            truncated = True

        # Info adicional para logs
        info = {
            "step_score": reward,
            "targets_hit": self.targets_hit_session,
            "is_success": 1 if hit_target else 0
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Monta o vetor de observação baseado no modo escolhido."""
        # Seno e Cosseno são usados em ambos os modos para evitar descontinuidade angular
        sin_theta = np.sin(self.angles)
        cos_theta = np.cos(self.angles)

        if self.obs_mode == "geometric":
            # Modo Matemático: O agente recebe o vetor distância.
            ee_pos = self.cache_kinematics[0]
            vec_diff = self.target - ee_pos
            return np.concatenate([sin_theta, cos_theta, vec_diff]).astype(np.float32)
        
        elif self.obs_mode == "physical":
            # Modo Sensorial: O agente recebe a distância medida pelos "olhos".
            _, eye_l, eye_r = self.cache_kinematics
            
            # Fallback de segurança: Se o cache não tem olhos (foi um step rápido), calcula agora.
            if eye_l is None or eye_r is None:
                _, eye_l, eye_r = self._forward_kinematics(calc_eyes=True)

            dist_l = np.linalg.norm(self.target - eye_l)
            dist_r = np.linalg.norm(self.target - eye_r)
            
            return np.concatenate([sin_theta, cos_theta, [dist_l, dist_r]]).astype(np.float32)
            
        return np.array([])

    def _forward_kinematics(self, calc_eyes: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Cinemática Direta.
        Calcula a posição (x,y) da ponta e, opcionalmente, dos olhos.
        """
        x, y = 0.0, 0.0
        angle_sum = 0.0
        last_sin, last_cos = 0.0, 1.0
        
        # Percorre as juntas somando vetores
        for i in range(self.num_joints):
            angle_sum += self.angles[i]
            last_sin = np.sin(angle_sum)
            last_cos = np.cos(angle_sum)
            x += self.link_lengths[i] * last_cos
            y += self.link_lengths[i] * last_sin
        
        ee_pos = np.array([x, y], dtype=np.float32)
        
        if not calc_eyes:
            return ee_pos, None, None
            
        # Calcula olhos perpendiculares ao último elo
        offset = self.head_width / 2.0
        
        # Rotação de +90 graus (Esquerda) e -90 graus (Direita)
        # Matematicamente simplificado usando sin/cos já calculados
        eye_l_x = x - (offset * last_sin)
        eye_l_y = y + (offset * last_cos)
        eye_r_x = x + (offset * last_sin)
        eye_r_y = y - (offset * last_cos)
        
        return ee_pos, np.array([eye_l_x, eye_l_y], dtype=np.float32), np.array([eye_r_x, eye_r_y], dtype=np.float32)

    def _sample_target(self) -> np.ndarray:
        """Gera um alvo aleatório dentro da área alcançável."""
        total_len = np.sum(self.link_lengths)
        r = np.random.uniform(0.1, total_len - 0.05)
        theta = np.random.uniform(0, 2 * np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    def render(self) -> Optional[np.ndarray]:
        """
        Renderiza o estado atual.
        Sempre desenha os olhos (visão humana), independente do modo de observação do agente.
        """
        if self.render_mode not in ["human", "rgb_array"]: return None
        
        # Lazy Import para não pesar no treino cego
        import matplotlib.pyplot as plt
        
        if self.fig is None or self.ax is None: 
            self.fig, self.ax = plt.subplots(figsize=(4,4))
        
        # Garante cálculo dos olhos para desenho
        ee_pos, eye_l, eye_r = self._forward_kinematics(calc_eyes=True)
        
        joint_positions = [np.array([0.0, 0.0])]
        angle_sum = 0.0
        for i in range(self.num_joints):
            angle_sum += self.angles[i]
            prev = joint_positions[-1]
            new = prev + self.link_lengths[i] * np.array([np.cos(angle_sum), np.sin(angle_sum)])
            joint_positions.append(new)

        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        
        # Desenha Braço
        self.ax.plot(*zip(*joint_positions), marker='o', color='blue', linewidth=3)
        
        # Desenha Alvo e Ponta
        tgt_tuple = (float(self.target[0]), float(self.target[1]))
        ee_tuple = (float(ee_pos[0]), float(ee_pos[1]))
        self.ax.add_patch(Circle(tgt_tuple, self.target_radius, color='red'))
        self.ax.add_patch(Circle(ee_tuple, self.ee_radius, color='green', alpha=0.3))
        
        # Desenha Olhos
        if eye_l is not None and eye_r is not None:
            self.ax.plot(eye_l[0], eye_l[1], marker='.', color='gold', markersize=8)
            self.ax.plot(eye_r[0], eye_r[1], marker='.', color='gold', markersize=8)
        
        mode_str = f"Obs: {self.obs_mode} | Step {self.steps}"
        if self.learning_mode == "continuous":
            mode_str += f" | Hits: {self.targets_hit_session}"
        self.ax.set_title(mode_str)

        if self.render_mode == "human": plt.pause(0.01); return None
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf, dtype=np.uint8)[..., :3]

    def close(self) -> None:
        if self.fig: plt.close(self.fig); self.fig = None