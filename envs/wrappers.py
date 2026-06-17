import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

class CartPoleWrapper(gym.Wrapper):
    """ CartPole Episódico: Score = Recompensa Nativa (1.0 por passo). """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["is_success"] = 1 if truncated else 0
        info["step_score"] = float(reward)
        return obs, reward, terminated, truncated, info
    
class ContinuousCartPoleWrapper(gym.Wrapper):
    """ CartPole Contínuo: Mantém o Shaping Interno essencial para o Lifelong Learning. """
    def __init__(self, env, success_threshold=100, max_steps_episodic=500):
        super().__init__(env)
        self.success_threshold = success_threshold 
        self.max_steps_episodic = max_steps_episodic
        self.fail_theta = 12.0 * (np.pi / 180.0)
        self.max_x = 2.4 
        self.max_theta = 30.0 * (np.pi / 180.0)
        self.consecutive_steps = 0
        self.total_successes = 0

    def reset(self, **kwargs):
        self.consecutive_steps = 0
        self.total_successes = 0 
        if hasattr(self.env.unwrapped, "steps_beyond_terminated"):
            # O  "# type: ignore" serve para o pylance não reclamar
            self.env.unwrapped.steps_beyond_terminated = None # type: ignore
        return self.env.reset(**kwargs)

    def step(self, action):
        if hasattr(self.env.unwrapped, "steps_beyond_terminated"):
            self.env.unwrapped.steps_beyond_terminated = None # type: ignore

        obs, raw_reward, terminated, truncated, info = self.env.step(action)
        x, x_dot, theta, theta_dot = self.env.unwrapped.state # type: ignore
        
        hit_wall = False
        if x > self.max_x: x = self.max_x; x_dot = 0.0; hit_wall = True
        elif x < -self.max_x: x = -self.max_x; x_dot = 0.0; hit_wall = True
        if theta > self.max_theta: theta = self.max_theta; theta_dot = 0.0; hit_wall = True
        elif theta < -self.max_theta: theta = -self.max_theta; theta_dot = 0.0; hit_wall = True
            
        if hit_wall:
            self.env.unwrapped.state = (x, x_dot, theta, theta_dot) # type: ignore
            obs = np.array(self.env.unwrapped.state, dtype=np.float32) # type: ignore

        terminated = False
        theta_abs = abs(theta)
            
        if not hit_wall and theta_abs < self.fail_theta:
            self.consecutive_steps += 1
        else:
            self.consecutive_steps = 0

        is_success_step = 0
        if self.consecutive_steps > 0 and self.consecutive_steps % self.max_steps_episodic == 0:
            is_success_step = 1
            self.total_successes += 1

        if self.total_successes >= self.success_threshold:
            truncated = True
        
        # O Contínuo usa a centralização como Score/Reward
        score_continuo = max(0.0, 1.0 - (theta_abs / self.fail_theta))
        if hit_wall or theta_abs >= self.fail_theta:
            score_continuo = -1.0
            
        info["step_score"] = score_continuo
        info["is_success"] = is_success_step
        reward = score_continuo 
            
        return obs, reward, terminated, truncated, info

class BipedalWalkerDiscreteWrapper(gym.ActionWrapper):
    """ (Mantém-se igual) """
    def __init__(self, env, is_independent: bool = False):
        super().__init__(env)
        self.is_independent = is_independent
        if self.is_independent:
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
        else:
            self.action_space = spaces.Discrete(81)

    def action(self, action):
        if self.is_independent:
            return (np.array(action, dtype=np.float32) - 1.0)
        else:
            act = int(action)
            return np.array([(act // 27) % 3 - 1, (act // 9) % 3 - 1, (act // 3) % 3 - 1, act % 3 - 1], dtype=np.float32)
    
class BipedalWalkerWrapper(gym.Wrapper):
    """ BipedalWalker Episódico: Score e Reward baseados puramente na física do Gym. """
    def __init__(self, env):
        super().__init__(env)
        self.cum_reward = 0.0

    def reset(self, **kwargs):
        self.cum_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cum_reward += float(reward)
        info["is_success"] = 1 if (terminated or truncated) and self.cum_reward >= 300.0 else 0
        info["step_score"] = float(reward)
        return obs, reward, terminated, truncated, info
    
class ReservoirWrapper(gym.Wrapper):
    """
    Reservatório Físico (Echo State Network / LSM), 
    para ampliar o espaço de representação dos agentes.
    Utiliza a equação Leaky Integrate-and-Fire com saturação.
    """
    def __init__(self, env, res_size=100, spectral_radius=0.95, leak_rate=0.2, sparsity=0.1, is_random=False):
        super().__init__(env)
        self.res_size = int(res_size)
        self.leak_rate = float(leak_rate)
        self.is_random = is_random
        
        obs_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.res_size,), dtype=np.float32)
        
        # 1. PESOS DE ENTRADA
        # Projeta a observação para o tamanho do reservatório
        self.W_in = (np.random.rand(self.res_size, obs_dim) * 2.0 - 1.0).astype(np.float32)
        
        # Tenta gerar uma rede física válida até 10 vezes para evitar azar
        valid_matrix = False
        for _ in range(10):
            # TOPOLOGIA DO RESERVATÓRIO
            W_res_raw = np.random.randn(self.res_size, self.res_size).astype(np.float32)
            
            # ESPARSIDADE
            mask = (np.random.rand(self.res_size, self.res_size) < float(sparsity)).astype(np.float32)
            W_res_raw *= mask
            
            # ESTABILIZAÇÃO (Raio Espectral)
            eigvals = np.linalg.eigvals(W_res_raw)
            max_eig = np.max(np.abs(eigvals))
            
            if max_eig > 1e-9:
                self.W_res = (W_res_raw * (float(spectral_radius) / max_eig)).astype(np.float32)
                valid_matrix = True
                break 

        # Se falhou 10 vezes seguidas, o problema está na estrutura da matriz
        if not valid_matrix:
            raise ValueError(
                f"[ERRO FATAL] Falha ao criar reservatório físico. "
                f"Com tamanho (size={self.res_size}) e esparsidade ({sparsity}), "
                f"as conexões geram matrizes nulas ou nilpotentes. "
                f"Aumente o tamanho ou reduza a esparsidade."
            )
            
        # h é o estado atual das ondas/tensões no reservatório
        self.h = np.zeros(self.res_size, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Zera a memória a cada novo episódio
        self.h = np.zeros(self.res_size, dtype=np.float32) 
        if self.is_random: return obs, info
        return self._process(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_random: return obs, reward, terminated, truncated, info
        return self._process(obs), reward, terminated, truncated, info

    def _process(self, obs):
        # Equação Leaky Integrate-and-Fire com Clipping:
        # h_t​=(1−leak_rate)⋅h_(t−1)+leak_rate⋅clip(W_in.​x_t​+W_res.​h_(t−1)​,−1,1)
        excitation = np.dot(self.W_in, obs) + np.dot(self.W_res, self.h)
        self.h = (1.0 - self.leak_rate) * self.h + self.leak_rate * np.clip(excitation, -1.0, 1.0)
        return self.h

    def get_reservoir_state(self):
        return {"W_in": self.W_in, "W_res": self.W_res}

    def load_reservoir(self, path):
        data = np.load(path)
        self.W_in = data["W_in"]
        self.W_res = data["W_res"]