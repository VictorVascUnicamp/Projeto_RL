# envs/factory.py
import os
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Union, Optional

# Importa Wrappers e Ambientes Customizados
from envs.wrappers import CartPoleWrapper, ContinuousCartPoleWrapper, BipedalWalkerDiscreteWrapper, BipedalWalkerWrapper
try:
    from envs.reacher_2d import Reacher2DEnv
except ImportError:
    Reacher2DEnv = None

def make_env(config: Dict[str, Any], reservoir_load_path: Optional[str] = None) -> gym.Env:
    """
    Fábrica de Ambientes.
    Instancia o ambiente correto baseada na configuração processada.
    """
    # 1.Busca de Parametros
    env_conf = config["environment"]
    train_conf = config["training"]
    
    name = env_conf["name"]
    target_count = int(train_conf["target_count"])
    learning_mode = train_conf["learning_mode"]
    episodic_max_steps = int(env_conf["max_steps"])
    episodes = int(train_conf["episodes"])

    # 2. Centralização da Duração
    if learning_mode == "continuous":
        max_steps = int(episodic_max_steps * episodes/2)
    else:
        max_steps = episodic_max_steps
    
    # --- AMBIENTE: REACHER 2D ---
    if name == "Reacher2D":
        if Reacher2DEnv is None: 
            raise ImportError("Classe Reacher2DEnv não encontrada.")
        
        # Extração de parâmetros Específicos do Reacher
        obs_mode = env_conf.get("obs_mode", "geometric")
        delta = float(env_conf.get("granularity_delta_rad", 0.098174)) 
        min_ang = float(env_conf.get("min_angle", -3.14159))
        max_ang = float(env_conf.get("max_angle", 3.14159)) #Pi

        env = Reacher2DEnv(
            num_joints=int(env_conf.get("num_joints", 2)),
            delta=delta,
            min_angle=min_ang,
            max_angle=max_ang,
            max_steps=max_steps,
            render_mode="rgb_array",
            learning_mode=learning_mode,
            target_count=target_count,
            obs_mode=obs_mode
        )
        
    # --- AMBIENTE: CARTPOLE ---
    elif name == "CartPole":
        
        # Inicialização
        env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=max_steps)
        
        if learning_mode == "continuous":
            env = ContinuousCartPoleWrapper(env, success_threshold=target_count, max_steps_episodic=episodic_max_steps)
        else:
            env = CartPoleWrapper(env)

    
    elif name == "BipedalWalker":

        env = gym.make("BipedalWalker-v3", render_mode="rgb_array", max_episode_steps=max_steps)

        # Aplica conversor de ações para DQN
        if config["agent"]["type"] == "DQN":
            independent = config.get("definitions", {}).get("agents", {}).get("DQN", {}).get("independent_joints", False)
            env = BipedalWalkerDiscreteWrapper(env, is_independent=independent)

        env = BipedalWalkerWrapper(env)
        
    else:
        raise ValueError(f"Ambiente desconhecido: {name}")

    agent_type = config["agent"]["type"]
    is_esn = agent_type == "ESN"
    use_res = env_conf.get("use_reservoir", False) or is_esn
    
    if use_res:
        from envs.wrappers import ReservoirWrapper

        if "ESN" not in config:
            raise KeyError("[ERRO FATAL] Ambiente exige reservatório, mas a chave 'ESN' não foi passada ao factory.")
            
        res_conf = config["ESN"]

        env = ReservoirWrapper(
            env,
            res_size=res_conf.get("reservoir_size", 100),
            spectral_radius=res_conf.get("spectral_radius", 0.95),
            leak_rate=res_conf.get("leak_rate", 0.2),
            sparsity=res_conf.get("sparsity", 0.1),
            is_random=(agent_type == "Random")
        )
        
        # No modo Evaluate, ele carrega exatamente a mesma matriz física que foi treinada
        if reservoir_load_path and os.path.exists(reservoir_load_path):
            env.load_reservoir(reservoir_load_path)
            
    return env