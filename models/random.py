# models/random.py
import numpy as np
from typing import Dict, Any, Union, Optional

from models.base import BaseAgent

class RandomAgent(BaseAgent):
    """
    Agente Baseline que toma decisões puramente aleatórias.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        env_conf = config["environment"]
        self.action_type = env_conf["action_type"]
        self.num_joints = int(env_conf.get("num_joints", 1))

    def get_action(self, obs: np.ndarray) -> Any:
        """
        Retorna uma ação aleatória válida conforme o tipo configurado.
        """
        if self.action_type == "multi_joint_3state":
            # Retorna vetor [0, 1, 2] para cada junta
            return np.random.randint(0, 3, size=self.num_joints)
        
        elif self.action_type == "discrete_2state":
            # Retorna 0 ou 1
            return int(np.random.randint(0, 2))
        
        elif self.action_type == "analog_muscle":
            # BipedalWalker: Retorna vetor de 4 torques contínuos entre -1.0 e 1.0
            return np.random.uniform(-1.0, 1.0, size=self.num_joints).astype(np.float32)
        
        else:
            raise ValueError(f"Tipo de ação desconhecido: {self.action_type}")


