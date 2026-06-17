# models/base.py
import numpy as np
from typing import Any, Union, Optional, Dict

class BaseAgent:
    """Interface padrão para todos os agentes."""
    
    def get_action(self, obs: np.ndarray) -> Union[int, np.ndarray]:
        raise NotImplementedError
    
    # Assinatura Universal: Aceita tudo que o DQN precisa, com defaults para segurança
    def update(self, obs: np.ndarray, action: Any, reward: float, 
               next_obs: Optional[np.ndarray] = None, done: bool = False):
        pass
    
    # Universal: Todos têm, mas só DQN faz algo
    def decay_epsilon(self):
        pass

    def get_state(self) -> Any:
        """Retorna o estado do modelo para ser salvo (dict ou bytes)."""
        return None
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Retorna metadados da arquitetura neural (Nós, Camadas, Parâmetros)."""
        return {"total_parameters": 0, "architecture": "Unknown"}
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
    
    def reset(self):
        """Reseta memória temporal (LSTM, Capacitores)."""
        pass
    
    # Propriedades opcionais que podem existir
    epsilon: float
    network: Any