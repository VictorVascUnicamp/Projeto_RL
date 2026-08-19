# utils.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, List
import copy

def clean_for_json(obj: Any) -> Any:
    """
    Sanitiza objetos Python para garantir que sejam serializáveis em JSON.
    Converte NumPy types, Sets, Ellipsis e transforma desconhecidos em string.
    """
    # 1. Tratamento de Ellipsis (...) explícito
    if obj is Ellipsis:
        return "..."

    # 2. Tratamento de Nulo
    if obj is None:
        return None

    # 3. Tratamento de Tipos Básicos 
    if isinstance(obj, (str, bool)):
        return obj
    
    # 4. Tratamento de Números 
    if isinstance(obj, (int, float)):
        return obj 
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
        
    # 5. Tratamento de Arrays NumPy
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist()) # Recurso: converte para lista e limpa a lista
    
    # 6. Tratamento de Dicionários
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    
    # 7. Tratamento de Listas, Tuplas e Sets (Iteráveis)
    # Convertemos tudo para lista para o JSON aceitar
    if isinstance(obj, (list, tuple, set)):
        return [clean_for_json(i) for i in obj]

    # 8. Fallback: Converte para string para não quebrar o salvamento.
    try:
        return str(obj)
    except:
        return "<Unserializable Object>"

def save_stats(path: str, stats: dict):
    """Salva stats usando o sanitizador."""
    
    clean_stats = clean_for_json(stats)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=2, ensure_ascii=False)

def create_results_folder(
    base_dir: str, 
    config: Dict[str, Any], 
    run_type_name: str
) -> str:
    """
    Cria a pasta final dos resultados.
    """
    env = config["environment"]["name"]
    agent = config["agent"]["type"]
    
    # 1. Cria a pasta "Pai" 
    parent_path = os.path.join(base_dir, env, agent, run_type_name)
    os.makedirs(parent_path, exist_ok=True)
    
    # 2. Lógica para criar o nome da pasta final
    mode = config["training"]["learning_mode"][:4]    
    partes_nome = [env, agent, mode]

    use_res = config.get(agent, {}).get("use_reservoir", False)

    if use_res:
        partes_nome.append("Res")
    
    if agent == "AHAH":
        if "AHAH" in config:
            ahah_def = config["AHAH"]
        else:
            ahah_def = config.get("definitions", {}).get("agents", {}).get("AHAH", {})
            
        bias = ahah_def.get("bias_update_mode", "C")
        is_fixed = ahah_def.get("encoder_type", "adaptive") == "fixed"
        use_temp = ahah_def.get("use_temporal_encoder", False)
        
        partes_nome.append(f"Bias{bias}")
        if use_temp:
            partes_nome.append("Temp")
        if is_fixed and not use_res:
            partes_nome.append("Fixed")

        if config.get("definitions", {}).get("agents", {}).get("AHAH", {}).get("use_monte_carlo", False):
            partes_nome.append("MC")
            
    nome_final_base = "_".join(partes_nome)

    # 3. Lógica para impedir sobrescrita de testes
    nome_final = nome_final_base
    counter = 1
    path = os.path.join(parent_path, nome_final)
    
    while os.path.exists(path):
        nome_final = f"{nome_final_base} ({counter})"
        path = os.path.join(parent_path, nome_final)
        counter += 1
        
    # Cria a pasta final e retorna o caminho
    os.makedirs(path, exist_ok=True)
    print(f"[UTILS] Pasta de resultados criada: {path}")
    return path

def create_agent(config: Dict[str, Any], env=None, model_path: Optional[str] = None,
                  decay_budget: int = 1, is_eval: bool = False) -> Any:
    """
    Fábrica universal de agentes. Instancia o agente correto e carrega os pesos se solicitado.
    """
    cfg = copy.deepcopy(config)
    agent_type = cfg["agent"]["type"]

    is_esn = agent_type == "ESN"
    
    if is_esn:
        # Se for o Agente esn, mascaramos ele como o seu Readout Padrão (Ex: PPO)
        agent_type = cfg.get("definitions", {}).get("agents", {}).get("ESN", {}).get("standard_model", "PPO")
        cfg["agent"]["type"] = agent_type

    agent = None
    if agent_type == "AHAH":
        from models.AHAH.controller import AHAHController
        agent = AHAHController(config, is_eval=is_eval)
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[ERRO FATAL] Pesos do modelo não encontrados no caminho: {model_path}")
            agent.load(model_path)
            
    elif agent_type == "DQN":
        from models.Classical.dqn import DQNAgent
        agent = DQNAgent(config, max_decay_calls=decay_budget)
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[ERRO FATAL] Pesos do modelo não encontrados no caminho: {model_path}")
            agent.load(model_path)
            if is_eval:
                min_eps = config.get("definitions", {}).get("agents", {}).get("DQN", {}).get("epsilon_end", 0.05)
                agent.epsilon = float(min_eps)
            
    elif agent_type == "PPO":
        from models.Classical.ppo import PPOAgent
        agent = PPOAgent(config, env=env)
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[ERRO FATAL] Pesos do modelo não encontrados no caminho: {model_path}")
            agent.load(model_path)
            if is_eval:
                agent.network.eval()
            
    elif agent_type == "Random":
        from models.random import RandomAgent
        agent = RandomAgent(config)
    else:
        raise ValueError(f"[Error] Modelo desconhecido.")
        
    return agent

def plot_learning_curve(data_list: List[List[float]],
                        save_path: str,
                        window: int = 50,
                        title: str = "Curva de Aprendizado",
                        x_label: str = "Episódio"):
    """
    Plota as curvas de aprendizado (Média Móvel + Desvio Padrão) de múltiplas sessões.
    Remove os dados brutos para visualização limpa.
    """
    
    plt.figure(figsize=(10, 6))
    
    # Prevenção contra listas vazias
    if not data_list or len(data_list) == 0 or len(data_list[0]) == 0:
        raise ValueError("[ERRO FATAL] Tentativa de plotar gráfico sem dados. A lista de métricas está vazia.")

    min_length = min([len(session) for session in data_list])
    if min_length == 0:
        raise ValueError("[ERRO FATAL] Pelo menos uma das sessões retornou zero passos gravados.")

    # Se tivermos mais de 2000 pontos, fazemos um downsampling na matriz toda para o gráfico não pesar
    if min_length > 2000:
        factor = min_length // 1000
        new_sessions = []
        for session in data_list:
            arr = np.array(session[:min_length])
            # Corta o final para ser perfeitamente divisível
            arr = arr[:(min_length // factor) * factor]
            reduced_arr = np.mean(arr.reshape(-1, factor), axis=1)
            new_sessions.append(reduced_arr)
            
        data_matrix = np.array(new_sessions)
        eff_window = max(1, window // factor)
        x_label = f"{x_label} (x{factor})"
    else:
        data_matrix = np.array([session[:min_length] for session in data_list])
        eff_window = max(1, window)

    # Aplica a Média Móvel com Janela Expansiva
    smoothed_sessions = []
    for row in data_matrix:
        if eff_window > 1:
            cumsum = np.cumsum(row)
            smoothed = np.zeros_like(row, dtype=float)
            
            # Para os primeiros pontos, a média cresce dinamicamente
            smoothed[:eff_window] = cumsum[:eff_window] / np.arange(1, eff_window + 1)
            
            # Para o restante da curva, aplica a janela móvel padrão
            if len(row) > eff_window:
                smoothed[eff_window:] = (cumsum[eff_window:] - cumsum[:-eff_window]) / eff_window
        else:
            smoothed = row
            
        smoothed_sessions.append(smoothed)
        
    smoothed_matrix = np.array(smoothed_sessions)

    # Calcula Média e Desvio Padrão através das curvas já suavizadas
    mean_curve = np.mean(smoothed_matrix, axis=0)
    std_curve = np.std(smoothed_matrix, axis=0)

    x_axis = np.arange(len(mean_curve))

    # Plota a Linha Sólida Principal e a Sombra de Variância
    plt.plot(x_axis, mean_curve, color='red', linewidth=2, label=f'Média Móvel ({window})')
    plt.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, color='red', alpha=0.2, label='Desvio Padrão')

    plt.ylabel("Valor")
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)

    plt.close()

def recursive_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Atualiza um dicionário recursivamente (Deep Merge).
    Ignora chaves que começam com '_' (comentários).
    """
    for k, v in u.items():
        # --- PROTEÇÃO DE COMENTÁRIOS ---
        if k.startswith("_"): continue
        
        if isinstance(v, dict):
            # Garante que o destino também seja dict antes de entrar
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERRO FATAL] Arquivo não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Verifica se é um arquivo já processado (salvo pelo train.py)
    is_processed = ("agent" in raw and "environment" in raw and "definitions" not in raw)
    processed = {}
    
    # FASE 1: EXTRAÇÃO E ACHATAMENTO (FAIL FAST)
    if is_processed:
        processed = raw
        target_env_def = processed["environment"]
        sel_ag = processed["agent"]["type"]
        use_res = processed.get(sel_ag, {}).get("use_reservoir", False) or sel_ag == "ESN"
        
        if use_res and "ESN" not in processed:
            raise KeyError("[ERRO FATAL] 'use_reservoir' ativado, mas a chave 'esn' não existe no arquivo.")
            
    else:
        if "selection" not in raw: 
            raise KeyError("[ERRO FATAL] Arquivo de configuração sem a chave 'selection'.")
        
        # Busca os dados
        processed["training"] = raw.get("training", {})
        sel_ag = raw["selection"]["agent"]
        sel_env = raw["selection"]["environment"]
        
        if "definitions" not in raw:
            raise KeyError("[ERRO FATAL] Arquivo sem a chave 'definitions'.")
        defs = raw["definitions"]

        # Resolve Agentes
        if "agents" not in defs or sel_ag not in defs["agents"]: 
            raise KeyError(f"[ERRO FATAL] Agente '{sel_ag}' não encontrado em definitions -> agents.")
        
        processed["agent"] = {"type": sel_ag}
        processed[sel_ag] = defs["agents"][sel_ag].copy()
        
        # Resolve Ambientes
        if "environments" not in defs or sel_env not in defs["environments"]: 
            raise KeyError(f"[ERRO FATAL] Ambiente '{sel_env}' não encontrado em definitions -> environments.")
        
        target_env_def = defs["environments"][sel_env].copy()

        # Resolve Dependências do Reservatório
        use_res = defs["agents"][sel_ag].get("use_reservoir", False) or sel_ag == "ESN"
        if use_res:
            if "ESN" not in defs["agents"]:
                raise KeyError("[ERRO FATAL] 'use_reservoir' ativado, mas 'esn' não encontrado em definitions -> agents.")
            
            # Traz o esn para a raiz
            processed["ESN"] = defs["agents"]["ESN"].copy()

            # Se o agente principal for esn, traz o modelo de readout para a raiz também
            if sel_ag == "ESN":
                readout_ag = processed["ESN"].get("standard_model", "PPO")
                if readout_ag not in defs["agents"]:
                    raise KeyError(f"[ERRO FATAL] Modelo de readout '{readout_ag}' não encontrado em definitions -> agents.")
                processed[readout_ag] = defs["agents"][readout_ag].copy()

    # FASE 2: CONSTRUÇÃO DO OBSERVATION_SPECS
    templates = target_env_def.get("spec_templates")

    # Verifica se precisa gerar a lista a partir de templates
    if templates is not None and (target_env_def["name"] == "BipedalWalker" or "spec_templates" in target_env_def):
        num_joints = int(target_env_def.get("num_joints", 2))
        mode = target_env_def.get("obs_mode", "geometric")
        
        generated_spec = []
        if target_env_def["name"] == "BipedalWalker":
            s = templates["angle"].copy(); s["name"] = "hull_angle"; generated_spec.append(s)
            for i in range(3):
                s = templates["vector"].copy(); s["name"] = f"hull_vel_{i}"; generated_spec.append(s)
            for leg in range(2):
                for joint in ["hip", "knee"]:
                    s = templates["angle"].copy(); s["name"] = f"pos_{leg}_{joint}"; generated_spec.append(s)
                    s = templates["vector"].copy(); s["name"] = f"vel_{leg}_{joint}"; generated_spec.append(s)
                s = templates["eye"].copy(); s["name"] = f"contact_{leg}"; generated_spec.append(s)
            for i in range(10):
                s = templates["eye"].copy(); s["name"] = f"lidar_{i}"; generated_spec.append(s)
                
        elif mode == "geometric":
            for i in range(num_joints):
                s = templates["angle"].copy(); s["name"] = f"sin_j{i+1}"; generated_spec.append(s)
            for i in range(num_joints):
                s = templates["angle"].copy(); s["name"] = f"cos_j{i+1}"; generated_spec.append(s)
            for axis in ["dx", "dy"]:
                s = templates["vector"].copy(); s["name"] = f"vec_{axis}"; generated_spec.append(s)
                
        elif mode == "physical":
            for i in range(num_joints):
                s = templates["angle"].copy(); s["name"] = f"sin_j{i+1}"; generated_spec.append(s)
            for i in range(num_joints):
                s = templates["angle"].copy(); s["name"] = f"cos_j{i+1}"; generated_spec.append(s)
            for eye in ["L", "R"]:
                s = templates["eye"].copy(); s["name"] = f"eye_{eye}_dist"; generated_spec.append(s)
        
        target_env_def["observation_spec"] = generated_spec

    # Salva o tamanho original para os logs de arquitetura
    target_env_def["orig_input_dim"] = len(target_env_def.get("observation_spec", []))

    # FASE 3: SOBRESCRITA DO RESERVATÓRIO
    if use_res:
        # Puxa direto da raiz processada
        res_conf = processed["ESN"]
        res_size = int(res_conf["reservoir_size"])

        # Identifica o consumidor dos dados para a quantização
        actual_consumer = res_conf.get("standard_model", "PPO") if sel_ag == "ESN" else sel_ag
        res_bits = 1 if actual_consumer == "AHAH" else 16
        
        target_env_def["observation_spec"] = [
            {"name": f"res_{i}", "type": "linear", "min": -1.0, "max": 1.0, "bits": res_bits}
            for i in range(res_size)
        ]
        # Remove os templates originais para não poluir o JSON
        target_env_def.pop("spec_templates", None)

    # Finalização
    if not is_processed:
        processed["environment"] = target_env_def

    return processed

def validate_config_rules(config: Dict[str, Any]):
    """
    Verifica inconsistências lógicas e perigosas na configuração.
    Lança erro ou avisa dependendo da gravidade.
    """
    # 1. Extração Segura dos Metadados
    train_conf = config["training"]
    env_conf = config["environment"]
    
    agent_meta = config["agent"]
    agent_type = agent_meta["type"]
    env_name = env_conf["name"]
    
    use_res = config.get(agent_type, {}).get("use_reservoir", False)
    freeze = train_conf.get("freeze", True)

    # Regra 1: DQN/PPO devem congelar na validação
    # Modelos tradicionais precisam de métricas estáveis. Treinar na validação é 'trapaça' ou instável.
    if agent_type in ["DQN", "PPO"] and not freeze:
        raise ValueError(f"[AVISO] '{agent_type}' configurado com 'freeze': false." 
                         "Treinar na validação é inadequado para este modelo.")

    # Regra 2: Aviso se AHAH não usar congelamento.
    if agent_type == "AHAH" and freeze:
        print("[INFO] AHAH configurado com 'freeze': true." 
              "O modelo AHAH pode não funcionar com os pesos congelados.")
        
    # Regra 3: Explosão Combinatorial do DQN no Reacher
    # Se as ações forem combinatoriais e o ambiente tiver muitas juntas, o espaço de ação explode (3^N)
    if agent_type == "DQN" and env_name == "Reacher2D":
        dqn_conf = config.get("dqn", {})
        
        num_joints = int(env_conf.get("num_joints", 2))
        is_independent = dqn_conf.get("independent_joints", False)
        
        if not is_independent and num_joints > 6:
            action_space_size = 3 ** num_joints
            raise ValueError(
                f"[CONFIG ERROR] DQN Combinatorial com {num_joints} juntas gera {action_space_size} saídas. "
                "Ative 'independent_joints': true ou reduza 'num_joints'.")
            
    # Regra 4: Reservatório substitui encoders no AHAH.
    if use_res:
        actual_agent = config.get("definitions", {}).get("agents", {}).get("ESN", {}).get("standard_model", "PPO") if agent_type == "ESN" else agent_type
        if actual_agent == "AHAH":
            ahah_conf = config.get("AHAH", config.get("definitions", {}).get("agents", {}).get("AHAH", {}))
            if ahah_conf.get("use_temporal_encoder", False):
                raise ValueError("[CONFIG ERROR] Reservatório físico e Encoders Temporais não podem ser usados juntos.")
            if ahah_conf.get("encoder_type", "adaptive") == "adaptive":
                raise ValueError("[CONFIG ERROR] O Reservatório físico já realiza a projeção. Para usar o AHAH, mude 'encoder_type' para 'fixed'.")