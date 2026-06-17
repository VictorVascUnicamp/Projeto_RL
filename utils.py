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
    shaping = config["environment"].get("use_reward_shaping", False)
    
    partes_nome = [env, agent, mode]
    partes_nome.append("ShapeT" if shaping else "ShapeF")
    
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
        if is_fixed:
            partes_nome.append("Fixed")
            
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

    is_lsm = agent_type == "LSM"
    use_res = cfg["environment"].get("use_reservoir", False) or is_lsm
    
    if is_lsm:
        # Se for o Agente LSM, mascaramos ele como o seu Readout Padrão (Ex: PPO)
        agent_type = cfg.get("definitions", {}).get("agents", {}).get("LSM", {}).get("standard_model", "PPO")
        cfg["agent"]["type"] = agent_type

    agent = None
    if agent_type == "AHAH":
        from models.AHAH.controller import AHAHController
        agent = AHAHController(config)
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

def plot_learning_curve(data: List[float],
                        save_path: str,
                        window: int = 50,
                        title: str = "Curva de Aprendizado",
                        x_label: str = "Episódio"):
    """
    Plota dados com downsampling inteligente se for muito longo.
    """
    plt.figure(figsize=(10, 6))
    
    # Se tivermos mais de 2000 pontos, reduz para visualização
    if len(data) > 2000:
        factor = len(data) // 1000
        # Reduz pela média de cada bloco
        reduced_data = np.mean(np.array(data[:len(data)//factor*factor]).reshape(-1, factor), axis=1)
        # Ajusta window para a nova escala
        window = max(1, window // factor)
        
        plt.plot(reduced_data, alpha=0.3, color='blue', label=f'Média (Blocos de {factor})', linewidth=1)
        
        if len(reduced_data) >= window:
            moving_avg = np.convolve(reduced_data, np.ones(window)/window, mode='valid')
            x_axis = range(window - 1, len(reduced_data))
            plt.plot(x_axis, moving_avg, color='red', linewidth=2, label=f'Média Móvel ({window})')
            
        plt.xlabel(f"{x_label} (x{factor})")
        
    else:
        # Plot normal
        plt.plot(data, alpha=0.3, color='blue', label='Valor', linewidth=1)
        if len(data) >= window and window > 1:
            moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
            x_axis = range(window - 1, len(data))
            plt.plot(x_axis, moving_avg, color='red', linewidth=2, label=f'Média Móvel ({window})')
        plt.xlabel(x_label)

    plt.ylabel("Valor")
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Verifica se é um arquivo já processado (salvo pelo train.py)
    # Arquivos processados têm 'agent' e 'environment' na raiz e não dependem de 'selection'
    is_processed = ("agent" in raw and "environment" in raw and "definitions" not in raw)

    processed = {}
    
    if is_processed:
        # Caminho Curto: Já está no formato certo, só precisamos regenerar listas dinâmicas
        processed = raw
        target_env_def = processed["environment"]
        sel_ag = processed["agent"]["type"]
        use_res = target_env_def.get("use_reservoir", False) or sel_ag == "LSM"
        
        if use_res:
            if "LSM" not in processed:
                raise KeyError("[ERRO FATAL] 'use_reservoir' ativado, mas a chave 'LSM' não existe no config.")
            res_conf = processed["LSM"]

        defs = {}
        
    else:
        # Caminho Longo: Processa o arquivo raw com 'definitions' e 'selection'
        if "selection" not in raw: 
            raise KeyError("[ERRO FATAL] Arquivo de configuração sem a chave 'selection'.")
        
        processed["training"] = raw.get("training", {})
        sel_ag = raw["selection"]["agent"]
        sel_env = raw["selection"]["environment"]
        defs = raw.get("definitions", {})

        # Resolve Agentes
        if sel_ag not in defs["agents"]: 
            raise KeyError(f"[ERRO FATAL] Agente '{sel_ag}' não encontrado em definitions -> agents.")
        
        processed["agent"] = {"type": sel_ag}
        processed[sel_ag] = defs["agents"][sel_ag].copy()
        
        # Resolve Ambientes
        if sel_env not in defs["environments"]: 
            raise KeyError(f"[ERRO FATAL] Ambiente '{sel_env}' não encontrado em definitions -> environments.")
        
        target_env_def = defs["environments"][sel_env].copy()

        use_res = target_env_def.get("use_reservoir", False) or sel_ag == "LSM"
        if use_res:
            if "LSM" not in defs["agents"]:
                raise KeyError("[ERRO FATAL] 'use_reservoir' ativado, mas 'LSM' não encontrado em definitions -> agents.")
            
            # [A CHAVE DO SUCESSO] Traz o LSM limpo para a raiz do config, acessível a todos
            processed["LSM"] = defs["agents"]["LSM"].copy()
            res_conf = processed["LSM"]

    # --- GERAÇÃO DE SPECS ---
    templates = target_env_def.get("spec_templates")

    # 1. VERIFICAÇÃO DO RESERVATÓRIO 
    if use_res:
        res_conf = defs.get("agents", {}).get("LSM", defs.get("agents", {}).get("Reservoir", {}))
        # Se o arquivo já for processado, o config do reservatório pode estar na raiz
        if not res_conf and is_processed:
            res_conf = processed.get("lsm", processed.get("reservoir", {}))

        res_size = int(res_conf.get("reservoir_size", 100))

        # Identifica o consumidor dos dados do reservatório
        actual_consumer = sel_ag
        if actual_consumer == "LSM":
            actual_consumer = res_conf.get("standard_model", "PPO")
            
        # O modleo AHAH sofre quantização de 1-bit para pular o SAOL e agilizar o treino.
        # Os modelos clássicos recebem alta resolução para manter o sinal analógico.
        res_bits = 1 if actual_consumer == "AHAH" else 16
        
        target_env_def["observation_spec"] = [
            {"name": f"res_{i}", "type": "linear", "min": -1.0, "max": 1.0, "bits": res_bits}
            for i in range(res_size)
        ]
        target_env_def.pop("spec_templates", None) # Ignora sensores geométricos/físicos

    # Verifica se precisa gerar a lista a partir de templates
    if "spec_templates" in target_env_def:
        num_joints = int(target_env_def.get("num_joints", 2))
        mode = target_env_def.get("obs_mode", "geometric")
        
        generated_spec = []
        if target_env_def["name"] == "BipedalWalker":
            # "angle" para posições (-3.14 a 3.14)
            # "vector" para velocidades (-5 a 5)
            # "eye" para LIDAR e contatos com o solo (0 a 1)
            
            #[0] Hull Angle
            s = templates["angle"].copy(); s["name"] = "hull_angle"; generated_spec.append(s)
            # [1, 2, 3] Velocidades do corpo
            for i in range(3):
                s = templates["vector"].copy(); s["name"] = f"hull_vel_{i}"; generated_spec.append(s)
            
            # [4 a 13] Juntas das Pernas
            for leg in range(2):
                for joint in ["hip", "knee"]:
                    s = templates["angle"].copy(); s["name"] = f"pos_{leg}_{joint}"; generated_spec.append(s)
                    s = templates["vector"].copy(); s["name"] = f"vel_{leg}_{joint}"; generated_spec.append(s)
                # Sensor de Contato com o chão (0 a 1)
                s = templates["eye"].copy(); s["name"] = f"contact_{leg}"; generated_spec.append(s)
                
            #[14 a 23] LIDAR (0 a 1)
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
        
        
        # Injeta a lista gerada no dicionário do ambiente
        target_env_def["observation_spec"] = generated_spec

    # Se não for processado, finaliza a montagem. Se for, já atualizamos a ref.
    if not is_processed:
        processed["environment"] = target_env_def

    return processed

def _clean_env_dict(env_def: Dict[str, Any]):
        """Limpa um dicionário de ambiente individual."""
        if "spec_templates" in env_def and "observation_spec" in env_def:
            # 1. Tenta inferir a resolução atual usada (pega do primeiro sensor)
            spec_list = env_def["observation_spec"]
            current_bits = None
            if spec_list and len(spec_list) > 0:
                current_bits = spec_list[0].get("bits")
            
            # 2. Atualiza os templates com a nova resolução (se encontrada)
            if current_bits is not None:
                for key, tmpl in env_def["spec_templates"].items():
                    if isinstance(tmpl, dict):
                        tmpl["bits"] = current_bits
            
            # 3. Remove a lista expandida (poluição)
            del env_def["observation_spec"]
        return env_def

def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepara o objeto de configuração para salvamento.
    
    Lógica Principal:
    Se um ambiente usa 'spec_templates', esta função:
    1. Lê os bits atuais da lista 'observation_spec' (que pode ter sido otimizada).
    2. Atualiza os templates com esses bits.
    3. Remove a lista 'observation_spec' para não poluir o JSON.
    
    Isso garante que o arquivo salvo contenha as regras (templates) atualizadas,
    mas deixe a geração da lista para o load_config em tempo de execução.
    """
    # Trabalha em uma cópia profunda para não quebrar a execução atual
    clean_conf = copy.deepcopy(config)

    # A. Limpa a chave ativa 'environment' (usada no train/tuning)
    if "environment" in clean_conf:
        clean_conf["environment"] = _clean_env_dict(clean_conf["environment"])

    # B. Limpa as definições usadas no config geral
    if "definitions" in clean_conf:
        defs = clean_conf["definitions"]
        
        # Procura em 'environments' (plural)
        if "environments" in defs:
            for env_name, env_data in defs["environments"].items():
                defs["environments"][env_name] = _clean_env_dict(env_data)
        
        # Procura em 'environment' (legacy)
        if "environment" in defs:
            defs["environment"] = _clean_env_dict(defs["environment"])
            
        # Procura ambientes soltos na raiz de definitions
        for key, val in defs.items():
            if isinstance(val, dict) and "spec_templates" in val:
                defs[key] = _clean_env_dict(val)

    return clean_conf

def validate_config_rules(config: Dict[str, Any]):
    """
    Verifica inconsistências lógicas e perigosas na configuração.
    Lança erro ou avisa dependendo da gravidade.
    """
    # 1. Extração Segura dos Metadados
    train_conf = config.get("training", {})
    env_conf = config.get("environment", {})
    
    agent_meta = config.get("agent", {})
    agent_type = agent_meta["type"]
    env_name = env_conf.get("name", "Unknown")
    
    use_res = env_conf.get("use_reservoir", False) or agent_type == "LSM"
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
        
    # Regra 4: AHAH requer Reward Shaping no CartPole
    if agent_type == "AHAH" and env_name == "CartPole":
        if not env_conf.get("use_reward_shaping", False):
            print("[AVISO] AHAH no CartPole geralmente requer 'use_reward_shaping': true para aprender")

            
    # Regra 5: Reservatório substitui encoders no AHAH.
    if use_res:
        actual_agent = config.get("definitions", {}).get("agents", {}).get("LSM", {}).get("standard_model", "PPO") if agent_type == "LSM" else agent_type
        if actual_agent == "AHAH":
            ahah_conf = config.get("AHAH", config.get("definitions", {}).get("agents", {}).get("AHAH", {}))
            if ahah_conf.get("use_temporal_encoder", False):
                raise ValueError("[CONFIG ERROR] Reservatório físico e Encoders Temporais não podem ser usados juntos.")
            if ahah_conf.get("encoder_type", "adaptive") == "adaptive":
                raise ValueError("[CONFIG ERROR] O Reservatório físico já realiza a projeção. Para usar o AHAH, mude 'encoder_type' para 'fixed'.")