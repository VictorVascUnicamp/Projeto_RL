# tuning.py
# Testa multiplas combinações de arquitetura e hiperparametros para maximizar a performance do agente.

# 1. Silencia avisos de pacotes antigos antes de qualquer importação, no nível do Sistema Operacional (Protege processos paralelos)
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pkg_resources"

import os
import sys
import json
import copy
import numpy as np
from numpy.random import SeedSequence
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import time
import multiprocessing
from joblib import Parallel, delayed
import glob
import shutil
import math
import argparse

from envs.factory import make_env
from models.base import BaseAgent
from models.AHAH.controller import AHAHController
from models.Classical.dqn import DQNAgent
from models.Classical.ppo import PPOAgent 
import utils

# --- Configurações Globais ---
SEEDS_PER_TRIAL = 3     # Repetir cada teste no modo contínuo
EVAL_LAST_PERCENT = 0.2 # Usa os últimos 20% dos episódios para calcular o score

# Configuração de Hardware 
TOTAL_CORES = multiprocessing.cpu_count()
# N_JOBS define quantos processos simultâneos. 
N_JOBS = max(1, int(TOTAL_CORES/2))

def get_random_params(agent_type: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Define o Espaço de Busca (Search Space) para cada agente.
    """
    # Busca o agente
    if agent_type not in base_config:
        raise KeyError(f"[ERRO FATAL] Configuração do agente '{agent_type}' ausente da raiz do base_config. Verifique o load_config.")
    agent_conf = base_config[agent_type]

    params = {}
    use_res = base_config.get("environment", {}).get("use_reservoir", False)

    if agent_type == "LSM":
        params = {
            "reservoir_size": int(np.random.choice([30, 50, 80, 100])),
            "spectral_radius": float(np.random.uniform(0.7, 1.2)),
            "leak_rate": float(np.random.uniform(0.05, 0.5)),
            "sparsity": float(np.random.uniform(0.05, 0.3))
        }
    
    elif agent_type == "AHAH":
        fibers = int(2 * np.random.randint(6, 10)) # Fibras pares
        params = {
            "alpha": float(10 ** np.random.uniform(-4, -1)), # Taxa Hebbiana
            "beta": float(10 ** np.random.uniform(-4, -1)),  # Taxa Anti-Hebbiana
            "noise_std": float(10 ** np.random.uniform(-4, -1)),
            "decay": float(10 ** np.random.uniform(-5, -2)),
            "fibers_per_joint": fibers
        }
    
        if agent_conf.get("use_temporal_encoder", False):
            params["temporal_memory"] = int(np.random.randint(2, 5))
            
        if agent_conf.get("encoder_type", "fixed") == "adaptive":
            params["encoder_beta"] = float(10 ** np.random.uniform(-4, -1))
            
        if use_res:
            params["global_resolution"] = 1
        else:
            params["global_resolution"] = 2*int(np.random.randint(3, 6))

            
    elif agent_type == "DQN":
        # Busca de Arquitetura Neural
        arch_options = [[128, 128], [256, 256], [512, 512], [256, 256, 256]]
        arch_idx = np.random.randint(0, len(arch_options))
        params = {
            "learning_rate": float(10 ** np.random.uniform(-5, -3)),
            "gamma": float(np.random.choice([0.95, 0.98, 0.99, 0.995])),
            "batch_size": int(np.random.choice([64, 128, 256])),
            "target_update_frequency": int(np.random.choice([100, 500, 1000])),
            "epsilon_decay_ratio": float(np.random.uniform(0.2, 0.8)),
            "hidden_layers": arch_options[arch_idx]
        }

    elif agent_type == "PPO":
        arch_options = [[64, 64], [128, 128], [256, 256]]
        arch_idx = np.random.randint(0, len(arch_options))
        params = {
            "learning_rate": float(10 ** np.random.uniform(-5, -3)), 
            "gamma": float(np.random.choice([0.95, 0.98, 0.99])),
            "gae_lambda": float(np.random.choice([0.9, 0.95, 0.98])),
            "clip_coef": float(np.random.choice([0.1, 0.2, 0.3])),
            "ent_coef": float(10 ** np.random.uniform(-3, -1)), 
            "update_epochs": int(np.random.choice([4, 10, 20])),
            "batch_size": int(np.random.choice([64, 128, 256])),
            "hidden_layers": arch_options[arch_idx]
        }

    elif agent_type == "Random":
        pass
    else:
        raise ValueError(f"[Error] Modelo desconhecido.")

    return params

def run_single_seed(config: Dict[str, Any], episodes_to_run: int, seed: int) -> Tuple[float, float, Dict[str, Any]]:
    """
    Executa uma simulação isolada (Seed única).
    Retorna o Score (Retorno médio ou Alvos).
    """
    np.random.seed(seed)
    import torch
    torch.set_num_threads(1) # Garante que rode em 1 núcleo apenas
    
    try:
        env = make_env(config)
    except Exception as e:
        print(f"[ERRO FATAL] {e}")
        return -float("inf"), 0.0, {}

    agent_type = config["agent"]["type"]
    agent: Optional[BaseAgent] = None
    
    learning_mode = config["training"]["learning_mode"]
    orig_max_steps = int(config["environment"]["max_steps"])
    raw_episodes = int(config["training"]["episodes"])

    if learning_mode == "continuous":
        decay_budget = int((orig_max_steps * raw_episodes) / 2)
    else:
        decay_budget = episodes_to_run

    agent = utils.create_agent(config, env=env, decay_budget=decay_budget)
    if not agent:
        env.close(); 
        return -float("inf"), 0.0, {}

    scores = [] 
    successes = []
    
    for ep_idx in range(episodes_to_run):
        obs, _ = env.reset(seed=seed+ep_idx)
        agent.reset()
        
        done = False
        ep_score = 0.0
        success = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
            agent.update(obs, action, float(reward), next_obs, done)
            
            # Coleta métrica padronizada
            ep_score += float(info.get("step_score", 0.0))
            
            obs = next_obs
            
            # Decay Contínuo
            if learning_mode == "continuous" and agent_type == "DQN": agent.decay_epsilon()

        # Decay Episódico
        if learning_mode != "continuous" and agent_type == "DQN": agent.decay_epsilon()

        scores.append(ep_score)
        successes.append(success)

    env.close()

    arch_info = agent.get_architecture_info() if agent_type != "Random" else {}
    env.close()
    
    if not scores: return -float("inf"), 0.0, arch_info

    # Pega a média dos últimos episódios (apenas modo episódico) para avaliar a performance final
    cutoff = max(1, int(len(scores) * EVAL_LAST_PERCENT))
    recent_score = float(np.mean(scores[-cutoff:]))
    recent_success = float(np.mean(successes[-cutoff:]))
    
    return recent_score, recent_success, arch_info

def run_trial_job(trial_id: int, params: Dict[str, Any], 
                  base_config: Dict[str, Any], seeds_to_run: int = 1) -> Dict[str, Any]:
    """
    Worker que gerencia um Trial (Teste de um conjunto de parâmetros).
    Pode rodar 1 seed ou várias seeds sequencialmente.
    """
    start_time = time.time()

    # 1. Clona Config e Injeta Parâmetros
    config = json.loads(json.dumps(base_config))
    agent_type = config["agent"]["type"]
    
    # Atualiza a raiz legada se ela existir
    if agent_type in config: 
        config[agent_type].update(params)
    else:
        raise KeyError(f"[ERRO FATAL] Chave do agente '{agent_type}' não existe na raiz do config.")

    if "reservoir_size" in params:
        res_size = int(params["reservoir_size"])
        # Verifica quem é o consumidor para manter os bits corretos
        actual_consumer = config.get("definitions", {}).get("agents", {}).get("LSM", {}).get("standard_model", "PPO") if agent_type == "LSM" else agent_type
        res_bits = 1 if actual_consumer == "AHAH" else 16
        
        # Regenera a especificação de sensores para bater com a nova matriz (50, 80, 120, etc.)
        config["environment"]["observation_spec"] = [
            {"name": f"res_{i}", "type": "linear", "min": -1.0, "max": 1.0, "bits": res_bits}
            for i in range(res_size)
        ]
    
    # Injeta Resolução Global (no Environment)
    if "global_resolution" in params:
        new_bits = params["global_resolution"]
        if "observation_spec" in config["environment"]:
             for sensor in config["environment"]["observation_spec"]: sensor["bits"] = new_bits
    
    # 2. Configura Redução (Reduction Factor)
    train_conf = config["training"]
    factor = int(train_conf.get("reduction_factor", 5))
    learning_mode = train_conf["learning_mode"]

    orig_e = int(train_conf["episodes"])
    orig_t = int(train_conf.get("target_count", 100))

    # Reduções
    config["training"]["target_count"] = max(1, orig_t // factor)
    red_ep = max(5, orig_e // factor)
    config["training"]["episodes"] = red_ep

    if learning_mode == "continuous":
        episodes_call = 1
        # O número de passos é gerenciado pelo factory.
    else:
        episodes_call = red_ep

    # 3. Execução das Seeds
    scores = []
    successes = []

    # Usamos o ID do trial e o tempo para criar entropia
    # spawn(N) cria N sementes filhas estatisticamente independentes
    entropy_source = int(time.time()) + trial_id * 10000
    ss = SeedSequence(entropy_source)
    # Gera 'seeds_to_run' sementes inteiras independentes
    child_seeds = ss.generate_state(seeds_to_run) 
    
    # 1. Declaramos o resultado como falho por padrão, para evitar erros
    resultado = {
        "trial_id": trial_id, 
        "score": -float("inf"), 
        "params": params, 
        "status": "failed",
        "success": 0.0
    }

    try:
        arch_info = {}
        
        for i, current_seed in enumerate(child_seeds):
            sc, suc, arch_info = run_single_seed(config, episodes_call, seed=int(current_seed))
            scores.append(sc)
            successes.append(suc)
            
        raw_score = float(np.mean(scores))
        raw_success = float(np.mean(successes))
        duration = time.time() - start_time

        # --- Penalização por Complexidade ---
        total_params = arch_info.get("total_parameters", 1)
        penalty_lambda = 0.03

        if raw_score == -float("inf"):
            penalized_score = -float("inf")
        else:
            penalty = abs(raw_score) * penalty_lambda * math.log10(max(1, total_params))
            penalized_score = raw_score - penalty
        
        # 2. Se deu certo, atualizamos o dicionário
        resultado = {
            "trial_id": trial_id,
            "score": penalized_score,   
            "raw_score": raw_score, 
            "success": raw_success,
            "params": params,
            "status": "success",
            "duration_seconds": round(duration, 2),
            "architecture_info": arch_info 
        }
        
        print(f"[TRIAL {trial_id} OK] Score Final: {penalized_score:.2f} (Bruto: {raw_score:.2f}) | Tempo: {duration:.1f}s")
        
        # Salva o arquivo temporário
        trial_file = os.path.join("temp_trials", f"temp_trial_{trial_id}.json")
        with open(trial_file, "w") as f:
            json.dump(utils.clean_for_json(resultado), f)
            
    except Exception as e:
        print(f"[TRIAL {trial_id} FAIL] {e}")

    return resultado

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    # [NOVO]: Flag autorizada para exportar o delta para a raiz
    parser.add_argument("--export_root", action="store_true", help="Salva o delta na raiz do projeto")
    args = parser.parse_args()

    base_config = utils.load_config("config.json")

    # --- VALIDAÇÃO DE REGRAS ---
    utils.validate_config_rules(base_config)

    agent_type = base_config["agent"]["type"]
    env_name = base_config["environment"]["name"]
    
    if agent_type == "Random": return

    # Define o número de testes baseado no nível de conhecimento prévio do modelo
    if agent_type == "AHAH":
        num_trials = 40  
    else:
        num_trials = 20  # Modelos clássicos já têm hiperparâmetros conhecidos na literatura

    learning_mode = base_config["training"]["learning_mode"]
    run_type_name = f"tuning_{learning_mode}"

     # --- INJEÇÃO DE DEPENDÊNCIAS (TRANSFER HPO) ---
    is_lsm = (agent_type == "LSM")
    use_res = base_config.get("environment", {}).get("use_reservoir", False) or is_lsm
    target_dependency = None

    if is_lsm:
        # Tuning do LSM busca a otimização do PPO (Readout)
        target_dependency = base_config.get("definitions", {}).get("agents", {}).get("LSM", {}).get("standard_model", "PPO")
    elif use_res:
        # Tuning do AHAH com reservatório busca as matrizes otimizadas do LSM
        target_dependency = "LSM"

    if target_dependency:
        print(f"\n[TRANSFER HPO] Buscando hiperparâmetros de '{target_dependency}' em '{env_name}' ({learning_mode})...")
        # Hierarquia: results / env / target_agent / tuning_mode / ** / best_params_delta.json
        search_pattern = os.path.join("results", env_name, target_dependency, run_type_name, "**", "best_params_delta.json")
        encontrados = glob.glob(search_pattern, recursive=True)
        
        if encontrados:
            arquivo_mais_recente = max(encontrados, key=os.path.getmtime)
            print(f"Injetando parâmetros de: {arquivo_mais_recente}")
            try:
                with open(arquivo_mais_recente, "r", encoding="utf-8") as f:
                    delta_dep = json.load(f)
                
                # Usa a função recursive_update para injetar com segurança apenas o agente alvo
                if "definitions" in delta_dep and "agents" in delta_dep["definitions"]:
                    if target_dependency in delta_dep["definitions"]["agents"]:
                        dep_params = delta_dep["definitions"]["agents"][target_dependency]
                        # Cria um dicionário filtro para não puxar lixo acidental do arquivo delta
                        filtro_injecao = {
                            "definitions": {
                                "agents": {
                                    target_dependency: dep_params
                                }
                            }
                        }
                        base_config = utils.recursive_update(base_config, filtro_injecao)
                        # Se o HPO injetou um Reservatório novo, a lista de sensores base DEVE ser atualizada!
                        if target_dependency == "LSM" and "reservoir_size" in dep_params:
                            res_size = int(dep_params["reservoir_size"])
                            res_bits = 1 if agent_type == "AHAH" else 16
                            base_config["environment"]["observation_spec"] = [
                                {"name": f"res_{i}", "type": "linear", "min": -1.0, "max": 1.0, "bits": res_bits}
                                for i in range(res_size)
                            ]
            except Exception as e:
                print(f"[AVISO] Falha ao injetar dependência: {e}")
        else:
            print(f"[AVISO] Delta para '{target_dependency}' não encontrado. Usando padrões do config.json.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from utils import create_results_folder
    
    tuning_dir = create_results_folder("results", base_config, run_type_name)
    
    # Cronômetro Global
    tuning_start_time = datetime.now()
    
    print(f"--- Tuning: {agent_type} em {env_name} ({learning_mode}) ---")
    print(f"Hardware: {TOTAL_CORES} Cores. Workers: {N_JOBS}")

    # Gera todos os parâmetros para cada teste
    all_trials = [get_random_params(agent_type, base_config) for _ in range(num_trials)]

    # --- LÓGICA DE RESUME TUNING ---
    os.makedirs("temp_trials", exist_ok=True)
    done_files = glob.glob(os.path.join("temp_trials", "temp_trial_*.json"))
    history =[]
    done_trials = set()
    
    # 1. Carrega os M testes que já foram concluídos antes da queda de energia
    for arq in done_files:
        try:
            with open(arq, "r") as f:
                res = json.load(f)
                history.append(res)
                done_trials.add(res["trial_id"])
        except Exception as e: 
            raise RuntimeError(f"[ERRO FATAL] O arquivo temporário {arq} está corrompido. Erro: {e}")

    # 2. Filtra os N - M testes que faltam
    remain_trials = []

    for i, params in enumerate(all_trials):
        if i not in done_trials: remain_trials.append((i, params))

    print(f"[RESUME] Foram encontrados {len(done_trials)} testes prontos. Faltam {len(remain_trials)} testes.")
    
    # ESTRATÉGIA DE PARALELISMO
    if learning_mode == "episodic":
        # Paraleliza TRIALS, cada teste roda 1 seed
        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(run_trial_job)(i, params, base_config, seeds_to_run=1) # Atraza a execução
            for i, params in remain_trials
        )
    else:
        # Paraleliza TRIALS também, mas cada trial roda N seeds SERIALMENTE.
        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(run_trial_job)(i, params, base_config, seeds_to_run=SEEDS_PER_TRIAL)
            for i, params in remain_trials
        )

    for res in results:
        if res and res["status"] == "success":
            history.append(res)

    best_score = -float("inf")
    best_params_dict = {}
    best_params = {}
    best_arch_info = {}

    for res in history:
        if res["score"] > best_score:
            best_score = res["score"]
            best_params_dict = res
            best_params = res["params"]
            best_arch_info = res.get("architecture_info", {})

    # Limpeza de HD: Apaga os arquivos temporários após terminar
    shutil.rmtree("temp_trials", ignore_errors=True)
    
    if not best_params:
        print("Nenhum trial teve sucesso.")
        sys.exit(1)
        
    tuning_end_time = datetime.now()
    duration = tuning_end_time - tuning_start_time
    total_seconds = duration.total_seconds()

    print(f"*** MELHOR RESULTADO: {best_score:.2f} ***")

    # Relatório
    report = {
        "agent": agent_type,
        "architecture_info": best_arch_info,
        "env": env_name,
        "mode": learning_mode,
        "seeds_per_trial": SEEDS_PER_TRIAL if learning_mode == "continuous" else 1,
        "sessions_run": num_trials,  
        "total_duration": round(total_seconds, 2),
        "best_score": float(best_score),
        "best_raw_score": float(best_params_dict.get("raw_score", best_score)), 
        "best_success": float(best_params_dict.get("success", 0.0)),
        "best_params": best_params
    }
    
    # Salva o relatório
    with open(os.path.join(tuning_dir, "tuning_report.json"), "w") as f:
        json.dump(utils.clean_for_json(report), f, indent=2, ensure_ascii=False)

    # Salva o Histórico
    with open(os.path.join(tuning_dir, "tuning_history.json"), "w") as f:
        json.dump(utils.clean_for_json({"trials": history}), f, indent=2, ensure_ascii=False)

    # 5. Geração e Salvamento do Delta

    # Usa a cópia do que já está na RAM (com as dependências LSM injetadas)
    best_config_snapshot = copy.deepcopy(base_config)
    if agent_type in best_config_snapshot:
        best_config_snapshot[agent_type].update(best_params)

    # Atualiza os hiperparâmetros diretamente na estrutura de definições
    if agent_type not in best_config_snapshot.get("definitions", {}).get("agents", {}):
        if "definitions" not in best_config_snapshot: 
            best_config_snapshot["definitions"] = {"agents": {}}
        best_config_snapshot["definitions"]["agents"][agent_type] = {}
        
    best_config_snapshot["definitions"]["agents"][agent_type].update(best_params)

    # Limpa a explosão de sensores
    clean_snapshot = utils.sanitize_config(best_config_snapshot)

    # Injeta a resolução apenas se ela existir e não houver reservatório
    if "global_resolution" in best_params and not use_res:
        new_bits = best_params["global_resolution"]
        if "observation_spec" in clean_snapshot.get("environment", {}):
            for sensor in clean_snapshot["environment"]["observation_spec"]:
                sensor["bits"] = new_bits


    delta_file = {
        "_metadata": {
            "timestamp": timestamp,
            "score": best_score,
            "note": "Use apply_delta.py para injetar esses dados no config.json"
        },
        "selection": {"agent": agent_type, "environment": env_name},
        "training": clean_snapshot.get("training", {}),
        "definitions": {
            "agents": {
                agent_type: clean_snapshot.get("definitions", {}).get("agents", {}).get(agent_type, {})
            },
            "environments": {
                # O sanitize_config reprojeta os dados em forma de template.
                env_name: clean_snapshot.get("environment", {})
            }
        }
    }

    delta_path = os.path.join(tuning_dir, "best_params_delta.json")
    with open(delta_path, "w", encoding="utf-8") as f:
        json.dump(utils.clean_for_json(delta_file), f, indent=2, ensure_ascii=False)

    # Espelha para o test_configs apenas se o Orquestrador pedir
    if "--export_root" in sys.argv:
        caminho_correio = os.path.join("test_configs", "latest_delta.json")
        with open(caminho_correio, "w", encoding="utf-8") as f:
            json.dump(utils.clean_for_json(delta_file), f, indent=2, ensure_ascii=False)

    print(f"\nDelta salvo em: {delta_path}")

if __name__ == "__main__":
    main()