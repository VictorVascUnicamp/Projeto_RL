# evaluate.py

# 1. Silencia avisos de pacotes antigos antes de qualquer importação, no nível do Sistema Operacional (Protege processos paralelos)
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pkg_resources"

import json
import time
import numpy as np
from numpy.random import SeedSequence
import argparse
import multiprocessing
from gymnasium.wrappers import RecordVideo
from typing import Any, Dict
from joblib import Parallel, delayed
import traceback

# Imports do Projeto
from envs.factory import make_env
from models.base import BaseAgent
from utils import load_config, clean_for_json, create_agent

# Configuração de Hardware
# Detecta núcleos e reserva alguns para o sistema
TOTAL_CORES = multiprocessing.cpu_count()
N_WORKERS = max(1, int(TOTAL_CORES / 2))

# --- FUNÇÕES AUXILIARES (HELPERS) ---

def _run_local_loop(env, agent: BaseAgent, episodes, seed_start=0, freeze=True, video_limit=None):
    """
    Loop serial para execução local (vídeo ou episódico rápido).
    Suporta lógica de Freeze e Limite de Passos para vídeo.
    """
    rets = []
    raw_rets = []
    succ = []
    lens = []
    
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_start + i)
        agent.reset()
        
        done = False
        success = 0
        ep_score = 0
        steps = 0
        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
            # Lógica de Aprendizado (Se não estiver congelado)
            if not freeze:
                agent.update(obs, action, float(reward), next_obs, done)
            
            # Coleta de métricas
            ep_score += float(info.get("step_score", 0.0))
            if info.get("is_success", 0) > 0:
                success += int(info["is_success"])

            obs = next_obs
            steps += 1
            
            # Trava de segurança para vídeos longos (Modo Contínuo)
            if video_limit and steps >= video_limit:
                break
                
        rets.append(ep_score)
        succ.append(success)
        lens.append(steps)
        
    return rets, raw_rets, succ, lens

def run_eval_session(config: Dict, agent_type: str, model_path: str, res_path: str, episodes: int, seed: int) -> Dict[str, Any]:
    """
    Função Worker (Executada em Paralelo).
    Roda uma bateria de episódios para coletar estatísticas (sem vídeo).
    Recria ambiente e agente do zero para garantir isolamento total de memória.
    """
    # Trava threads numéricas internas para evitar conflito com o paralelismo de processos
    import torch
    torch.set_num_threads(1)

    # Força o clone paralelo a ignorar avisos antes de instanciar o ambiente
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    
    try:
        # 1. Cria Ambiente
        env = make_env(config, reservoir_load_path=res_path)
        
        # 2. Carrega Agente
        agent = create_agent(config, env=env, model_path=model_path, is_eval=True)
        
        if not agent: return {}

        # 3. Configurações de Execução
        train_conf = config.get("training", {})
        freeze = train_conf.get("freeze", True) # Se freeze, congela os pesos

        returns = []
        success = []
        lengths = []
        
        # 4. Loop de Episódios
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            
            # Reseta memória de curto prazo
            agent.reset()
            
            done = False
            ep_score = 0
            succ = 0
            steps = 0
            
            while not done:
                
                action = agent.get_action(obs)
                
                next_obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                
                # Se freeze=False o modelo aprende
                if not freeze:
                    agent.update(obs, action, float(reward), next_obs, done)

                # Coleta de métricas
                ep_score += float(info.get("step_score", 0.0))
                if info.get("is_success", 0) > 0:
                    succ += int(info["is_success"])
                
                obs = next_obs
                steps += 1
            
            returns.append(ep_score)
            success.append(succ)
            lengths.append(steps)
            
        env.close()
        return {"scores": returns, "success": success,"lengths": lengths}
        
    except Exception as e:
        print(f"[EVAL ERROR Worker] {e}", flush=True)
        traceback.print_exc()  
        print("-" * 60)
        return {}

def evaluate(result_dir: str):
    """
    Função Principal de Avaliação.
    Gerencia a orquestração (Paralela vs Serial) e a gravação de vídeo.
    """
    start_time = time.time()
    
    # 1. Carrega Configuração
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[EVAL ERROR] Config não encontrado: {result_dir}")
        return

    # load_config regenera as listas de sensores a partir dos templates se necessário
    config = load_config(config_path)
    
    agent_type = config["agent"]["type"]
    env_name = config["environment"]["name"]
    train_conf = config["training"]
    learning_mode = train_conf["learning_mode"]
    freeze = train_conf.get("freeze", True)
    
    print(f"\n--- Avaliando Modelo ({learning_mode}) ---", flush=True)
    print(f"Path: {result_dir}", flush=True)
    print(f"Freeze: {freeze}", flush=True)
    if not freeze: 
        print(f"[INFO] O modelo ira aprender durante a validação.")
        print(f"Selecione 'Freeze = true' para desabilitar o aprendizado")
    
    ext = "npz" if agent_type == "AHAH" else "pth"
    model_path = os.path.join(result_dir, f"final_model.{ext}")
    res_load_path = os.path.join(result_dir, "reservoir.npz")
    
    # 2. Configura Duração do Teste
    config["training"]["target_count"] = 99999

    # Desativa parada precoce para validar tudo
    if "early_stopping_return" in config["environment"]:
        del config["environment"]["early_stopping_return"] 

    config["training"]["episodes"] = 50 # No contínuo, multiplica os passos.
    if learning_mode == "continuous":
        eval_episodes = 1
    else:
        eval_episodes = 50

    total_scores = []
    total_success = []
    total_lengths = []
    
    # Prepara pasta de vídeo
    video_folder = os.path.join(result_dir, "videos")
    os.makedirs(video_folder, exist_ok=True)

    # Pré_inicialização para o pylance não reclamar
    agent_vid = None
    agent = None

    # 3. Execução

    # Fase 1: Estatística 
    # Rodamos várias sessões em paralelo para ter uma média robusta de desempenho
    print(f"Rodando {N_WORKERS} sessões em paralelo...", flush=True)
    
    # Gerador de seeds robusto
    ss = SeedSequence(int(time.time()))
    child_seeds = ss.generate_state(N_WORKERS)
    
    results = Parallel(n_jobs=N_WORKERS, backend="loky")(
        delayed(run_eval_session)(config, agent_type, model_path, res_load_path, eval_episodes, int(child_seeds[i]))
        for i in range(N_WORKERS)
    )
    
    # Consolida resultados
    for res in results:
        if res and "scores" in res:
            total_scores.extend(res["scores"])
            total_success.extend(res["success"])
            total_lengths.extend(res["lengths"])
        
    # Fase 2: Vídeo
    print("Gerando vídeo de demonstração...", flush=True)
    
    # Cria ambiente local com wrapper de gravação
    env_vid = make_env(config, reservoir_load_path=res_load_path) 
    env_vid = RecordVideo(env_vid, video_folder, episode_trigger=lambda x: True, name_prefix="eval_cont")
    
    # Carrega agente localmente (não via worker)
    agent_vid = create_agent(config, env=env_vid, model_path=model_path, is_eval=True)
    
    if agent_vid:
        # Roda loop local com limite forçado de passos
        n_video = 3 if learning_mode == "continuous" else 6
        _run_local_loop(env_vid, agent_vid, n_video, freeze=freeze, video_limit=2000)
    
    env_vid.close()

    # 4. Cálculo de Métricas Finais
    total_duration = time.time() - start_time
    total_steps_hardware = sum(total_lengths)
    fps = total_steps_hardware / (total_duration + 1e-9)
    
    # Filtra os workers que retornaram dados com sucesso
    valid_results = [r for r in results if r]
    n_sessions = len(valid_results)
    
    # Agregação estatística por SESSÃO
    if learning_mode == "continuous":
        # No contínuo, a vitória é o último valor acumulado de sucesso no único episódio longo
        success_per_worker = [r["success"][-1] for r in valid_results if "success" in r and r["success"]]
    else:
        # No episódico, a vitória é a soma dos sucessos dentro dos 50 episódios rodados
        success_per_worker = [sum(r["success"]) for r in valid_results if "success" in r]

    # Média e Desvio Padrão por Sessão de 50 episódios (ou 1 longo)
    mean_success = float(np.mean(success_per_worker)) if success_per_worker else 0.0
    std_success = float(np.std(success_per_worker)) if success_per_worker else 0.0
    
    # Média do score de cada worker para garantir uma métrica macro justa
    scores_per_worker = [np.mean(r["scores"]) for r in valid_results if "scores" in r]
    mean_score_global = float(np.mean(scores_per_worker)) if scores_per_worker else 0.0
    std_score_global = float(np.std(scores_per_worker)) if scores_per_worker else 0.0

    # Usa o agente criado para o vídeo para extrair a arquitetura real do modelo avaliado
    arch_info = agent_vid.get_architecture_info() if agent_vid and agent_type != "Random" else {}
    
    use_res = config.get(agent_type, {}).get("use_reservoir", False) or agent_type == "LSM"
    if use_res and isinstance(arch_info, dict) and arch_info:
        orig_in = config["environment"].get("orig_input_dim", "?")
        res_size = arch_info.get("input_dim", "?") 
        out_dim = arch_info.get("output_dim", arch_info.get("output_fibers", "?"))
        arch_info["architecture"] = f"({orig_in} -> Reservoir[{res_size}] -> {out_dim})"

    final_stats = {
        "agent": agent_type,
        "env": env_name,
        "architecture_info": arch_info,
        "mode": learning_mode,
        "sessions_run": n_sessions, 
        "episodes_per_session": eval_episodes, 
        "total_duration": round(total_duration, 2),
        "inference_fps": round(fps, 2),
        "mean_steps_per_episode": float(np.mean(total_lengths)) if total_lengths else 0.0, 
        "mean_score_global": mean_score_global,
        "std_score_global": std_score_global,
        "mean_success": mean_success,
        "std_success": std_success
    }
    
    # Salva JSON limpo
    with open(os.path.join(result_dir, "eval_stats.json"), "w") as f:
        json.dump(clean_for_json(final_stats), f, indent=2)
    
    # Salva histórico bruto
    with open(os.path.join(result_dir, "eval_history.json"), "w") as f:
        json.dump({"scores": total_scores, "success": total_success, "lengths": total_lengths}, f)

    print(f"[EVAL DONE] Score: {final_stats.get('mean_score', 0):.2f} | FPS: {final_stats['inference_fps']:.0f}", flush=True)

if __name__ == "__main__":
    # Bloco de Auto-Execução
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=str)
    args = parser.parse_args()
    
    target_dir = args.path
    if not target_dir:
        # Busca automática
        base = "results"
        if os.path.exists(base):
            latest_time = 0
            for root, dirs, files in os.walk(base):
                if "config.json" in files:
                    try:
                        t = os.path.getmtime(root)
                        if t > latest_time:
                            latest_time = t
                            target_dir = root
                    except Exception as e:
                        raise RuntimeError(f"Falha ao ler metadados da pasta {root}: {e}")
    
    if target_dir and os.path.exists(target_dir):
        evaluate(target_dir)
    else:
        print("Diretório não encontrado.")