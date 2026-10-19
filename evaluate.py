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
import glob
import copy

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
        np.random.seed(seed) # Garante semente aleatória para o Random
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

            learning_mode = train_conf["learning_mode"]
            
            if learning_mode == "continuous":
                # Normaliza a pontuação total do teste contínuo para blocos de 1.000 passos, igual os gráficos.
                score_padronizado = (ep_score / (steps + 1e-9)) * 1000.0
                returns.append(float(score_padronizado))
            else:
                returns.append(float(ep_score))

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
    Função Principal de Avaliação Estatística Inter-Sementes.
    Avalia cada um dos modelos treinados nas sessões independentes sob a mesma bateria de testes.
    Executa o benchmark oficial congelado e, se configurado, o protocolo adaptativo.
    """
    start_time = time.time()
    
    # 1. Carrega Configuração Oficial
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERRO FATAL] Arquivo config.json não encontrado em: {result_dir}")

    config = load_config(config_path)
    agent_type = config["agent"]["type"]
    env_name = config["environment"]["name"]
    train_conf = config["training"]
    learning_mode = train_conf["learning_mode"]
    
    # Verifica a preferência de adaptação no config original
    frozen = train_conf.get("freeze", True)
    
    # Configura episódios de teste
    config["training"]["episodes"] = 50
    eval_episodes = 1 if learning_mode == "continuous" else 50
    ext = "npz" if agent_type == "AHAH" else "pth"
    res_load_path = os.path.join(result_dir, "reservoir.npz")


    # Localiza todos os modelos salvos das sessões de treino
    if agent_type == "Random":
        # Random não salva pesos em disco. Cria workers virtuais sem arquivos.
        eval_workers = N_WORKERS
        session_model_files = [None] * eval_workers
    else:
        session_model_files = sorted(glob.glob(os.path.join(result_dir, f"model_session_*.{ext}")))
        if not session_model_files:
            # Fallback caso execute em pastas legadas com apenas um modelo
            single_path = os.path.join(result_dir, f"final_model.{ext}")
            session_model_files = [single_path] if os.path.exists(single_path) else []

        if not session_model_files:
            raise FileNotFoundError(
                f"[ERRO FATAL EVAL] Nenhum modelo ('model_session_*.{ext}' ou 'final_model.{ext}') "
                f"foi encontrado em '{result_dir}'. O treinamento pode ter falhado ao salvar os pesos."
            )
    
    n_models = len(session_model_files)
    eval_workers = min(N_WORKERS, n_models) if n_models > 0 else 1

    # Semente base única para teste emparelhado
    EVAL_BASE_SEED = 10001

    def run_benchmark_suite(is_frozen: bool, stats_filename: str, video_subfolder: str):
        """Executa a avaliação paralela dos modelos e gera vídeo demonstrativo sob o regime especificado."""
        regime_name = "TRADICIONAL" if is_frozen else "ADAPTATIVO"
        print(f"\n--- Bateria de Testes ({regime_name}) ---", flush=True)

        cfg_run = copy.deepcopy(config)
        cfg_run["training"]["freeze"] = is_frozen
        cfg_run["training"]["episodes"] = 50

        # Executa em paralelo: worker i avalia o modelo i enfrentando a mesma EVAL_BASE_SEED
        results = Parallel(n_jobs=eval_workers, backend="loky")(
            delayed(run_eval_session)(
                cfg_run, agent_type, 
                session_model_files[i % len(session_model_files)], 
                res_load_path, eval_episodes, 
                (EVAL_BASE_SEED + i) if agent_type == "Random" else EVAL_BASE_SEED
            )
            for i in range(eval_workers)
        )

        valid = [r for r in results if r and "scores" in r]
        if not valid:
            raise RuntimeError(
                f"[ERRO FATAL EVAL] Todas as sessões de avaliação paralelas falharam no regime {regime_name}. "
                f"Verifique os logs de exceção dos workers acima."
            )

        # Métricas agregadas por modelo treinado
        scores_per_model = [np.mean(r["scores"]) for r in valid]
        succ_per_model = [r["success"][-1] if learning_mode == "continuous" else sum(r["success"]) for r in valid]
        lengths_per_model = [np.mean(r["lengths"]) for r in valid]

        # Estatística Inter-Sementes
        stats = {
            "agent": agent_type,
            "env": env_name,
            "mode": learning_mode,
            "frozen_evaluation": is_frozen,
            "models_evaluated": len(valid),
            "episodes_per_model": eval_episodes,
            "paired_eval_seed": EVAL_BASE_SEED,
            "mean_score": float(np.mean(scores_per_model)),
            "std_score": float(np.std(scores_per_model)),
            "mean_success": float(np.mean(succ_per_model)),
            "std_success": float(np.std(succ_per_model)),
            "mean_steps": float(np.mean(lengths_per_model)),
            "scores_per_model": scores_per_model
        }

        with open(os.path.join(result_dir, stats_filename), "w") as f:
            json.dump(clean_for_json(stats), f, indent=2)

        # Gravação de Vídeo demonstrativo do regime
        vid_dir = os.path.join(result_dir, "videos", video_subfolder)
        os.makedirs(vid_dir, exist_ok=True)
        print(f"Gravando vídeo ({regime_name}) em: {vid_dir}...", flush=True)

        best_model_path = None if agent_type == "Random" else os.path.join(result_dir, f"final_model.{ext}")
        env_vid = make_env(cfg_run, reservoir_load_path=res_load_path)
        env_vid = RecordVideo(env_vid, vid_dir, episode_trigger=lambda x: True, name_prefix=f"eval_{video_subfolder}")
        agent_vid = create_agent(cfg_run, env=env_vid, model_path=best_model_path, is_eval=True)
        
        if agent_vid:
            n_video = 3 if learning_mode == "continuous" else 6
            _run_local_loop(env_vid, agent_vid, n_video, seed_start=EVAL_BASE_SEED, freeze=is_frozen, video_limit=2000)
        env_vid.close()

        print(f"[{regime_name} OK] Score: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
        return stats

    # PROTOCOLO 1 (Oficial): Sempre executa com pesos congelados
    main_stats = run_benchmark_suite(is_frozen=True, stats_filename="eval_stats.json", video_subfolder="frozen")

    # PROTOCOLO 2 (Adicional): Executa se o config permitir adaptação contínua (freeze: false)
    if not frozen:
        run_benchmark_suite(is_frozen=False, stats_filename="eval_adaptive_stats.json", video_subfolder="adaptive")

    total_duration = time.time() - start_time
    print(f"\n[EVAL COMPLETO] Tempo total de validação: {total_duration:.1f}s", flush=True)

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