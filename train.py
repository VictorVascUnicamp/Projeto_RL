# train.py

# 1. Silencia avisos de pacotes antigos antes de qualquer importação, no nível do Sistema Operacional (Protege processos paralelos)
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pkg_resources"

import os
import json
import time
import numpy as np
from numpy.random import SeedSequence
import argparse
import multiprocessing
from typing import Any, Dict, List, cast, Optional
from collections import defaultdict
import shutil
import traceback

# Joblib: Biblioteca para computação paralela
from joblib import Parallel, delayed

# Imports dos módulos do projeto
from envs.factory import make_env
import utils
from evaluate import evaluate 

# --- Configuração de Hardware ---
# Detecta o número de núcleos do processador.
TOTAL_CORES = multiprocessing.cpu_count()
# Define quantos processos rodarão simultaneamente. Alguns ficam livres para o OS.
N_JOBS = max(1, int(TOTAL_CORES/2))

def train_session(
    session_id: int, 
    config: Dict[str, Any], 
    seed: int, 
    checkpoints_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Função Worker (Trabalhador).
    Executa uma sessão de treinamento completa de forma isolada.
    Pode ser rodada em paralelo (joblib) ou serialmente.
    
    Args:
        session_id: Identificador numérico da sessão (0 é a principal).
        config: Dicionário completo de configurações.
        seed: Semente aleatória para reprodutibilidade desta sessão.
        checkpoints_dir: Onde salvar backups (apenas sessão 0 salva).
    """

    # Restringe bibliotecas numéricas a 1 thread por processo. Necessario para o joblib.
    import torch
    torch.set_num_threads(1)
    
    # 1. Configuração do Ambiente e Seeds
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_env(config)
    except Exception as e:
        print(f"[ERRO Sessão {session_id}] Falha ao criar env: {e}", flush=True)
        traceback.print_exc()
        print("-" * 60)
        return {"status": "failed", "error": "Falha ao criar env: {e}"}

    # Leitura de configurações
    agent_type = config["agent"]["type"]
    train_conf = config["training"]
    learning_mode = train_conf["learning_mode"]
    
    # Definições de duração
    max_steps_env = int(config["environment"]["max_steps"])
    raw_episodes = int(train_conf["episodes"])
    
    if learning_mode == "continuous":
        # Modo Contínuo: 1 episódio. Número de episódios aumenta a duração do episódio.
        episodes = 1
        decay_budget = int(max_steps_env*raw_episodes/2)
    else:
        episodes = raw_episodes
        decay_budget = episodes

    # Define a frequência de salvamento
    save_frequency = train_conf.get("save_frequency", 50)
    save_interval_eps = max(10, int(episodes / save_frequency)) 
    save_interval_steps = max(1000, int(max_steps_env / save_frequency))

    # 2. Instanciação do Agente
    agent = utils.create_agent(config, env=env, decay_budget=decay_budget)
    if not agent:
        env.close()
        return {"status": "failed", "error": "Falha ao criar o agente"}

    # Coletor de métricas locais
    session_metrics = defaultdict(list)
    start_time = time.time()
    
    # Log inicial
    print(f"[Sessão {session_id}] Iniciando treino de {episodes} ciclos...", flush=True)
    
    # Prepara pasta específica desta sessão para evitar colisão de arquivos
    # Só cria se o diretório pai foi fornecido
    my_ckpt_dir = None
    if checkpoints_dir is not None:
        my_ckpt_dir = os.path.join(checkpoints_dir, f"session_{session_id}")
        os.makedirs(my_ckpt_dir, exist_ok=True)

    success = 0

    # Parametros para early stopping e salvamento do melhor modelo
    best_ma_score = -float('inf')
    best_model_state = None
    ma_val = int(train_conf.get("ma_value", 20))
    # Limite para término precoce
    target_count = train_conf.get("target_count", None)
    
    # 3. Loop Principal de Treinamento
    for ep in range(episodes):
        try:
            obs, _ = env.reset(seed=seed + ep)
            
            # Limpa memória de curto prazo
            agent.reset()

            # Inicializamos as variáveis
            done = False
            ep_score = 0.0   
            window_score = 0.0  
            steps = 0
            
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                
                agent.update(obs, action, float(reward), next_obs, done)
                
                # Coleta as pontuações do ambiente
                step_score = float(info.get("step_score", 0.0))
                raw = float(info.get("raw_reward", reward))

                ep_score += step_score
                window_score += step_score
                
                # Coleta os sucessos
                if info.get("is_success", 0) > 0:
                    success += int(info["is_success"])
                
                # --- LÓGICA DO MODO CONTÍNUO ---
                if learning_mode == "continuous":
                    # Salva os dados a cada N passos para gerar o gráfico
                    if steps > 0 and steps % save_interval_steps == 0:
                        session_metrics["window_scores"].append(window_score)
                        session_metrics["success"].append(success)
                        window_score = 0.0 # Zera para o próximo periodo
                    
                    agent.decay_epsilon()

                    # Checkpoint
                    if my_ckpt_dir and steps > 0 and steps % save_interval_steps == 0:
                        ext = "npz" if agent_type == "AHAH" else "pth"
                        if agent_type != "Random":
                            agent.save(os.path.join(my_ckpt_dir, f"latest_model.{ext}"))
                
                obs = next_obs
                steps += 1

            # --- LÓGICA DO FIM DO EPISÓDIO ---
            agent.decay_epsilon()
        
            if learning_mode != "continuous":
                # No modo episódico, a "janela" é o episódio inteiro.
                session_metrics["window_scores"].append(ep_score)
                session_metrics["success"].append(success)
                
            # Salva o Total para o JSON, diferente dos gráficos.
            session_metrics["ep_scores"].append(ep_score)
            session_metrics["lengths"].append(steps)
            
            # --- [NOVO] RASTREAMENTO DO MELHOR MODELO & PARADA PRECOCE ---
            if agent_type != "Random":
                # Calcula a média móvel com base nos retornos episódicos reais
                window = min(len(session_metrics["ep_score"]), ma_val)
                if window > 0:
                    current_ma = np.mean(session_metrics["ep_score"][-window:])
                    
                    # Salva os pesos na RAM se bater o recorde histórico
                    if current_ma > best_ma_score:
                        best_ma_score = current_ma
                        best_model_state = agent.get_state()
                        
                    # Parada precoce baseada na taxa de sucesso com janela de tolerância
                    if target_count is not None:
                        window_x = int(target_count * 1.1) # Tolerância de 10%
                        # Verifica se temos episódios suficientes + 1 para subtrair com segurança
                        if len(session_metrics["success"]) >= window_x+1:
                            # Calcula as vitórias conquistadas dentro da janela móvel
                            success_in_window = session_metrics["success"][-1] - session_metrics["success"][-(window_x+1)]
                            if success_in_window >= target_count:
                                print(f"\n[Sessão {session_id}] [EARLY STOPPING] Alvo de performance atingido ({success_in_window} sucessos nos últimos {window_x} episódios)." 
                                    f"Parando o treino no episódio {ep+1}!", flush=True)
                                break  

            # Feedback em Tempo Real (Apenas sessão principal e modo episódico)
            if learning_mode == "episodic" and session_id == 0:
                if (ep + 1) % save_interval_eps == 0:
                    avg_score = np.mean(session_metrics["ep_scores"][-10:])
                    print(f"[Ep {ep+1}/{episodes}] Score: {avg_score:.2f} | Passos: {steps}", flush=True)

            # CHECKPOINT POR EPISÓDIO
            if my_ckpt_dir and (ep + 1) % save_interval_eps == 0:
                ext = "npz" if agent_type == "AHAH" else "pth"
                if agent_type != "Random":
                    save_name = os.path.join(my_ckpt_dir, f"latest_model.{ext}")
                    agent.save(save_name)

        except Exception as e:
            print(f"\n[Sessão {session_id}] [ERRO FATAL] Ambiente colapsou no episódio {ep+1} ({type(e).__name__}: {e}).")
            print("Salvando o melhor modelo histórico e encerrando sessão para não perder os dados")
            break  # Interrompe o loop de episódios e segue para salvar os resultados

    env.close()
    total_time = time.time() - start_time
    
    # 4. Extração do Estado do Melhor Modelo
    final_model_state = best_model_state if best_model_state is not None else agent.get_state()

    # Extrai todos os dados da arquitetura
    arch_info = agent.get_architecture_info() if agent_type != "Random" else {}

    use_res = config["environment"].get("use_reservoir", False) or agent_type == "ESN"
    if use_res and isinstance(arch_info, dict) and arch_info:
        orig_in = config["environment"].get("orig_input_dim", "?")
        res_size = arch_info.get("input_dim", "?") 
        out_dim = arch_info.get("output_dim", arch_info.get("output_fibers", "?"))
        
        arch_info["architecture"] = f"({orig_in} -> Reservoir[{res_size}] -> {out_dim})"

    # Extração do Reservatório
    res_state = None
    curr = env
    while hasattr(curr, "env"):
        if hasattr(curr, "get_reservoir_state"):
            res_state = curr.get_reservoir_state() # type: ignore
            break
        curr = curr.env # type: ignore

    # Retorna pacote de resultados
    return {
        "status": "success",
        "metrics": dict(session_metrics),
        "total_time": total_time,
        "model_state": final_model_state, 
        "reservoir_state": res_state,
        "arch_info": arch_info
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    # 1. Carregamento e Preparação
    config = utils.load_config(args.config)
    
    # --- VALIDAÇÃO DE REGRAS ---
    utils.validate_config_rules(config)

    agent_type = config["agent"]["type"]
    env_name = config["environment"]["name"]
    train_conf = config["training"]
    learning_mode = train_conf["learning_mode"]
    
    # Cria pastas de resultado organizadas por data
    run_type_name = f"train_{learning_mode}"
    results_dir = utils.create_results_folder("results", config, run_type_name)
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Salva cópia do config na nova pasta
    shutil.copy(args.config, os.path.join(results_dir, "config.json"))

    # CRONÔMETRO GLOBAL
    global_start_time = time.time()

    # GERAÇÃO DE SEMENTES ROBUSTAS
    ss = SeedSequence(int(time.time()))

    # 2. Execução (Estratégia Híbrida)
    if learning_mode == "continuous":
        # MODO CONTÍNUO:
        # Rodamos várias sessões em paralelo e pegamos a melhor.
        num_sessions = N_JOBS
        print(f"--- Treino Contínuo: {num_sessions} sessões paralelas ---", flush=True)

        # Gera N sementes filhas independentes
        child_seeds = ss.generate_state(num_sessions)
        results = Parallel(n_jobs=num_sessions, backend="loky")(
            delayed(train_session)(i, config, int(child_seeds[i]), checkpoints_dir)
            for i in range(num_sessions)
        )
    else:
        # MODO EPISÓDICO:
        # Rodamos 1 sessão sequencial na thread principal.
        num_sessions = 1
        print(f"--- Treino Episódico: Sessão Única ---", flush=True)
        
        # Gera 1 semente única
        single_seed = int(ss.generate_state(1)[0])
        res = train_session(0, config, single_seed, checkpoints_dir)
        results = [res]

    global_total_time = time.time() - global_start_time

    # 3. Consolidação dos Resultados
    valid_results = [r for r in results if r is not None and r["status"] == "success"]
    if not valid_results:
        print("[ERRO] Todas as sessões falharam.", flush=True)
        return

    # === ESCOLHA DO CAMPEÃO ===

    # Função interna para dar uma nota a cada sessão concluída
    def avaliar_sessao(sessao):
        lista_de_scores = sessao["metrics"].get("ep_scores",[])
        if len(lista_de_scores) > 0:
            return float(np.mean(lista_de_scores))
        else:
            # Se a lista estiver vazia (erro), retorna "menos infinito" para nunca ser escolhida
            return -float('inf')

    # Prepara display e métricas do campeão
    best_res = max(valid_results, key=avaliar_sessao)
    avg_score = avaliar_sessao(best_res)
    # Prepara display para o terminal
    total_success = best_res["metrics"]["success"][-1] if best_res["metrics"].get("success") else 0
    score_display = f"Score: {avg_score:.2f} | Sucessos Totais: {total_success}"

    print(f"--- Melhor Resultado: {score_display} ---", flush=True)
    
    # Salva Modelo Final (Campeão)
    if agent_type != "Random" and best_res["model_state"]:
        ext = "npz" if agent_type == "AHAH" else "pth"
        save_path = os.path.join(results_dir, f"final_model.{ext}")
        if agent_type == "AHAH":
            with open(save_path, "wb") as f: f.write(best_res["model_state"])
        else:
            import torch
            torch.save(best_res["model_state"], save_path)

    # Salva Matrizes do Reservatório para a Validação [NOVO]
    if best_res.get("reservoir_state"):
        np.savez(os.path.join(results_dir, "reservoir.npz"), **best_res["reservoir_state"])
    
    # Calcula Estatísticas Globais
    best_metrics = best_res["metrics"]
    best_scores = best_metrics.get("ep_scores", [])  
    best_successes = best_metrics.get("success",[])
    best_lengths = best_metrics.get("lengths",[])
    
    # FPS Global (Esse precisa usar o tempo de parede total e a soma de todos os passos de todas as sessões para medir eficiência de hardware)
    total_steps_hardware = sum([sum(r["metrics"].get("lengths",[])) for r in valid_results])
    fps_global = total_steps_hardware / (global_total_time + 1e-9)

    arch_info = best_res.get("arch_info", {})

    # Lógica de Separação de Métricas (Contínuo vs Episódico)
    if learning_mode == "continuous":
        # Extrai listas de 1 elemento de todas as sessões
        all_scores = [r["metrics"]["ep_scores"][0] for r in valid_results if r["metrics"].get("ep_scores")]
        all_successes = [r["metrics"]["success"][-1] for r in valid_results if r["metrics"].get("success")]
        all_lengths = [r["metrics"]["lengths"][0] for r in valid_results if r["metrics"].get("lengths")]
        
        mean_score = float(np.mean(all_scores)) if all_scores else 0.0
        max_score = float(best_metrics["ep_scores"][0]) if best_metrics.get("ep_scores") else 0.0
        
        mean_success = float(np.mean(all_successes)) if all_successes else 0.0
        max_success = float(best_metrics["success"][-1]) if best_metrics.get("success") else 0.0
        
        mean_steps = int(np.mean(all_lengths)) if all_lengths else 0
        min_steps = int(best_metrics["lengths"][0]) if best_metrics.get("lengths") else 0
    else:
        # Modo episódico focado na melhor sessão
        mean_score = float(np.mean(best_scores)) if best_scores else 0.0
        max_score = float(np.max(best_scores)) if best_scores else 0.0
        
        # No episódico, o sucesso é o acúmulo total
        total_succ = int(best_successes[-1]) if best_successes else 0
        mean_success = total_succ
        max_success = total_succ
        
        mean_steps = int(np.mean(best_lengths)) if best_lengths else 0
        min_steps = mean_steps

    final_stats = {
        "agent": agent_type,
        "env": env_name,
        "architecture_info": arch_info,
        "mode": learning_mode,
        "sessions_run": len(valid_results),
        "total_duration": round(global_total_time, 2), 
        "fps_global": round(fps_global, 2),
        "total_episodes": len(best_lengths),
        "mean_steps": mean_steps, 
        "min_steps": min_steps, 
        "mean_score": mean_score, 
        "max_score": max_score, 
        "mean_success": mean_success,
        "max_success": max_success
    }

    utils.save_stats(os.path.join(results_dir, "train_stats.json"), final_stats)
    
    # Salva histórico detalhado
    full_history = [r["metrics"] for r in valid_results]
    utils.save_stats(os.path.join(results_dir, "train_history_full.json"), {"sessions": full_history})
    
    # --- 4. GRÁFICOS (Baseados na melhor sessão) ---
    metrics = best_res["metrics"]
    graphic_scores = metrics.get("window_scores",[])
    ma_val = int(train_conf.get("ma_value", 20))
    
    # Recalcula o intervalo de salvamento para o texto do eixo X
    max_steps_env = int(config["environment"]["max_steps"])
    save_freq = train_conf.get("save_frequency", 50)
    save_interval_steps = max(100, int(max_steps_env / save_freq))
    
    eixo_x_nome = f"Passos (x{save_interval_steps})" if learning_mode == "continuous" else "Episódios"
    
    # 4.1 Gráfico do Retorno Bruto (Raw Reward)
    if len(graphic_scores) > 0:
        # Cria uma janela para o raw reward
        win_score = ma_val if train_conf.get("ma_mode") == "constant" else max(1, len(graphic_scores) // ma_val)
        
        utils.plot_learning_curve(
            graphic_scores, 
            os.path.join(results_dir, "curve_scores.png"), 
            window=win_score, 
            title="Curva de Scores",
            x_label=eixo_x_nome
        )

    # 4.2 Gráfico da Métrica Discreta (Sucessos)
    if "success" in metrics and len(metrics["success"]) > 0:
        # Calcula a janela especificamente para os sucessos
        win_succ_base = ma_val if train_conf.get("ma_mode") == "constant" else max(1, len(metrics["success"]) // ma_val)
        win_succ = max(1, win_succ_base // 2)  # Janela um pouco menor para ver patamares nítidos
        
        utils.plot_learning_curve(
            metrics["success"], 
            os.path.join(results_dir, "curve_success.png"), 
            window=win_succ, 
            title="Curva de Sucessos",
            x_label=eixo_x_nome
        )

    # Limpa a pasta de checkpoints após o sucesso para poupar HD
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir, ignore_errors=True)
        
    # 5. Avaliação Final
    print("--- Avaliação Automática ---", flush=True)
    evaluate(results_dir)

if __name__ == "__main__":
    main()