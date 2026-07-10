# train.py

# 1. Silencia avisos de pacotes antigos antes de qualquer importação, no nível do Sistema Operacional (Protege processos paralelos)
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pkg_resources"

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
    checkpoints_dir: Optional[str] = None,
    global_stop_event: Optional[Any] = None
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

    # Força o clone paralelo a ignorar avisos antes de instanciar o ambiente
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    
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
        # Se outro worker já venceu, encerra o treino.
        if global_stop_event and global_stop_event.is_set():
            break

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
                # Aborta o episódio se outro worker declarou vitória
                if global_stop_event and global_stop_event.is_set():
                    done = True
                    break

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

                    # Parada Global para o Modo Contínuo
                    if learning_mode == "continuous" and target_count is not None:
                        if success >= target_count:
                            print(f"\n[Sessão {session_id}] [VITÓRIA CONTÍNUA] Alvo atingido ({success}/{target_count}).", flush=True)
                            if global_stop_event: global_stop_event.set()
                            done = True
                            break
                
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
                window = min(len(session_metrics["ep_scores"]), ma_val)
                if window > 0:
                    current_ma = np.mean(session_metrics["ep_scores"][-window:])
                    
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
                            success_in_window = session_metrics["success"][-1] - session_metrics["success"][-(window_x + 1)]
                            if success_in_window >= target_count:
                                print(f"\n[Sessão {session_id}] [EARLY STOPPING] Alvo atingido ({success_in_window} sucessos nos últimos {window_x} eps)." 
                                    f"Parando o treino no episódio {ep+1}!", flush=True)
                                # Encerra os outros workers
                                if global_stop_event: global_stop_event.set()
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

    use_res = config.get(agent_type, {}).get("use_reservoir", False) or agent_type == "ESN"
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
        "save_interval_steps": save_interval_steps,
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

    # 2. Execução
    # Rodamos várias sessões em paralelo e pegamos a melhor.
    num_sessions = N_JOBS
    print(f"--- Treino Contínuo: {num_sessions} sessões paralelas ---", flush=True)

    # Cria um comunicador global para as sessões pararem juntas
    manager = multiprocessing.Manager()
    global_stop_event = manager.Event()

    # Gera N sementes filhas independentes
    child_seeds = ss.generate_state(num_sessions)
    results = Parallel(n_jobs=num_sessions, backend="loky")(
        delayed(train_session)(i, config, int(child_seeds[i]), checkpoints_dir, global_stop_event)
        for i in range(num_sessions)
    )

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
    
    # 4. Estatísticas Globais

    # Estatísticas Globais de Hardware
    total_steps_hardware = sum([sum(r["metrics"].get("lengths",[])) for r in valid_results])
    fps_global = total_steps_hardware / (global_total_time + 1e-9)
    arch_info = best_res.get("arch_info", {})

    # Coleta dos dados das sessões
    all_scores = [np.mean(r["metrics"]["ep_scores"]) for r in valid_results if r["metrics"].get("ep_scores")]
    all_lengths = [np.mean(r["metrics"]["lengths"]) for r in valid_results if r["metrics"].get("lengths")]
    all_successes = [r["metrics"]["success"][-1] for r in valid_results if r["metrics"].get("success")]

    # Média transversal das 6 sessões 
    mean_score_global = float(np.mean(all_scores)) if all_scores else 0.0
    mean_success_global = float(np.mean(all_successes)) if all_successes else 0.0
    mean_steps_global = float(np.mean(all_lengths)) if all_lengths else 0.0

    # Desempenho exato da sessão cujo modelo foi salvo no disco
    best_idx = valid_results.index(best_res)
    best_session_score = float(all_scores[best_idx]) if all_scores else 0.0
    best_session_success = float(all_successes[best_idx]) if all_successes else 0.0
    best_session_steps = float(all_lengths[best_idx]) if all_lengths else 0.0

    final_stats = {
        "agent": agent_type,
        "env": env_name,
        "architecture_info": arch_info,
        "mode": learning_mode,
        "sessions_run": len(valid_results),
        "total_duration": round(global_total_time, 2), 
        "fps_global": round(fps_global, 2),
        "total_episodes": len(best_res["metrics"].get("lengths", [])),
        "mean_steps_global": mean_steps_global, 
        "best_session_steps": best_session_steps, 
        "mean_score_global": mean_score_global, 
        "best_session_score": best_session_score, 
        "mean_success_global": mean_success_global,
        "best_session_success": best_session_success
    }

    utils.save_stats(os.path.join(results_dir, "train_stats.json"), final_stats)
    
    # Salva histórico detalhado
    full_history = [r["metrics"] for r in valid_results]
    utils.save_stats(os.path.join(results_dir, "train_history_full.json"), {"sessions": full_history})
    
    # --- 4. GRÁFICOS ---
    ma_val = int(train_conf.get("ma_value", 50))
    
    # Recalcula o intervalo de salvamento para o texto do eixo X
    save_interval_steps = valid_results[0].get("save_interval_steps", 100)
    eixo_x_nome = f"Passos (x{save_interval_steps})" if learning_mode == "continuous" else "Episódios"
    
    # Extrai as curvas de todas as sessões em listas conjuntas
    all_scores_curves = []
    all_success_curves = []
    
    for r in valid_results:
        mets = r["metrics"]
        s_curve = mets.get("window_scores") if len(mets.get("window_scores", [])) > 0 else mets.get("ep_scores", [])
        succ_curve = mets.get("success", [])
        
        if len(s_curve) > 0: all_scores_curves.append(s_curve)
        if len(succ_curve) > 0: all_success_curves.append(succ_curve)

    if len(all_scores_curves) > 0 and len(all_scores_curves[0]) > 0:

        # 4.1 Gráfico de Scores
        win_score = ma_val if train_conf.get("ma_mode") == "constant" else max(1, len(all_scores_curves[0]) // ma_val)
        utils.plot_learning_curve(
            all_scores_curves, 
            os.path.join(results_dir, "curve_scores.png"), 
            window=win_score, 
            title="Curva de Scores",
            x_label=eixo_x_nome
        )
        
        # 4.2 Gráfico de Scores iniciais (Early Scores) 
        total_points = len(all_scores_curves[0])
        early_limit = max(10, total_points // 10) 

        early_scores = [s[:early_limit] for s in all_scores_curves]
        if len(early_scores[0]) > 0:
            win_early = ma_val if train_conf.get("ma_mode") == "constant" else max(1, len(early_scores[0]) // ma_val)
            # Trava de segurança para modo 'constant' não engolir o gráfico curto
            win_early = min(win_early, max(1, len(early_scores[0]) // 2))
            
            utils.plot_learning_curve(
                early_scores, 
                os.path.join(results_dir, "curve_early_scores.png"), 
                window=win_early, 
                title="Curva de Scores (Fase Inicial)",
                x_label=eixo_x_nome
            )

    # 4.3 Gráfico de Sucessos
    if len(all_success_curves) > 0 and len(all_success_curves[0]) > 0:
        win_succ_base = ma_val if train_conf.get("ma_mode") == "constant" else max(1, len(all_success_curves[0]) // ma_val)
        win_succ = max(1, win_succ_base // 2)
        utils.plot_learning_curve(
            all_success_curves, 
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