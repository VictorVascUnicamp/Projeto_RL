# orchestrator.py

# 1. Silencia avisos de pacotes antigos antes de qualquer importação
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import os
import subprocess
import json
import sys  
import glob 
import shutil
from utils import recursive_update

# --- PASTA DOS TESTES---
pasta_configs = "test_configs"
os.makedirs(pasta_configs, exist_ok=True)

# Busca todos os arquivos terminados em .json dentro da pasta
configs_para_rodar = glob.glob(os.path.join(pasta_configs, "*.json"))

if not configs_para_rodar:
    print(f"[AVISO] Nenhum arquivo .json encontrado na pasta '{pasta_configs}'.")
    sys.exit()

print("=== INICIANDO ORQUESTRADOR DE EXPERIMENTOS ===")

arquivo_log = "experimentos_concluidos.txt"
concluidos =[]

# Lê a memória de concluidos (Segurança de interrupção)
if os.path.exists(arquivo_log):
    with open(arquivo_log, "r") as f:
        concluidos = f.read().splitlines()


def executar_passo(comando, nome_passo, config_atual):
    ''' Função auxiliar que roda um passo, checa erros e gera o laudo.'''

    print(f"[{nome_passo}] Executando...")
    # Roda o processo capturando APENAS os erros vermelhos 
    resultado = subprocess.run(comando, stderr=subprocess.PIPE, text=True)
    
    if resultado.returncode != 0:
        print(f"\n[ERRO CRÍTICO] Falha na etapa: {nome_passo}")
        with open("experimentos_falhos.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"FALHA NO EXPERIMENTO: {config_atual}\n")
            f.write(f"ETAPA DO ERRO: {nome_passo}\n")
            f.write(resultado.stderr + "\n")
            f.write(f"{'='*60}\n")

        return False 
    return True 

for config_base in configs_para_rodar:
    if not os.path.exists(config_base):
        print(f"\n[AVISO] Arquivo {config_base} não encontrado. Pulando...")
        continue
        
    if config_base in concluidos:
        print(f"\n[AVISO] Experimento {config_base} já foi concluído anteriormente. Pulando...")
        continue

    print(f"\n{'='*60}")
    print(f"EXPERIMENTO ATUAL: {config_base}")
    print(f"{'='*60}")
    
    # PASSO 1: Deep Merge do Config
    # Lemos o config oficial e o config de teste
    with open("config.json", "r", encoding="utf-8") as f: config_oficial = json.load(f)
    with open(config_base, "r", encoding="utf-8") as f: config_teste = json.load(f)
    
    # Injetamos as novidades do teste no oficial sem destruir as chaves antigas
    config_oficial = recursive_update(config_oficial, config_teste)
    
    # Salvamos de volta no config.json
    with open("config.json", "w", encoding="utf-8") as f: 
        json.dump(config_oficial, f, indent=2, ensure_ascii=False)
    print("[1/4] Configuração mesclada com sucesso.")
    
    # Se o agente for Random, pulamos Tuning e Delta
    agent_type = config_oficial.get("selection", {}).get("agent", "Unknown")
    if agent_type == "Random":
        print("[2/4] Agente Random detectado. Pulando Tuning.")
        print("[3/4] Agente Random detectado. Pulando Apply Delta.")
    else:
        # PASSO 2: Tuning (Pedindo a exportação para a raiz)
        if not executar_passo([sys.executable, "tuning.py", "--export_root"], "2/4 Tuning", config_base):
            continue
            
        # PASSO 3: Aplicar o Delta
        print("[3/4] Procurando e aplicando melhores parâmetros (Delta)...")
        caminho_delta_temporario = os.path.join("test_configs", "latest_delta.json")
        
        # Passa o endereço exato para o apply_delta.py
        if not executar_passo([sys.executable, "apply_delta.py", caminho_delta_temporario], "3/4 Apply Delta", config_base):
            continue
            
        # [LIMPEZA]: O Orquestrador deleta o delta de comunicação após injetá-lo com sucesso
        if os.path.exists(caminho_delta_temporario):
            os.remove(caminho_delta_temporario)

    if not executar_passo([sys.executable, "train.py"], "4/4 Treinamento", config_base):
        continue

    # Se chegou até aqui sem cair em nenhum 'continue', foi um sucesso
    print(f"\n[SUCESSO] Ciclo completo para {config_base}.")
    with open(arquivo_log, "a") as f:
        f.write(config_base + "\n")
        
    # Esvazia a fila deletando o arquivo JSON que já foi processado com sucesso
    os.remove(config_base)

# === LIMPEZA FINAL ===
# Verifica se a pasta test_configs ficou totalmente vazia
arquivos_restantes = glob.glob(os.path.join(pasta_configs, "*.json"))

if len(arquivos_restantes) == 0:
    print("\n=== TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS COM SUCESSO ===")
    shutil.rmtree(pasta_configs, ignore_errors=True)
    if os.path.exists(arquivo_log):
        os.remove(arquivo_log)
else:
    print(f"\n=== ORQUESTRAÇÃO FINALIZADA COM FALHAS ===")
    print(f"Restaram {len(arquivos_restantes)} configs com erro na pasta '{pasta_configs}'.")
    print("Verifique o arquivo 'experimentos_falhos.txt'.")