# apply_delta.py
import json
import os
import argparse
import shutil
import glob
import sys
from utils import recursive_update

def main():
    parser = argparse.ArgumentParser(description="Atualiza o config.json com um resultado especificado.")
    parser.add_argument("delta_path", nargs="?", type=str, help="Caminho ou Pasta do arquivo best_params_delta.json")
    args = parser.parse_args()

    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"[ERRO FATAL] {config_path} não encontrado.")
        sys.exit(1)

    target_path = args.delta_path
    
    # --- LÓGICA DE BUSCA FLEXÍVEL (As 3 regras) ---
    if not target_path:
        # Regra 1: Nada foi passado. Busca o mais recente na pasta results.
        print("Nenhum arquivo especificado. Buscando o delta mais recente na pasta results/...")
        arquivos_delta = glob.glob('results/**/best_params_delta.json', recursive=True)
        if not arquivos_delta:
            print("[ERRO FATAL] Nenhum arquivo best_params_delta.json encontrado na pasta results/.")
            sys.exit(1)
        target_path = max(arquivos_delta, key=os.path.getmtime)
        
    elif os.path.isdir(target_path):
        # Regra 2: Passou um diretório. Busca o delta mais recente dentro dele.
        print(f"Buscando delta dentro de: {target_path}")
        arquivos_delta = glob.glob(os.path.join(target_path, '**/best_params_delta.json'), recursive=True)
        if not arquivos_delta:
            # Tenta busca direta caso a pasta informada seja exatamente a pasta final do tuning
            path_direto = os.path.join(target_path, 'best_params_delta.json')
            if os.path.exists(path_direto):
                target_path = path_direto
            else:
                print(f"[ERRO FATAL] Nenhum best_params_delta.json encontrado no diretório {target_path}.")
                sys.exit(1)
        else:
            target_path = max(arquivos_delta, key=os.path.getmtime)
            
    else:
        # Regra 3: Passou o caminho exato do arquivo.
        pass

    if not os.path.exists(target_path):
        print(f"[ERRO FATAL] Arquivo Delta não encontrado no caminho: '{target_path}'.")
        sys.exit(1)

    print(f"\n--- Aplicando Delta ---")
    print(f"Fonte do Delta: {target_path}")

    # 1. Carrega Arquivos
    with open(target_path, "r", encoding="utf-8") as f:
        delta = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Backup de Segurança
    shutil.copy(config_path, "config_backup.json")
    
    delta_agent = delta.get("selection", {}).get("agent", "Unknown")
    delta_env = delta.get("selection", {}).get("environment", "Unknown")
    print(f"Target: Agente {delta_agent} em {delta_env}")

    # 4. Updates (Deep Merge)
    config["selection"] = recursive_update(config.get("selection", {}), delta.get("selection", {}))
    
    if "training" in delta:
        config["training"] = recursive_update(config.get("training", {}), delta["training"])

    if "definitions" in delta:
        config["definitions"] = recursive_update(config.get("definitions", {}), delta["definitions"])

    # 5. Salva
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("--- Sucesso! Config.json atualizado no HD. ---")

if __name__ == "__main__":
    main()