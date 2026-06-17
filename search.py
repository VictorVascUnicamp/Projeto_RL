# search.py
import os
import json
import glob
from typing import List, Dict, Any

# --- ÁREA DE CONFIGURAÇÃO DO USUÁRIO ---

# 1. Defina aqui o que você está procurando
def filter_condition(data: Dict[str, Any]) -> bool:
    """
    Retorna True se o modelo deve aparecer na lista.
    'data' contém chaves misturadas do config e do train_stats.
    Exemplos:
      data['agent_type'], data['env_name'], 
      data['total_time'], data['alpha'], data['num_joints']...
    """
    
    # Exemplos
    cond1 = (data.get("agent_type") == "AHaH")

    cond2 = (data.get("env_name") == "Reacher2D")
    
    cond3 = (data.get("total_duration", 9999) < 60) 

    cond4 = (data.get("bias_update_mode", "C") == "A")
    
    alpha = data.get("alpha", 1.0) 
    cond5 = (alpha < 0.01)
    
    # COMBINE SUAS CONDIÇÕES AQUI:
    return cond1 and cond4
    
# 2. Regra de ordenamento
SORT_KEY = "mean_score" 
REVERSE_SORT = True  # True para maior primeiro, False para menor

# ---------------------------------------

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Achatador de dicionários recursivo.
    Transforma {'AHAH': {'alpha': 0.1}} em {'ahah_alpha': 0.1}
    Também tenta manter chaves curtas se não houver colisão (ex: só 'alpha').
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # repete o processo até chegar na raiz
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            # salva também a chave curta se for única
            # (Isso permite acessar data['alpha'] em vez de data['ahah_alpha'])
            items.append((k, v)) 
    return dict(items)

def load_run_data(folder_path: str) -> Dict[str, Any]:
    """
    Lê config.json e train_stats.json de uma pasta e funde tudo.
    """
    data = {"path": folder_path, "folder_name": os.path.basename(folder_path)}
    
    # Lê Config
    cfg_path = os.path.join(folder_path, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                raw_cfg = json.load(f)
                data.update(flatten_dict(raw_cfg))
        except:
            pass

    # Lê Stats de Treino
    stats_path = os.path.join(folder_path, "train_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                raw_stats = json.load(f)
                data.update(flatten_dict(raw_stats))
        except:
            pass
            
    return data

def main():
    root_dir = "results"
    print(f"Buscando em '{root_dir}'...\n")
    
    found_models = []
    
    # Caminha recursivamente procurando pastas que tenham config.json
    # (os.walk é mais robusto que glob para profundidade variável)
    for current_root, dirs, files in os.walk(root_dir):
        if "config.json" in files and "train_stats.json" in files:
            
            # Carrega dados
            data = load_run_data(current_root)
            
            # Aplica Filtro do Usuário
            if filter_condition(data):
                found_models.append(data)

    if not found_models:
        print("Nenhum modelo encontrado com os critérios atuais.")
        return

    # Ordenação
    found_models.sort(
        key=lambda x: x.get(SORT_KEY, -float('inf')), 
        reverse=REVERSE_SORT
    )

    # Exibição em Tabela Simples
    # Defina aqui quais colunas você quer ver no print
    cols =[
        ("Data/Pasta", "folder_name", 30),
        ("Agente", "type", 10), 
        ("Env", "name", 15),    
        ("Score", "mean_score", 15),         
        ("Sucessos", "mean_successes", 10),         
        ("Tempo(s)", "total_duration", 10), 
        ("Alpha", "alpha", 10), 
        ("Juntas", "num_joints", 8)
    ]

    # Cabeçalho
    header = "".join([f"{c[0]:<{c[2]}} | " for c in cols])
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for m in found_models:
        row_str = ""
        for label, key, width in cols:
            val = m.get(key, "-")
            
            # Formatação de números floats
            if isinstance(val, float):
                val_str = f"{val:.2f}" if abs(val) > 1e-4 else f"{val:.2e}"
            else:
                val_str = str(val)
                
            # Trunca se for muito longo
            if len(val_str) > width:
                val_str = val_str[:width-2] + ".."
                
            row_str += f"{val_str:<{width}} | "
        print(row_str)
    
    print("-" * len(header))
    print(f"Encontrados: {len(found_models)} modelos.")

if __name__ == "__main__":
    main()