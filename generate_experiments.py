# generate_experiments.py
import os
import json
import itertools
import copy

# 1. Cria a pasta isolada
pasta_destino = "test_configs"
os.makedirs(pasta_destino, exist_ok=True)

# 2. Carrega o Molde
with open("config.json", "r", encoding="utf-8") as f:
    base_config = json.load(f)

# --- A LISTA DE GRADES DE TESTE INTELIGENTE ---
grades_de_teste = [
    # # GRADE 1: Baselines
    # {
    #     "env": ["CartPole", "Reacher2D"],
    #     "agent": ["AHAH","Random","PPO", "DQN"],
    #     "mode": ["episodic", "continuous"],
    #     "bias": ["B"],            
    #     "temporal": [False],      
    #     "encoder": ["adaptive"],
    #     "use_reservoir": [False]   
    # },
    # # GRADE 2: AHAH Ablation (Testando Bias e Encoders)
    # {
    #     "env": ["CartPole", "Reacher2D"],
    #     "agent": ["AHAH"],
    #     "mode": ["episodic", "continuous"],
    #     "bias": ["A", "C"],       
    #     "temporal": [False],
    #     "encoder": ["adaptive"],
    #     "use_reservoir": [False]
    # },
    # {
    #     "env": ["CartPole", "Reacher2D"],
    #     "agent": ["AHAH"],
    #     "mode": ["episodic", "continuous"],
    #     "bias": ["B"],
    #     "temporal": [False],
    #     "encoder": ["fixed"],
    #     "use_reservoir": [False]      
    # },
    # # GRADE 3: O Teste de Inércia do AHAH (Temporal Encoder)
    # {
    #     "env": ["CartPole"],
    #     "agent": ["AHAH"],
    #     "mode": ["episodic", "continuous"],
    #     "bias": ["B"],
    #     "temporal": [True],
    #     "encoder": ["adaptive"],
    #     "use_reservoir": [False]
    # },
    # # GRADE 4: BipedalWalker
    # {
    #     "env": ["BipedalWalker"],
    #     "agent": ["Random","PPO", "DQN", "AHAH"],
    #     "mode": ["episodic"],
    #     "bias": ["B"],
    #     "temporal": [False],
    #     "encoder": ["adaptive"],
    #     "use_reservoir": [False]
    # },
    # GRADE 5: Otimização do ESN
    {
        "env": ["CartPole", "Reacher2D"],
        "agent": ["ESN"],
        "mode": ["episodic", "continuous"]
    },
    {
        "env": ["BipedalWalker"],
        "agent": ["ESN"],
        "mode": ["episodic"]
    },
    # # GRADE 6: AHAH com Reservatório
    # {
    #     "env": ["CartPole", "Reacher2D"],
    #     "agent": ["AHAH"],
    #     "mode": ["episodic", "continuous"],
    #     "bias": ["B"],
    #     "temporal": [False],
    #     "encoder": ["fixed"],
    #     "use_reservoir": [True]
    # },
    # {
    #     "env": ["BipedalWalker"],
    #     "agent": ["AHAH"],
    #     "mode": ["episodic"],
    #     "bias": ["B"],
    #     "temporal": [False],
    #     "encoder": ["fixed"],
    #     "use_reservoir": [True]
    # }
]

contador = 0

for grade in grades_de_teste:
    envs = grade.get("env", ["CartPole"])
    agents = grade.get("agent", ["AHAH"])
    modes = grade.get("mode", ["episodic"])
    biases = grade.get("bias", [None])
    temporais = grade.get("temporal", [None])
    encoders = grade.get("encoder", [None])
    use_reservoirs = grade.get("use_reservoir", [False])

    for env, agent, mode, bias, temporal, encoder, use_res in itertools.product(envs, agents, modes, biases, temporais, encoders, use_reservoirs):
        cfg = copy.deepcopy(base_config)
        
        # --- APLICAÇÃO GERAL ---
        cfg["selection"]["environment"] = env
        cfg["selection"]["agent"] = agent
        cfg["training"]["learning_mode"] = mode
        
        cfg["training"]["target_count"] = 100
        
        env_dict = cfg["definitions"]["environments"][env]

        if agent not in ["Random", "ESN"]:
            cfg["definitions"]["agents"][agent]["use_reservoir"] = use_res
        
        # Duração do treino  
        if agent == "Random": cfg["training"]["episodes"] = 50
        else: cfg["training"]["episodes"] = 5000 if env == "BipedalWalker" else 1000

        # Monta o nome do arquivo dinamicamente
        nome_partes = [env, agent, mode[:4]]

        if use_res and agent != "ESN":
            nome_partes.append("Res")
        
        # --- APLICAÇÃO ESPECÍFICA DO AHAH ---
        if agent == "AHAH":
            cfg["definitions"]["agents"]["AHAH"]["bias_update_mode"] = bias
            cfg["definitions"]["agents"]["AHAH"]["use_temporal_encoder"] = temporal
            cfg["definitions"]["agents"]["AHAH"]["encoder_type"] = encoder
            cfg["training"]["freeze"] = False

            if use_res:
                temporal = False
                encoder = "fixed"
            
            nome_partes.append(f"Bias{bias}")
            if temporal:
                nome_partes.append("Temp")
            if encoder == "fixed" and not use_res:
                nome_partes.append("Fixed")
        else:
            cfg["training"]["freeze"] = True
            
        # 5. Salva o arquivo final
        nome_arquivo = "exp_" + "_".join(nome_partes) + ".json"
        caminho_final = os.path.join(pasta_destino, nome_arquivo)
        
        with open(caminho_final, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            
        print(f"Gerado: {nome_arquivo}")
        contador += 1

print(f"\n[SUCESSO] {contador} arquivos de configuração criados na pasta '{pasta_destino}'!")