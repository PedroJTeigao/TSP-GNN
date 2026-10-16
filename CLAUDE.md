# Projeto: GNN para otimização combinatória (TSP)

## Contexto
- Projeto na CMU com o Prof. Soummya. Fase 1: GNN supervisionada para o TSP.
- Pipeline: LKH (elkai) gera os rótulos -> GNN prevê as arestas do tour ->
  decodificação gulosa + 2-opt -> gap de otimalidade vs. LKH.
- Modelos a comparar: GCN, GATv2 e Gated GCN residual (Joshi et al. 2019).

## Ambiente
- Ativar sempre com `source env.sh`. Instalar pacotes só dentro de `.venv`.
- Python 3.12, PyTorch 2.14 (CUDA), PyTorch Geometric 2.8, elkai, networkx.

## Regras (servidor partilhado com outros trabalhos)
- Trabalhar SÓ dentro desta pasta.
- Perguntar antes de apagar qualquer ficheiro.
- Código, comentários, README e mensagens de commit em inglês; as explicações para mim e o guia de estudo em português.
- Nunca mexer em ../agentic-hpga, ../phase2, ../phase2_agent_memory,
  /scratch/pcanaste/venv nem no Ollama.
- Nunca parar processos: o Ollama usa a GPU para outro trabalho.
- GPU T4 partilhada: ~7 GB livres. Manter cada treino abaixo de 6 GB.
- Treinos longos com `nice -n 10`.
- Não editar ~/.bashrc nem ~/.claude.
- Dados em data/ e resultados em results/ (fora do git).
- Avançar passo a passo e explicar o que cada passo faz: estou a aprender.

## Repositório
- GitHub: https://github.com/PedroJTeigao/TSP-GNN (branch main)
- O /scratch não tem backup: fazer commit e push no fim de cada sessão.
