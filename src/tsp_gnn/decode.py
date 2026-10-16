"""Decodificação: das probabilidades das arestas (saída da GNN) a um tour válido.

Fluxo: edge_heatmap -> greedy_decode -> two_opt (opcional) -> verify_tour -> optimality_gap.
Tudo em NumPy, no CPU (uma instância de cada vez).
"""

import numpy as np
import torch

from tsp_gnn.data import distance_matrix, is_valid_tour, tour_length


def _to_numpy(a):
    """Aceita tensores do PyTorch (também na GPU) ou arrays e devolve um array NumPy."""
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def edge_heatmap(edge_index, probs, n):
    """Matriz simétrica (n, n) de probabilidades + máscara dos pares que estão no grafo.

    Cada par do grafo tem duas previsões (i->j e j->i); usamos a média das duas.
    """
    src, dst = _to_numpy(edge_index)
    heat = np.zeros((n, n))
    heat[src, dst] = _to_numpy(probs)
    in_graph = np.zeros((n, n), dtype=bool)
    in_graph[src, dst] = True
    return (heat + heat.T) / 2, in_graph | in_graph.T


def greedy_decode(heatmap, in_graph, coords):
    """Constrói um tour aceitando arestas por ordem de preferência.

    Ordem das candidatas: primeiro as do grafo (probabilidade decrescente, desempate
    pela mais curta); depois todos os pares fora do grafo (distância crescente), para
    garantir que sai sempre um tour completo.
    Uma aresta (i, j) é aceite se i e j têm grau < 2 e estão em fragmentos diferentes.
    """
    n = len(coords)
    dist = distance_matrix(coords)
    i, j = np.triu_indices(n, k=1)               # todos os pares i < j
    g = in_graph[i, j]

    gi, gj = i[g], j[g]                           # pares do grafo
    order_g = np.lexsort((dist[gi, gj], -heatmap[gi, gj]))  # chave principal: -prob
    oi, oj = i[~g], j[~g]                         # pares fora do grafo
    order_o = np.argsort(dist[oi, oj], kind="stable")
    candidates = np.concatenate([np.stack([gi[order_g], gj[order_g]], axis=1),
                                 np.stack([oi[order_o], oj[order_o]], axis=1)])

    # Union-find: parent[c] aponta para outra cidade do mesmo fragmento; a raiz identifica-o.
    parent = list(range(n))

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]         # compressão de caminho (acelera)
            c = parent[c]
        return c

    degree = np.zeros(n, dtype=int)
    adj = [[] for _ in range(n)]                  # vizinhos de cada cidade no tour
    num_edges = 0
    for a, b in candidates.tolist():
        if degree[a] >= 2 or degree[b] >= 2:
            continue                              # uma cidade só tem 2 vizinhos no tour
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            continue                              # fecharia um subciclo
        parent[root_a] = root_b                   # junta os dois fragmentos
        degree[a] += 1; degree[b] += 1
        adj[a].append(b); adj[b].append(a)
        num_edges += 1
        if num_edges == n - 1:
            break                                 # já é um caminho que passa por todas

    # A n-ésima aresta fecha o ciclo: liga as duas pontas (grau 1) do caminho.
    a, b = np.where(degree == 1)[0]
    adj[a].append(b); adj[b].append(a)

    # Converter as arestas numa ordem de visita, a partir da cidade 0.
    tour, prev, cur = [0], -1, 0
    for _ in range(n - 1):
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return np.array(tour)


def two_opt(coords, tour):
    """Melhora o tour com 2-opt até não haver nenhuma troca que o encurte.

    Trocar as arestas (a,b) e (c,d) por (a,c) e (b,d) equivale a inverter o troço b..c.
    """
    dist = distance_matrix(coords)
    t = np.array(tour)
    n = len(t)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            # j percorre as arestas não adjacentes a (t[i], t[i+1]);
            # com i = 0, a última aresta (t[n-1], t[0]) é adjacente e fica de fora.
            js = np.arange(i + 2, n if i > 0 else n - 1)
            a, b = t[i], t[i + 1]
            c, d = t[js], t[(js + 1) % n]
            gain = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]  # variação do comprimento
            k = np.argmin(gain)
            if gain[k] < -1e-9:                   # tolerância contra erros de arredondamento
                j = js[k]
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                improved = True
    return t                                      # t[0] nunca se move: continua a começar em 0


def verify_tour(tour, n):
    """Lança ValueError a explicar o problema se o tour não visitar cada cidade uma vez."""
    if not is_valid_tour(tour, n):
        tour = np.asarray(tour)
        missing = sorted(set(range(n)) - set(tour.tolist()))
        repeated = sorted({c for c in tour.tolist() if (tour == c).sum() > 1})
        raise ValueError(f"tour inválido: tamanho {len(tour)} (esperado {n}), "
                         f"em falta {missing}, repetidas {repeated}")
    return tour


def optimality_gap(length, opt_len):
    """Gap de otimalidade em %: quanto o tour é mais comprido do que o do LKH."""
    return 100.0 * (length / opt_len - 1.0)


def tour_edges(tour):
    """Conjunto das arestas não dirigidas do tour, como pares (menor, maior)."""
    tour = np.asarray(tour)
    return {(min(a, b), max(a, b)) for a, b in zip(tour.tolist(), np.roll(tour, -1).tolist())}


def decode(data, probs, use_2opt=True):
    """Decodifica um grafo (Data) com as probabilidades das suas arestas.

    Devolve um dicionário com o tour, o seu comprimento e o gap face ao LKH (opt_len).
    """
    coords = _to_numpy(data.x).astype(np.float64)
    n = len(coords)
    heat, in_graph = edge_heatmap(data.edge_index, probs, n)
    tour = greedy_decode(heat, in_graph, coords)
    if use_2opt:
        tour = two_opt(coords, tour)
    verify_tour(tour, n)
    length = tour_length(coords, tour)
    return {"tour": tour, "length": length,
            "gap": optimality_gap(length, float(data.opt_len))}
