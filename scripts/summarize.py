"""Junta os resultados das baselines e de todas as corridas num relatório.

Lê results/baselines.json (+ _per_instance.npz) e results/runs/{gcn,gat,gated}_seed*/
(eval.json, eval_per_instance.npz, history.json) e gera:
    results/tabela.md                 tabelas (gap ± desvio-padrão entre seeds; comparações emparelhadas)
    results/figs/curvas_val.{png,pdf}  gap de validação por época, uma linha por modelo
    results/figs/gaps_por_metodo.{png,pdf}  gaps por método em TSP-20 e TSP-50, com barras de erro
Exemplo:
    python scripts/summarize.py
"""

import argparse
import datetime
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # sem janela: só gravar ficheiros
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

MODELS = {"gcn": "GCN", "gat": "GATv2", "gated": "Gated GCN"}
BASELINES = {"greedy": "Guloso clássico", "christofides": "Christofides"}
DATASETS = {"tsp20_test": "TSP-20", "tsp50_test": "TSP-50"}
TIE_TOL = 1e-6   # diferença de gap (pontos %) abaixo da qual duas instâncias contam como empate
ALPHA = 0.05

# Cores (paleta de referência, validadas para daltonismo). Cada cor tem um só significado:
MODEL_COLORS = {"gcn": "#2a78d6", "gat": "#eb6834", "gated": "#1baf7a"}   # figura das curvas
VARIANT_COLORS = {"greedy": "#6250d6", "greedy+2opt": "#eda100"}          # figura das barras
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# As duas comparações emparelhadas: (método do modelo, método da baseline, descrição)
COMPARISONS = [("greedy+2opt", "greedy+2opt", "modelo + 2-opt vs. guloso + 2-opt"),
               ("greedy", "greedy", "modelo sem 2-opt vs. guloso clássico sem 2-opt")]


# ---------------------------------------------------------------- leitura

def load_runs(runs_dir):
    """Corridas {modelo: [corrida, ...]} ordenadas por seed. Ignora pastas como quick_*."""
    runs = {m: [] for m in MODELS}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        match = re.fullmatch(r"(gcn|gat|gated)_seed(\d+)", os.path.basename(path))
        files = ("eval.json", "history.json", "eval_per_instance.npz")
        if not match or not all(os.path.exists(os.path.join(path, f)) for f in files):
            continue  # nome de outro tipo (ex.: quick_gcn) ou corrida ainda a decorrer
        with open(os.path.join(path, "eval.json")) as f:
            ev = json.load(f)
        with open(os.path.join(path, "history.json")) as f:
            history = json.load(f)
        with np.load(os.path.join(path, "eval_per_instance.npz")) as f:
            per_instance = {k: f[k] for k in f.files}
        runs[match.group(1)].append({"seed": int(match.group(2)), "eval": ev,
                                     "history": history, "per_instance": per_instance})
    for m in runs:
        runs[m].sort(key=lambda r: r["seed"])
    return runs


def check_consistency(runs):
    """Avisa se as corridas de um modelo diferem em algo além da seed. Devolve os commits usados."""
    commits = set()
    for m, rs in runs.items():
        configs = [{k: v for k, v in r["eval"]["meta"]["config"].items()
                    if k not in ("seed", "pos_weight")} for r in rs]
        if any(c != configs[0] for c in configs):
            print(f"AVISO: as corridas de {m} têm configurações diferentes: {configs}")
        for r in rs:
            meta = r["eval"]["meta"]
            commits.add(meta["train_git_commit"] + (" (com alterações por gravar)" if meta["git_dirty"] else ""))
    if len(commits) > 1:
        print(f"AVISO: as corridas vêm de commits diferentes: {sorted(commits)}")
    return sorted(commits)


# ---------------------------------------------------------------- estatística

def mean_std(values):
    """'média ± desvio-padrão' entre seeds (só a média se houver uma seed)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return f"{values.mean():.2f}"
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def paired_compare(model_gaps, base_gaps, tol=TIE_TOL):
    """Compara dois métodos instância a instância (gaps em %; menor é melhor).

    Devolve as percentagens de instâncias em que o modelo é melhor/igual/pior, a média e a
    mediana da diferença (modelo - baseline) e o p-valor do teste de Wilcoxon bilateral
    (os empates exatos são descartados pelo teste; se todas forem empate, p = 1).
    """
    diff = np.asarray(model_gaps) - np.asarray(base_gaps)
    nonzero = np.abs(diff) > tol
    p = wilcoxon(diff[nonzero]).pvalue if nonzero.sum() > 0 else 1.0
    return {"better": 100 * np.mean(diff < -tol), "equal": 100 * np.mean(~nonzero),
            "worse": 100 * np.mean(diff > tol), "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)), "p": float(p)}


# ---------------------------------------------------------------- tabelas

def table_gaps(baselines, runs):
    """Tabela 1: gap médio ± desvio-padrão entre seeds."""
    cols = [(d, m) for d in DATASETS for m in ("greedy", "greedy+2opt")]
    head = "| Método | " + " | ".join(f"{DATASETS[d]} {'gulosa' if m == 'greedy' else '+ 2-opt'}"
                                      for d, m in cols) + " | Seeds | Épocas escolhidas |"
    lines = [head, "|" + "---|" * (len(cols) + 3)]
    for key, name in BASELINES.items():
        cells = [f"{baselines[d][key if m == 'greedy' else key + '+2opt']['gap_mean']:.2f}" for d, m in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " | — (determinístico) | — |")
    for key, name in MODELS.items():
        rs = runs[key]
        if not rs:
            continue
        cells = [mean_std([r["eval"][d][m]["gap_mean"] for r in rs]) for d, m in cols]
        seeds = ", ".join(str(r["seed"]) for r in rs)
        epochs = ", ".join(str(r["eval"]["meta"]["best_epoch"]) for r in rs)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {seeds} | {epochs} |")
    return "\n".join(lines)


def table_paired(base_per_instance, runs, threshold):
    """Tabela 2: comparações emparelhadas por instância (uma por seed, depois agregadas)."""
    lines = ["| Comparação | Conjunto | Modelo | Melhor | Igual | Pior | Dif. média (pp) "
             "| Dif. mediana (pp) | p máx. (seeds) | Conclusão |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for model_method, base_method, desc in COMPARISONS:
        for d in DATASETS:
            base = base_per_instance[f"{d}__{base_method}"]
            for key, name in MODELS.items():
                if not runs[key]:
                    continue
                res = [paired_compare(r["per_instance"][f"{d}__{model_method}"], base) for r in runs[key]]
                avg = {k: np.mean([x[k] for x in res]) for k in res[0]}
                p_max = max(x["p"] for x in res)
                same_sign = len({np.sign(x["median_diff"]) for x in res}) == 1
                # Direção pela mediana; se a mediana for 0 (muitos empates), pela média.
                direction = avg["median_diff"] if avg["median_diff"] != 0 else avg["mean_diff"]
                if p_max < threshold and same_sign:
                    verdict = "**melhor**" if direction < 0 else "**pior**"
                else:
                    verdict = "n.s."  # não significativo
                lines.append(f"| {desc} | {DATASETS[d]} | {name} | {avg['better']:.1f}% | {avg['equal']:.1f}% "
                             f"| {avg['worse']:.1f}% | {avg['mean_diff']:+.2f} | {avg['median_diff']:+.2f} "
                             f"| {p_max:.1e} | {verdict} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- figuras

def style_axes(ax):
    """Eixos discretos: sem caixa em cima/direita, grelha horizontal clara."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, figs_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figs_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_curves(runs, figs_dir):
    """Gap de validação por época: média das seeds (linha), mín–máx (faixa), épocas escolhidas (pontos)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ends = []
    for key, name in MODELS.items():
        rs = runs[key]
        if not rs:
            continue
        gaps = np.array([[h["val_gap_greedy"] for h in r["history"]] for r in rs])  # (seeds, épocas)
        epochs = np.arange(1, gaps.shape[1] + 1)
        color = MODEL_COLORS[key]
        ax.fill_between(epochs, gaps.min(0), gaps.max(0), color=color, alpha=0.15, linewidth=0)
        ax.plot(epochs, gaps.mean(0), color=color, linewidth=2, label=f"{name} (n={len(rs)})")
        best = [(r["eval"]["meta"]["best_epoch"], r["eval"]["meta"]["best_val_gap"]) for r in rs]
        ax.scatter(*zip(*best), s=36, color=color, edgecolors="white", linewidths=1.5, zorder=3)
        ends.append([gaps.mean(0)[-1], name])

    # Nomes na ponta das linhas (texto em tinta escura), afastados se estiverem muito próximos.
    ends.sort()
    min_sep = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_sep)
    for y, name in ends:
        ax.text(epochs[-1] + 0.4, y, name, va="center", color=INK, fontsize=9)

    style_axes(ax)
    ax.set_xlim(0.5, epochs[-1] + 2.8)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Época", color=INK)
    ax.set_ylabel("Gap de validação, gulosa sem 2-opt (%)", color=INK)
    ax.set_title("Curvas de treino (TSP-20): média das seeds, faixa = mín–máx, ● = época escolhida",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.88, 1.0))
    save(fig, figs_dir, "curvas_val")


def plot_bars(baselines, runs, figs_dir):
    """Gap médio por método, gulosa vs. + 2-opt, em TSP-20 e TSP-50 (mesmo eixo y)."""
    methods = [(k, n, None) for k, n in BASELINES.items()] + [(k, n, runs[k]) for k, n in MODELS.items() if runs[k]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    width = 0.38
    x = np.arange(len(methods))
    for ax, d in zip(axes, DATASETS):
        for v_idx, variant in enumerate(("greedy", "greedy+2opt")):
            means, errs = [], []
            for key, _, rs in methods:
                if rs is None:  # baseline: determinística, sem barra de erro
                    means.append(baselines[d][key if variant == "greedy" else key + "+2opt"]["gap_mean"])
                    errs.append(0.0)
                else:
                    vals = [r["eval"][d][variant]["gap_mean"] for r in rs]
                    means.append(np.mean(vals))
                    errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            offset = (v_idx - 0.5) * width
            ax.bar(x + offset, means, width, color=VARIANT_COLORS[variant], edgecolor="white", linewidth=2,
                   label="Gulosa" if variant == "greedy" else "Gulosa + 2-opt")
            # Barras de erro só onde há variação entre seeds (as baselines não têm).
            has_err = np.array(errs) > 0
            ax.errorbar((x + offset)[has_err], np.array(means)[has_err], yerr=np.array(errs)[has_err],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        style_axes(ax)
        ax.set_xticks(x, [n.replace(" ", "\n", 1) for _, n, _ in methods], fontsize=9)  # nomes em 2 linhas
        ax.set_title(DATASETS[d], color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Gap face ao LKH (%)", color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")  # no TSP-20 há espaço livre em cima
    fig.suptitle("Gap médio por método (teste). Barras de erro: desvio-padrão entre seeds; "
                 "baselines são determinísticas", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figs_dir, "gaps_por_metodo")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Relatório: tabelas e figuras dos resultados.")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    with open(os.path.join(args.results, "baselines.json")) as f:
        baselines = json.load(f)
    with np.load(os.path.join(args.results, "baselines_per_instance.npz")) as f:
        base_per_instance = {k: f[k] for k in f.files}
    runs = load_runs(os.path.join(args.results, "runs"))
    commits = check_consistency(runs)

    n_tests = len(COMPARISONS) * len(DATASETS) * sum(1 for m in MODELS if runs[m])
    threshold = ALPHA / n_tests  # correção de Bonferroni para os testes todos

    report = "\n".join([
        "# Resultados da fase 1",
        "",
        f"Gerado por `scripts/summarize.py` em {datetime.date.today().isoformat()}. "
        f"Commits do treino: {', '.join(commits)}; baselines: {baselines['meta']['git_commit']}.",
        "",
        "## Tabela 1 — gap médio face ao LKH (%)",
        "",
        "Modelos: média ± desvio-padrão **entre seeds** (variação de corrida para corrida). "
        "Baselines: determinísticas. Treino em TSP-20; TSP-50 mede a generalização.",
        "",
        table_gaps(baselines, runs),
        "",
        "## Tabela 2 — comparações emparelhadas por instância",
        "",
        f"Para cada seed, compara-se o gap de cada instância com o da baseline; a tabela mostra as médias "
        f"entre seeds. *Igual* = diferença ≤ {TIE_TOL:g} pp. Dif. = modelo − baseline (negativo = modelo melhor). "
        f"Teste de Wilcoxon bilateral (empates descartados); *p máx.* é o maior p-valor entre as seeds. "
        f"Conclusão significativa só se p máx. < {threshold:.2e} (Bonferroni: {ALPHA}/{n_tests} testes) "
        f"e o sinal da mediana for o mesmo em todas as seeds.",
        "",
        table_paired(base_per_instance, runs, threshold),
        "",
        "## Figuras",
        "",
        "- `figs/curvas_val.png` — gap de validação por época",
        "- `figs/gaps_por_metodo.png` — gap por método em TSP-20 e TSP-50",
        "",
    ])
    with open(os.path.join(args.results, "tabela.md"), "w") as f:
        f.write(report)

    figs_dir = os.path.join(args.results, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    plot_curves(runs, figs_dir)
    plot_bars(baselines, runs, figs_dir)

    print(report)
    print(f"corridas: { {m: [r['seed'] for r in rs] for m, rs in runs.items()} }")
    print(f"gravado: {args.results}/tabela.md e {figs_dir}/curvas_val.(png|pdf), gaps_por_metodo.(png|pdf)")


if __name__ == "__main__":
    main()
