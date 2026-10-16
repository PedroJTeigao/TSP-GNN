# Fase 1 explicada — guia de estudo

Este guia explica, em linguagem simples, tudo o que foi feito na fase 1: o problema, os
métodos, as decisões e os resultados. Serve para estudar e para preparar a apresentação.
Em cada secção indica-se **onde está no código** (📁). Os números vêm de `results/tabela.md`
e de `README.md`.

---

## 1. O problema: o que é o TSP e porque é difícil

**O TSP (Travelling Salesman Problem, problema do caixeiro-viajante):** dadas *n* cidades,
encontrar o percurso mais curto que visita cada cidade **exatamente uma vez** e volta ao
início. Esse percurso chama-se **tour**.

No nosso caso as cidades são pontos aleatórios num quadrado [0,1]², e a distância entre duas
cidades é a distância em linha reta (euclidiana).

**Porque é difícil:** o número de tours possíveis explode. Fixando a cidade de partida e
contando cada tour uma só vez (ida = volta), há (n−1)!/2 tours:

| n | nº de tours |
|---|---|
| 7 | 360 |
| 20 | ≈ 6 × 10¹⁶ |
| 50 | ≈ 3 × 10⁶² |

Com 20 cidades, mesmo testando mil milhões de tours por segundo, levaria cerca de 2 anos.
O TSP é **NP-difícil**: não se conhece nenhum algoritmo que encontre sempre o ótimo em
tempo polinomial. Por isso usam-se **heurísticas**: métodos que dão tours bons, rapidamente,
sem garantia de ótimo.

**Um valor de referência útil:** para cidades uniformes em [0,1]², o tour ótimo mede em média
cerca de **3,83** com 20 cidades e **5,70** com 50 (valores medidos na literatura com solvers
exatos). Usámo-los para verificar os nossos dados.

📁 `src/tsp_gnn/data.py` (`generate_instances`, `tour_length`)

---

## 2. Os métodos clássicos

### LKH — o "professor"
**LKH (Lin–Kernighan–Helsgaun)** é uma das melhores heurísticas para o TSP. Parte de um tour e
melhora-o repetidamente, trocando várias arestas de uma vez (movimentos *k-opt*), usando
listas de candidatos inteligentes para não testar trocas inúteis.

- Não garante o ótimo, mas em instâncias de 20–50 cidades chega lá quase sempre.
- **Como confirmámos:** (1) em 5 instâncias de 7 cidades comparámos com a força bruta (todas as
  720 ordens) e coincidiu sempre; (2) as médias dos nossos conjuntos (3,83–3,85 em TSP-20,
  5,68 em TSP-50) batem com os valores ótimos da literatura.
- O LKH só aceita distâncias inteiras, por isso multiplicamos por 10⁶ e arredondamos (erro
  desprezável, ~10⁻⁶ por aresta).
- Na fase 1 o LKH é o **professor**: os seus tours são as "respostas certas" que a GNN aprende
  a imitar, e a referência para medir o gap.

📁 `src/tsp_gnn/data.py` (`solve_lkh`), `scripts/make_dataset.py`

### Christofides — com garantia de 1,5×
1. Constrói a **árvore geradora mínima** (MST): a forma mais barata de ligar todas as cidades sem ciclos.
2. Encontra as cidades com um nº **ímpar** de ligações na árvore.
3. Emparelha essas cidades duas a duas com o custo mínimo (*emparelhamento perfeito mínimo*).
4. Árvore + emparelhamento = todas as cidades com grau par → existe um **circuito euleriano**
   (passa por cada aresta uma vez).
5. Percorre o circuito e salta as cidades já visitadas (atalhos).

**Garantia:** o tour nunca é mais de **1,5 × o ótimo**. Intuição: a árvore custa ≤ ótimo
(tirar uma aresta ao tour ótimo dá uma árvore) e o emparelhamento custa ≤ ½ ótimo; os atalhos
nunca aumentam o custo (desigualdade triangular). Testámos esta garantia em
`tests/test_baselines.py`.

📁 `src/tsp_gnn/baselines.py` (`christofides_tour`)

### Guloso ("greedy edge")
Ordena **todos** os pares de cidades da distância mais curta para a mais longa e vai aceitando
arestas, desde que (1) nenhuma cidade fique com mais de 2 ligações e (2) não se feche um ciclo
antes de passar por todas. Simples e rápido, mas míope: escolhas boas agora podem obrigar a
ligações péssimas no fim.

📁 `src/tsp_gnn/baselines.py` (`greedy_distance_tour`), que reutiliza `decode.greedy_decode`

### 2-opt
Melhoria local: pega em duas arestas do tour, (a,b) e (c,d), e troca-as por (a,c) e (b,d) —
o que equivale a **inverter o troço** entre *b* e *c*. Se o tour ficar mais curto, aceita.
Repete até nenhuma troca ajudar. O resultado é um **ótimo local**: não melhora com nenhuma
troca de 2 arestas, mas pode não ser o ótimo global. Intuitivamente, o 2-opt "desfaz
cruzamentos" (um tour com arestas que se cruzam nunca é ótimo).

📁 `src/tsp_gnn/decode.py` (`two_opt`)

---

## 3. A ideia da fase 1: aprender com o solver

Em vez de desenhar uma heurística à mão, **aprendemos uma a partir de exemplos**:

1. Geramos milhares de instâncias e resolvemo-las com o LKH.
2. Para cada par de cidades vizinhas, o rótulo é **1 se essa aresta está no tour do LKH**, 0 se não.
3. Treinamos uma GNN a prever esse rótulo → para cada aresta, uma **probabilidade** de estar no tour.
4. A matriz dessas probabilidades chama-se **heatmap** ("mapa de calor").
5. Um algoritmo simples (gulosa + 2-opt) transforma o heatmap num tour válido.

**Porque prever arestas e não o tour diretamente?** Prever um tour é prever uma sequência
com uma restrição dura (cada cidade uma vez) — difícil para uma rede. Prever arestas é uma
**classificação binária** simples, feita para todas as arestas em paralelo; a validade do tour
fica a cargo da decodificação, que é garantida por construção. Esta é a abordagem de
Joshi et al. (2019). A alternativa (construir o tour cidade a cidade, *autoregressivo*,
como Kool et al. 2019) é normalmente treinada com aprendizagem por reforço.

**"Supervisionada"** quer dizer que aprendemos com respostas certas dadas por um professor
(o LKH), em vez de por tentativa e erro.

📁 `src/tsp_gnn/graph.py` (`tour_labels`), `src/tsp_gnn/decode.py` (`edge_heatmap`)

---

## 4. Grafos e GNN

### O que é uma GNN
Uma **GNN (Graph Neural Network)** trabalha sobre um grafo: nós (cidades) ligados por arestas.
Cada nó tem um vetor de características (no início, as suas coordenadas). Em cada camada,
cada nó:

1. recebe **mensagens** dos vizinhos,
2. **agrega-as** (soma, média, média ponderada…),
3. **atualiza** o seu vetor.

Isto chama-se **message passing** (passagem de mensagens). Depois de *L* camadas, cada nó
"sabe" o que se passa até *L* saltos de distância. Fórmula geral:

```
h_i ← ATUALIZA( h_i , AGREGA_{j vizinho de i}  MENSAGEM(h_i, h_j, e_ij) )
```

**Propriedade importante:** uma GNN não depende da ordem em que numeramos as cidades
(*equivariância a permutações*). Testámo-lo: renumerar as cidades dá os mesmos logits.

### O grafo k-NN
Não ligamos todas as cidades entre si: cada cidade liga-se às suas **k = 10 vizinhas mais
próximas** (k-NN, *k nearest neighbours*), nos dois sentidos.

**Porquê k-NN e não o grafo completo:**
- **Custo:** o grafo completo tem n(n−1) arestas (2 450 em TSP-50), cresce com n²; o k-NN
  tem ~n·k, cresce linearmente.
- **Quase não se perde nada:** as arestas do tour ótimo ligam quase sempre cidades próximas.
- **Classes mais equilibradas:** no grafo completo só ~4% das arestas (TSP-50) são do tour; no
  k-NN são ~16,6% em ambos os tamanhos.
- **Generalização:** cada nó tem ~10–15 vizinhos em TSP-20 e em TSP-50 — vizinhanças
  parecidas.

### O k = 10 e a cobertura
**Cobertura** = % das arestas do tour ótimo que estão no grafo. Com k = 10: **99,955%**
(TSP-20) e **99,920%** (TSP-50). Os poucos grafos incompletos (9/1000 e 20/500) têm sempre
exatamente uma aresta do tour de fora. Com k = 5 a cobertura cai para ~98%; com k = 15 chega a
~100%, mas há mais arestas negativas. k = 10 é o equilíbrio. Como a cobertura não é 100%, a
decodificação precisa de um plano B (ver secção 7).

📁 `src/tsp_gnn/graph.py` (`knn_edges`, `instance_to_data`), `scripts/graph_stats.py`

---

## 5. Os três modelos

Todos partilham a mesma estrutura (`EdgeGNN`):

1. **Embeddings:** coordenadas → vetor de 64 números (h⁰); distância → vetor de 64 (e⁰).
2. **6 camadas** do encoder, cada uma com **ligação residual** (somar a entrada à saída, para o
   treino de redes profundas não "encravar") e **BatchNorm** (normalizar valores para
   estabilizar o treino): `h ← h + ReLU(BN(camada(h)))`.
3. **Cabeça:** um pequeno MLP recebe [h_i, h_j, e_ij] e devolve um **logit** por aresta
   (número real; probabilidade = sigmoide(logit)).

O que muda é a camada do passo 2.

### GCN — "média dos vizinhos"
**Intuição:** cada cidade faz a média das vizinhas, com pesos fixos que só dependem de
quantas vizinhas cada uma tem. Não "olha" para as distâncias.

```
h_i' = Σ_{j ∈ N(i) ∪ {i}}  1/√(d̂_i · d̂_j) · W h_j          (d̂ = grau + 1)
```

### GATv2 — "atenção: quais vizinhos importam mais?"
**Intuição:** cada cidade decide **quanto ouvir** cada vizinha, com base nas duas cidades e
na distância entre elas. Usa 4 "cabeças" de atenção em paralelo.

```
s_ij = aᵀ · LeakyReLU(W_s h_j + W_t h_i + W_e e_ij)      pontuação
α_ij = softmax_j(s_ij)                                   pesos que somam 1
h_i' = Σ_j α_ij · W_s h_j
```
"v2" porque aplica a não-linearidade antes de `a`, o que permite que cada cidade ordene as
vizinhas de forma diferente (atenção "dinâmica").

### Gated GCN — "cada aresta tem a sua própria memória"
**Intuição:** além dos nós, **cada aresta tem um vetor que evolui** de camada para camada — a
rede vai formando uma "opinião" sobre cada aresta, que é exatamente o que queremos classificar.
As **portas** (gates) decidem, característica a característica, quanto deixar passar de cada
vizinho.

```
ê_ij = A h_i + B h_j + C e_ij                            nova aresta
η_ij = σ(ê_ij) / (Σ_j' σ(ê_ij') + ε)                     portas (entre 0 e 1, por dimensão)
h_i  ← h_i  + ReLU(BN(U h_i + Σ_j η_ij ⊙ V h_j))
e_ij ← e_ij + ReLU(BN(ê_ij))
```

### Tabela de diferenças

| | GCN | GATv2 | Gated GCN |
|---|---|---|---|
| Pesos dos vizinhos | fixos (graus) | aprendidos, 1 número por cabeça | aprendidos, 1 por dimensão |
| Usa a distância nas mensagens | não | sim (na atenção) | sim (no vetor da aresta) |
| Vetor da aresta atualizado | não | não | sim, em cada camada |
| Parâmetros (H=64, L=6) | 42 625 | 92 929 | 143 233 |

**Atenção:** os modelos têm capacidades diferentes (o Gated tem ~3,4× os parâmetros do GCN).
Igualar o nº de parâmetros é trabalho futuro.

📁 `src/tsp_gnn/models.py` (`EdgeGNN`, `GatedGCNLayer`), `tests/test_models.py`

---

## 6. O treino

### A perda
Cada aresta é uma pergunta de sim/não ("está no tour?"). Usamos a **entropia cruzada binária**
(BCE), que penaliza muito as previsões confiantes e erradas. Com z = logit, σ = sigmoide,
y = rótulo:

```
perda = − [ w · y · log σ(z)  +  (1 − y) · log(1 − σ(z)) ]
```

### O desbalanceamento (pos_weight)
Só ~16,5% das arestas do grafo são do tour. Sem correção, a rede aprenderia a dizer quase
sempre "não" e acertava 83% das vezes sem aprender nada. O **pos_weight** `w` = nº de
negativas / nº de positivas ≈ **5,05** faz cada aresta positiva pesar ~5× mais, equilibrando
as duas classes. É calculado **só no conjunto de treino**.

### Otimização
Adam com taxa de aprendizagem 10⁻³, que desce em forma de cosseno até ~0 ao longo de 20 épocas
(*cosine annealing*); batches de 64 grafos; seed fixa.

### Validação vs. teste
- **Treino** (12 000 instâncias): onde a rede aprende.
- **Validação** (500): no fim de cada época, mede-se o gap guloso; guarda-se o modelo da
  **melhor época**. É aqui que se fazem escolhas.
- **Teste** (1 000 TSP-20 + 500 TSP-50): só é usado **no fim**, uma vez, para medir. Nunca é
  usado para escolher nada — senão o resultado ficaria otimista.

Seeds diferentes para cada conjunto garantem que não há instâncias repetidas.

📁 `src/tsp_gnn/train.py` (`compute_pos_weight`, `train_epoch`, `eval_loss`),
`scripts/train_model.py`

---

## 7. A decodificação

### Do heatmap a um tour: a gulosa com union-find
1. Simetrizar: a rede dá uma probabilidade para i→j e outra para j→i; usamos a média.
2. Ordenar as arestas do grafo por probabilidade (desempate: a mais curta primeiro); depois,
   como **plano B**, todos os pares fora do k-NN, por distância.
3. Aceitar uma aresta se as duas cidades têm < 2 ligações **e** estão em **fragmentos
   diferentes**.

O **union-find** é uma estrutura que responde muito depressa a "estas duas cidades já estão
ligadas por um caminho?". Cada fragmento tem um "representante"; juntar dois fragmentos é
apontar um representante para o outro. Se as duas cidades têm o mesmo representante, ligá-las
fecharia um ciclo antes do tempo (um *subciclo*).

### O fecho do ciclo
Depois de n−1 arestas temos um caminho que passa por todas as cidades; as duas pontas têm
uma só ligação. A n-ésima aresta liga essas duas pontas e fecha o tour — mesmo que esteja fora
do grafo. Por isso, quando falta **uma** aresta do tour no k-NN, a "rede perfeita" (rótulos
verdadeiros) recupera-a e o gap é 0%. **Teste de sanidade:** com os rótulos verdadeiros, os
1 500 tours de teste saíram **iguais** aos do LKH.

### O 2-opt e porque "2-opt" não é um número único
Há muitas maneiras de aplicar as trocas do 2-opt, e cada uma chega a um ótimo local diferente.
Medimos três variantes sobre os mesmos tours iniciais:

| Gap médio (%) | Nosso (melhor por *i*) | Melhor troca global | Primeira melhoria |
|---|---|---|---|
| TSP-20 guloso + 2-opt | 2,02 | 1,31 | 2,59 |
| TSP-50 guloso + 2-opt | 3,66 | 2,78 | 4,14 |

A diferença entre variantes (até ~1,4 pontos) é maior do que algumas diferenças entre
modelos! Por isso **usamos sempre a mesma função** para os modelos e para as baselines — só
assim a comparação é justa.

📁 `src/tsp_gnn/decode.py` (`greedy_decode`, `two_opt`, `verify_tour`),
`scripts/check_decode.py`, `tests/test_decode.py`

---

## 8. A avaliação

### O gap de otimalidade
```
gap = 100 × (comprimento do tour / comprimento do LKH − 1)     [%]
```
Gap 0% = tão bom como o LKH; gap 5% = 5% mais comprido.

### As baselines
Guloso clássico e Christofides, cada um com e sem 2-opt. A **baseline a bater** é o
**guloso + 2-opt**: é o que a nossa pipeline faria com uma rede que só soubesse que "arestas
curtas são boas".

### As seeds
Treinar com seeds diferentes (0, 1, 2) dá redes diferentes. O "±" nas tabelas é o desvio-padrão
**entre seeds** — mede quanto o resultado depende da sorte da corrida. Com uma só seed, uma
conclusão pode estar errada (aconteceu: ver secção 9).

### O teste de Wilcoxon
Queremos saber se um modelo é **mesmo** melhor do que a baseline ou se a diferença é sorte.
Como os dois métodos são testados nas **mesmas** instâncias, comparamos instância a
instância (teste **emparelhado**):

1. Para cada instância, a diferença d = gap do modelo − gap da baseline.
2. Ordena-se |d| e atribuem-se postos (1 = menor diferença).
3. Soma dos postos das diferenças positivas vs. negativas. Se o modelo fosse igual à baseline,
   as duas somas seriam parecidas.
4. O **p-valor** diz quão provável seria uma diferença tão grande por acaso.

Porquê Wilcoxon e não o teste t? Os gaps são assimétricos, com muitos valores perto de 0 —
o Wilcoxon não assume distribuição normal.

**Bonferroni:** fazemos 12 testes. Com o limiar habitual de 0,05, cerca de 1 em 20 testes daria
"significativo" por acaso. Dividimos o limiar pelo nº de testes: 0,05 / 12 ≈ **0,0042**.
**Critério conservador:** usamos o **maior** p-valor das 3 seeds — só conta se for
significativo nas três corridas.

📁 `src/tsp_gnn/decode.py` (`optimality_gap`), `scripts/baselines.py`, `scripts/summarize.py`
(`paired_compare`)

---

## 9. Os resultados e o que significam

| Método | TSP-20 gulosa | TSP-20 + 2-opt | TSP-50 gulosa | TSP-50 + 2-opt |
|---|---|---|---|---|
| Guloso clássico | 12,83 | 2,02 | 16,63 | 3,66 |
| Christofides | 9,06 | 2,16 | **11,02** | **3,63** |
| GCN | 8,53 ± 0,19 | 1,79 ± 0,06 | 15,91 ± 0,49 | 3,66 ± 0,07 |
| GATv2 | 6,04 ± 0,22 | 1,23 ± 0,07 | 14,68 ± 1,02 | 4,07 ± 0,25 |
| Gated GCN | **4,85 ± 0,20** | **0,90 ± 0,03** | 16,38 ± 0,80 | 4,78 ± 0,17 |

**As mensagens principais:**
1. **Em TSP-20 a rede aprende.** Sem 2-opt, os três modelos batem o guloso clássico (o Gated é
   melhor em 83% das instâncias). A ordem Gated < GATv2 < GCN repete-se nas 3 seeds e bate com
   a lógica: quem usa mais informação (distâncias, vetor de aresta) aprende mais.
2. **O 2-opt esbate as diferenças.** Depois do 2-opt, o GCN deixa de ser significativamente
   melhor do que o guloso + 2-opt; o GATv2 e o Gated continuam (o Gated chega a 0,90%, menos de
   metade do guloso + 2-opt).
3. **Em TSP-50 a generalização falha.** Nenhum modelo chega ao Christofides; com 2-opt, o Gated
   é significativamente **pior** do que o guloso + 2-opt. A ordem inverte-se: o melhor em
   TSP-20 é o pior em TSP-50. A variação entre seeds também cresce (±0,5–1,0 vs. ±0,2).
4. **As seeds importam.** Só com a seed 0, o GATv2 + 2-opt em TSP-50 parecia significativamente
   pior; com 3 seeds a diferença deixa de ser significativa.
5. **Curvas de treino:** o GCN estagna logo após a época 2 (as épocas escolhidas — 13, 2, 5 —
   são sobretudo ruído); o GATv2 e o Gated ainda estão a melhorar na época 20.

![Curvas de treino](figs/curvas_val.png)

![Gap por método](figs/gaps_por_metodo.png)

📁 `results/tabela.md`, `scripts/summarize.py`

---

## 10. A generalização 20 → 50 e a escala das distâncias

Treinámos só com 20 cidades e testámos com 50, no **mesmo** quadrado [0,1]². Com mais cidades
no mesmo espaço, as cidades ficam mais próximas umas das outras:

| Distância média | TSP-20 | TSP-50 | Razão |
|---|---|---|---|
| arestas do grafo k-NN | 0,3828 | 0,2198 | 0,57 |
| arestas do tour | 0,1925 | 0,1136 | 0,59 |

A teoria prevê que a distância típica encolha com 1/√n: √(20/50) = 0,63 (o valor medido é um
pouco menor por efeitos de fronteira). Ou seja, em TSP-50 a rede vê distâncias ~**0,58×** mais
pequenas do que alguma vez viu no treino — está **fora da distribuição**. Além disso, a
BatchNorm, em avaliação, usa médias e variâncias guardadas do TSP-20, que já não batem certo.

**Boa notícia:** a razão entre arestas do grafo e do tour é ~2,0 nos dois tamanhos — a
estrutura relativa mantém-se, só a escala muda. Isto sugere a correção para a fase 2:
normalizar as distâncias (multiplicar por √n, ou dividir pela distância média do k-NN de cada
grafo). Na fase 1 **não** normalizámos de propósito, para mostrar o problema. Atenção: os
resultados são **coerentes** com esta explicação, mas só a experiência da fase 2 a pode provar.

📁 `scripts/graph_stats.py`, `src/tsp_gnn/graph.py` (`edge_attr`)

---

## 11. Limitações e próximos passos

**Limitações da fase 1**
- Só cidades uniformes e tamanhos pequenos (20 e 50); treino só em TSP-20.
- Modelos pequenos (H=64, L=6) e com capacidades diferentes entre si.
- 20 épocas: o GATv2 e o Gated ainda não tinham estabilizado.
- A esta escala o LKH é rápido (≈ 36 ms de CPU por instância de TSP-20); a GNN + gulosa
  (~1 ms) é mais rápida mas pior. A vantagem de velocidade só importaria em problemas maiores
  — o que exige resolver primeiro a generalização.

**Fase 2 (já decidido)**
- Normalizar as distâncias e repetir a experiência 20 → 50.
- Treinar também em TSP-50 (`tsp50_train`).
- Igualar o nº de parâmetros entre modelos; treinar mais épocas.

**Aplicação a carteiras (a confirmar com o professor)** — o plano concreto não está definido
neste repositório; uma ligação natural: a **seleção de carteiras com restrição de
cardinalidade** (escolher K de N ativos) também é combinatória. A receita da fase 1 transfere-se:
um grafo de ativos (ligados, por exemplo, pela correlação), uma GNN que prevê a probabilidade de
cada ativo entrar na carteira (como o heatmap), um solver exato em instâncias pequenas como
professor, e uma decodificação que garante as restrições + melhoria local (como gulosa + 2-opt).

**Papel da LLM (a confirmar com o professor)** — possibilidades a discutir: uma LLM como
*agente* que orquestra a pipeline (escolhe solver, decodificação e hiperparâmetros), que
**gera heurísticas** em código e as avalia automaticamente, ou que **explica** as decisões e os
resultados. A fase 1 dá a infraestrutura de avaliação (baselines, gap, testes estatísticos)
necessária para qualquer uma destas opções.

---

## 12. Perguntas que o professor pode fazer

**Porque aprendizagem supervisionada e não por reforço?** É mais simples e estável: há um
professor (LKH) que dá respostas quase ótimas e baratas a esta escala. O custo é precisar de
rótulos, o que fica caro para instâncias grandes.

**Como sabes que o LKH dá o ótimo?** Coincide com a força bruta em 7 cidades e as médias batem
com os valores ótimos publicados (3,83 / 5,70).

**Porque k = 10?** Cobertura de ~99,9% das arestas do tour com ~16,6% de positivas; k = 5 perde
~2% das arestas, k = 15 quase não ganha cobertura e desequilibra mais as classes.

**Porque é que a vantagem da GNN diminui com o 2-opt?** O 2-opt corrige muitos dos erros do
guloso (desfaz cruzamentos), por isso os dois pontos de partida acabam em ótimos locais
parecidos. A GNN só se destaca depois do 2-opt se der um ponto de partida claramente melhor
(GATv2 e Gated em TSP-20).

**Porque falha em TSP-50?** Mais provavelmente porque as distâncias encolhem ~0,58× e a
BatchNorm usa estatísticas do TSP-20: a rede vê valores que nunca viu. Testa-se na fase 2
normalizando as distâncias.

**A comparação entre modelos é justa, se têm nº de parâmetros diferentes?** Em parte: usam os
mesmos H, L, dados, treino e decodificação, mas não a mesma capacidade. Fica como experiência
futura igualar os parâmetros.

**Porque escolher a época pelo gap de validação e não pela perda?** O que nos interessa é a
qualidade do tour; a perda por aresta e o gap nem sempre andam juntos. E nunca pelo teste.

**Porque Wilcoxon e não teste t? E porquê Bonferroni?** Os gaps não são normais (assimétricos,
muitos perto de 0); o Wilcoxon não assume normalidade e aproveita o emparelhamento. Bonferroni
controla os falsos positivos ao fazer 12 testes.

**Porque três seeds?** Para medir a variação entre corridas; uma conclusão mudou ao passar de 1
para 3 seeds.

**Os resultados são reprodutíveis?** Sim: seeds fixas, commit registado em cada resultado e
scripts para cada passo. Na GPU, somas atómicas tornam as corridas quase (não bit a bit)
iguais.

**E o tempo de execução?** GNN + gulosa ≈ 0,5–1 ms por instância; com 2-opt 1–5 ms; LKH
≈ 36 ms (TSP-20) e ≈ 450 ms (TSP-50) de CPU por instância. Mas a esta escala a qualidade do
LKH é muito superior.

---

## 13. Glossário

- **Aresta:** ligação entre duas cidades.
- **Baseline:** método de referência com que se compara (aqui: guloso, Christofides).
- **Batch:** grupo de grafos processados juntos num passo de treino (64).
- **BatchNorm:** normaliza as características (média 0, variância 1) ao longo do batch; em
  avaliação usa médias guardadas do treino.
- **BCE (entropia cruzada binária):** perda para classificação sim/não.
- **Christofides:** heurística com garantia de ≤ 1,5 × o ótimo.
- **Cobertura:** % das arestas do tour ótimo presentes no grafo k-NN.
- **Embedding:** vetor de números que representa uma entrada (coordenadas, distância).
- **Época:** uma passagem completa pelos dados de treino.
- **Equivariância a permutações:** renumerar as cidades não muda as previsões.
- **Fora da distribuição:** dados diferentes dos vistos no treino (ex.: distâncias mais pequenas).
- **Gap de otimalidade:** quanto (%) um tour é mais comprido do que o do LKH.
- **Generalização:** funcionar bem em dados diferentes dos de treino (ex.: 20 → 50 cidades).
- **GNN:** rede neuronal que trabalha sobre grafos.
- **Guloso:** escolhe sempre a melhor opção imediata, sem voltar atrás.
- **Heatmap:** matriz de probabilidades de cada aresta estar no tour.
- **Heurística:** método rápido que dá boas soluções sem garantia de ótimo.
- **k-NN:** ligar cada cidade às suas k vizinhas mais próximas.
- **LKH:** Lin–Kernighan–Helsgaun, heurística de topo para o TSP; o nosso "professor".
- **Ligação residual:** somar a entrada de uma camada à sua saída (h + f(h)).
- **Logit:** saída da rede antes da sigmoide; probabilidade = sigmoide(logit).
- **Message passing:** cada nó recebe, agrega e usa informação dos vizinhos.
- **NP-difícil:** classe de problemas sem algoritmo polinomial conhecido para o ótimo.
- **Ótimo local:** solução que nenhuma pequena alteração (ex.: troca 2-opt) melhora.
- **p-valor:** probabilidade de uma diferença tão grande aparecer por acaso.
- **pos_weight:** peso extra das arestas positivas na perda (≈ 5,05).
- **Seed:** número que fixa a aleatoriedade, para os resultados serem reprodutíveis.
- **Sigmoide (σ):** função que transforma qualquer número num valor entre 0 e 1.
- **Subciclo:** ciclo fechado que não passa por todas as cidades (proibido num tour).
- **Teste emparelhado:** compara dois métodos nas mesmas instâncias.
- **Tour:** percurso que visita cada cidade uma vez e volta ao início.
- **TSP:** problema do caixeiro-viajante.
- **2-opt:** melhoria local que troca duas arestas (inverte um troço do tour).
- **Union-find:** estrutura que diz depressa se duas cidades já estão ligadas.
- **Validação:** conjunto usado para escolher (ex.: a melhor época); o **teste** só mede.
- **Wilcoxon:** teste estatístico emparelhado que não assume distribuição normal.
