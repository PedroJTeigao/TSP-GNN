## Conclusion

1. **Does the baseline work?** Yes: all 450 episodes passed the independent verifier with no collision, all 120 E1 lots were delivered in full, and LKH-3 + prioritized sustains 350, 604 and 1131 orders/h on the small, medium and large warehouses.
2. **Do conflicts concentrate?** Only moderately: the hottest 10 % of cells hold 67–99 % of the waits of an episode, but only 10–25 % of the cells see any wait, and the same share of cells chosen on other seeds catches 30–53 %, a little more than the busiest cells by traffic (21–45 %).
3. **What must the GNN predict in phase B?** Where and when the next wait will happen, per (cell, time box) and from the current state, because a static map is a weak predictor, the positive class is rare (0.6–2.2 % of boxes), and the model must beat the traffic map and spend the exact solver only where it pays: CBS gained 4.2 % over prioritized on the small warehouse and nothing on the medium one.

## Design

The grid is 4-connected, in the Kiva style: parking rows on top, blocks of shelves separated by one-cell corridors, and a free bottom row with the stations. Shelves are obstacles. Facing shelves of neighbouring blocks may share an access cell. A step moves a robot one cell or leaves it in place, and counts as one second. Service takes 3 steps at a shelf and at a station. Vertex and edge conflicts are forbidden.

An order has 3 to 12 products, each stored on one shelf, and goes to one station. The batch mode gives K orders at time 0; the lifelong mode draws Poisson arrivals. The number of products is twice the number of shelves.

The simulator has two layers. Layer 1 chooses the robot and the visit order; layer 2 moves the robots.

- **Assignment:** the free robot closest to any shelf of the order takes the oldest waiting order. The rule is simple on purpose and is the first thing to improve.
- **Visit order:** a TSP with fixed ends, solved by nearest neighbour (NN), NN + 2-opt, or LKH-3 on an explicit matrix with a dummy city. Orders with fewer than 4 stops are solved exactly by Held-Karp when LKH-3 is asked for.
- **PIBT:** one step at a time, with priority by age. Robots in service stay fixed.
- **Prioritized planning:** space-time A* in a rolling window, 20 steps planned and 5 executed, up to 3 retries with a new order of robots, then PIBT.
- **Windowed CBS:** exact on the next goal of every robot, with conflicts counted over 8 steps and 3 executed. Each call has a 30 s limit. A time-out hands 10 steps to PIBT, and 900 s per episode ends the use of CBS. It ran on small and medium only.

`verify.py` shares no code with the simulator. It reads the saved trajectory and events, and checks collisions, moves, picks, service times, deliveries and overlapping orders. Its tests alter episodes by hand and confirm that each violation is caught.

`pogema` was left out: it requires numpy 1.26 or older and pydantic 1.9 or older, and the project has numpy 2.5.

## Layer 1

Gaps are against LKH-3 on the same 200 orders. Lengths are shortest paths on the empty grid. 2-opt is nearly optimal at this size (1.6–1.7 % gap, optimal on 68–72 % of orders) because an order has only 7.6 stops on average. LKH-3 costs 11–16 ms per order, 150 to 200 times more than 2-opt. The per-order figures are measured here and are not comparable with the literature on long tours.

## E1

Each row is 5 seeds, mean ± 95 % interval; a seed fixes the warehouse catalogue and the orders, so the comparisons below are paired.

- **Sequence:** NN costs 7–9 % of delay against LKH-3. LKH-3 and 2-opt differ by about 1 % at most, and by nothing measurable on the small warehouse.
- **Planner:** prioritized beats PIBT by 3.2, 5.0 and 6.7 %, at 1.2 to 1.5 times its computation on the LKH-3 rows. On the medium warehouse CBS ties with prioritized (−0.0 %, p = 0.94) at about 180 times the computation; on the small one it gains 4.2 %.
- **CBS at 30 robots:** 23 % of the steps ran under PIBT on average (9–62 % by episode), after 17 time-outs per episode, and 3 of the 15 episodes used the whole 900 s budget. So the medium row is a mix of CBS and PIBT, not exact planning.

How to read the time columns: windowed planners plan only every few steps, so their p50 is close to 0 and the mean is the fair figure. The time includes the sequencing done in that step, which explains the LKH-3 rows. In the three episodes inspected, the slowest step was step 0, when all robots get their first order. Sequencing with LKH-3 goes through a subprocess, and these times were taken with 4 processes running on a shared machine.

## E2

Sequencing is LKH-3 throughout. Rates are multiples of the E1 throughput of LKH-3 + PIBT. The three requested rates are loose (0.5), medium (0.8) and saturated (1.2).

The rule fixed before the run was that a method keeps up if the backlog grows, over the second half, at less than 5 % of the arrival rate. The first run did not reach the breaking point on the large warehouse, so the sweep was extended to 1.3, 1.4, 1.5, 1.7 and 2.0 and all of E2 was repeated. "Saturated" is therefore only a name: it does not saturate the large warehouse.

The rule is noisy at low rates: Poisson arrivals alone give about 12 % of the rate per run at 0.067 orders per step. That explains the isolated failures at 0.8 and 0.9 on the small warehouse, where the backlog stays near 9 orders. The table of limits therefore gives the first rate that fails together with every higher one. This reading was added after seeing the data.

The real limit is above the E1 throughput, because a lot ends with a tail in which few robots are busy. The clean measure of capacity is the throughput at 2.0 x, in the second half of the episode, when the queue never empties. PIBT stops keeping up at 1.2–1.3 x and prioritized at 1.2–1.4 x. Prioritized delivers 5–9 % more at saturation. Backlog at the end includes the orders in progress, up to one per robot.

## E3

Waits and detours are read from the E1 trajectories. A wait is a robot with a pending goal that stays in place outside service; a detour is a move that does not bring it closer to its goal.

The per-episode share of waits in the hottest 10 % of cells is inflated: with few waits and few cells touched, most waits fall into the top 10 % whatever their pattern, and prioritized reaches 99 % simply because it has fewer waits. The fair columns are the ones where the cells are chosen on all runs together (42–63 %) and on the other seeds (30–53 %). A uniform map would give 10 %, and the busiest cells by visits give 21–45 %, so there is a real spatial signal beyond traffic, but a small one.

The heat maps show waits spread along the horizontal aisles between shelves, plus the cells beside the stations, rather than a few fixed bottlenecks. Almost every wait has an identifiable blocker (89–100 %), and only 0.6–2.2 % of (cell, 50 steps) boxes contain a wait, so the events are sharp in time.

Congestion is a modest cost: PIBT loses 7–11 % of its robot-steps and prioritized 3–7 %.

## Limitations

- **Naive assignment:** the rule ignores other orders and congestion, and one robot serves one order at a time.
- **No batteries, no rotations, discrete time:** a robot turns at no cost, never charges, and all moves take one step.
- **No re-sequencing:** the visit order is fixed when the order is assigned and uses distances on the empty grid.
- **Five seeds:** intervals are wide, and E2 slopes at low rates are noisy.
- **Fixed layout parameters:** one corridor width and one robot density per size. Density changes the congestion figures.
- **CBS is limited by the budget:** the 30 s limit and the 900 s budget shape the medium results, and large was not attempted.
- **Timing:** measured on a shared machine with 4 processes, under `nice`. Isolated slow steps of about 1 s in E2 were not investigated.
- **E1 capacity as a unit:** rate multiples refer to the E1 throughput, which is below the sustained capacity.
- **First E2 run:** the verifier treated orders cut by the horizon as errors. After the fix the whole of E2 was rerun in `e2b/`. The first 180 episodes gave the same deliveries, and `e2/` remains on disk.
