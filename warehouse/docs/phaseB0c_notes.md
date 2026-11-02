## Recommendation

1. **With imperfect execution, is there margin above 3 % for a delay predictor?** No evidence of it: the function known gains −0.5 to +2.5 % on the best rule in R2 with PIBT, and −0.3 to +0.9 % with prioritized; hindsight clears 3 % once in ten cells (+3.1 ± 4.3 %) and only by keeping the best of 3 runs, while its mean of 3 never beats the rules by more than +0.1 %; with 5 seeds the intervals are too wide to exclude 3 % on the makespan alone, and mean delay says the same (at most +2.3 ± 9.4 %).
2. **Only in R2, or also in R1?** In neither: R2 is not clearly above R1, so making delays predictable from the state did not open a margin; the arms that use delays at all, oracles or maps, stay within noise of the blind ones.
3. **What would a model have to see, and what would its label be?** If one is tried anyway it sees local occupancy, robot type, floor zone and distance to stations, plus the planned paths of the other robots, which no arm here used, and its label is the delay realized per edge, without turns; the baseline to beat is prioritized under action dependencies with turns known in layer 1, because the largest gap in this phase is not a prediction: on the blind arm prioritized beats PIBT by 6–8 % at 30 robots and 11–14 % at 90.

## Decision

Gain in makespan over 5 seeds, positive is better, with a 95 % interval. The best of (b, c, d) is the lowest mean in each row, chosen after the fact, which leans towards the rules. "Best of 3" keeps the best hindsight run per seed, as in B0, which leans towards the oracle; delays are random, so that choice alone produces a gain. The mean of 3 has no such selection and is the fair reading of hindsight: it never exceeds +0.1 %.

Read the intervals: a half-width of 3–7 % means that 5 seeds cannot tell a gain of 3 % from zero in most rows. These rows show no margin; they do not prove there is none. Only the sign pattern across rows is firm: neither oracle is above the rules consistently, in either regime or at either density.

## Decision delay

The same for mean delay, which depends less on the last trip than the makespan does. Hindsight is negative or close to zero nearly everywhere; the function known reaches +2.3 ± 9.4 % once, at medium 90 with PIBT, and is within ±1.3 % in every other row.

## Rotation

**(b) − (a), rotation known in layer 1, is worth nothing measurable:** between −4.1 and +1.9 % with PIBT, and +0.8 to +1.8 % with prioritized, all within noise; in mean delay, −3.1 to +2.4 %. It does cut turns per move by 2–7 %.

That does not mean rotation is cheap. Rotation itself cost +22 % of makespan in the pilot (682 to 835 steps, PIBT on medium with 30 robots, no delays, seed 0 only), the same order as Varambally 2022. But that cost falls on every assignment almost alike, so knowing it changes who does what very little.

## Arms

"Own delay" is slips plus stopped steps per move; turns are counted apart. The free flow of each arm has rotation and no delays, so its percentage moves with the plan of that arm as well as with its makespan.

## Regimes

Both regimes lost the same, as asked: 0.15–0.16 steps of own delay per move with PIBT. The calibration targets PIBT, so prioritized at 90 robots lands 8 % lower in R2 (0.146 against 0.158). Delays and congestion together put the blind arm at 123–177 % of free flow, well above the 105 % of the nominal model in B0.

## Types

The type of robot does not show a stable gain: −1.7, +3.1, +0.1 and +2.3 %, with intervals that mostly cover zero. It is also a constant per robot, which needs measuring rather than learning.

## Protocol

**Execution (src/warehouse/exec.py).** A turn costs 1 step whatever the angle, as asked, including a half turn. With probability p a robot that would advance does not. With probability q per step of trying to move it stops for 5–20 steps, never during a service. A stopped robot appears to the planner as a robot in service: seen, not predicted. Random draws are per robot and step, so arms of a seed share their delays as far as their trajectories allow.

**No collisions.** One executor serves both planners: a plan fixes who goes through each cell in what order, and a robot enters a cell only when all earlier visitors have left it or leave in that step (Hönig 2019). PIBT is the case of a plan one step long. With no turns and no delays the executor repeats the old simulator exactly, which a test checks. The verifier gained an independent check, that no robot changes direction between two moves without a step in place, and all reported episodes pass it with 0 collisions.

**R1:** p = 0.10, q = 0.002, values chosen for this phase, since the plan gave none.

**R2:** p(robot, cell, n) = min(0.9, s × speed[type] × (1 + 0.25 n) × (1 + near(cell))), where n counts the other robots within 2 cells along the grid, near falls from 1 at a station to 0 at 6 cells, and type is the robot index modulo 3 with speed 0.5, 1, 1.5. q(cell) = q_bad on the bad floor, 20 % of the free cells in patches, fixed per layout, and 0 elsewhere. s and q_bad are calibrated per warehouse on seeds 100–105, apart from those of the report.

**Arms.** The stack is Clarke & Wright with 3 orders + LKH-3 + Hungarian over trips, with reassignment. Trips are fixed per seed: plain distances for (a), distances with turns for the rest. Turns reach the Hungarian and the visit order. Delays reach those and also the paths of the planner, without the turns. (b) to (f) all include the turns. The map of (c) comes from the (a) runs of the other 4 seeds, in the same regime and with the same planner; γ is fitted as in B0.

**(e)** adds to (b) the map of traffic alone, from the other seeds with the robots' own delays taken out, plus the expected delay of the function, per robot type, with occupancy distributed as in the other seeds. So in R1 it reduces to (c), and there it is a control: −2.4 to +1.2 %, as it should be. **(f)** starts from the times of the (b) run of the same seed.

**Extra arm.** `e0` is an addition: (e) without robot types, to see what the type adds.

## Budget

About 9 h of CPU against the ~6 h asked, over by 50 %: 6.0 h of episodes in the report, 0.8 h of a first version thrown away, and about 2.2 h of calibration, trips and diagnostics, from wall-clock stages rather than a meter. The cuts: prioritized did not run on large, and at 90 robots it has no `e0`. Fixing the trips per seed removed about half of the cost of an episode. The first version gave the distances with turns to PIBT as well; that made (b) lose to (a) by 6–8 % in delay, since a field that ignores the robot's heading sends it on detours. A diagnostic showed the turns help layer 1 and hurt the field, and that a field of delays without turns helps (delay 529 to 518 on (c)). Those runs are in `results/warehouse/phaseB0c_v1/`. Also added: a tie rule in PIBT, which keeps a robot going straight between equal options, in every arm (about a third fewer turns on seed 0, same makespan), and a stricter calibration, because the first one left stops 35 % off target.

## Limitations

- Five seeds with random delays: differences of 3–7 % are noise in many rows, so "no margin" is the reading of the pattern, not a proof.
- (e) is not a strict ceiling: it uses the expected occupancy per cell from other seeds, not the live occupancy, and prioritized reads one table for all types. A predictor of live occupancy was not measured.
- Delays are charged to distances only. No arm reorders trips during an episode in response to a delay, beyond the reassignment before picking starts, and no arm uses the planned paths of other robots.
- R1 and R2 use parameters chosen here, and no other intensity was run. Stronger delays could change the result.
- The bad floor is fixed per layout, so the static map can learn it. A floor that changes between days would favour a live predictor.
- Trips are not re-grouped with delays, so grouping cannot avoid the bad floor. B0 found that a traffic map changes grouping little, but that was without delays.
- Large ran with 60 robots, not its usual 80, as asked, with 5 orders per robot; on prioritized it did not run.
- The offline lot ends on one last trip, so the makespan is noisy; a lifelong run with throughput could be sharper.
