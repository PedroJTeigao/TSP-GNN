# FCNF phase 1: generator statistics

MILP = HiGHS, 1 thread per instance, 8 instances in parallel. *open* = fraction of arcs open in the best solution (the base rate of the label). *unproven* = stopped at the time limit with a feasible solution; *gap* = HiGHS relative MIP gap of those.

| set | n | instances | arcs (mean) | MILP s mean | median | max | open | std open / inst. | optimal | unproven | gap unproven (%) | no solution | capacities rescaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fcnf20_train | 20 | 12000 | 101.0 | 0.23 | 0.12 | 3.7 | 0.106 | 0.018 | 12000 | 0 | — | 0 | 130 |
| fcnf20_val | 20 | 1000 | 101.0 | 0.23 | 0.12 | 3.7 | 0.106 | 0.018 | 1000 | 0 | — | 0 | 12 |
| fcnf20_test | 20 | 1000 | 101.0 | 0.22 | 0.12 | 2.6 | 0.105 | 0.018 | 1000 | 0 | — | 0 | 12 |
| fcnf50_test | 50 | 500 | 249.2 | 6.66 | 5.07 | 62.8 | 0.115 | 0.010 | 500 | 0 | — | 0 | 1 |
| fcnf100_test | 100 | 200 | 496.5 | 85.43 | 112.30 | 120.0 | 0.120 | 0.007 | 105 | 95 | 5.53 | 0 | 0 |

## By fixed-cost ratio F / (c u)

| set | ratio | instances | MILP s mean | open |
|---|---|---|---|---|
| fcnf20_train | 0.5 | 3986 | 0.21 | 0.105 |
| fcnf20_train | 1 | 3933 | 0.23 | 0.106 |
| fcnf20_train | 2 | 4081 | 0.25 | 0.106 |
| fcnf20_val | 0.5 | 349 | 0.20 | 0.104 |
| fcnf20_val | 1 | 330 | 0.22 | 0.107 |
| fcnf20_val | 2 | 321 | 0.28 | 0.108 |
| fcnf20_test | 0.5 | 351 | 0.23 | 0.106 |
| fcnf20_test | 1 | 332 | 0.21 | 0.104 |
| fcnf20_test | 2 | 317 | 0.22 | 0.106 |
| fcnf50_test | 0.5 | 183 | 7.26 | 0.114 |
| fcnf50_test | 1 | 162 | 5.80 | 0.115 |
| fcnf50_test | 2 | 155 | 6.84 | 0.116 |
| fcnf100_test | 0.5 | 60 | 92.11 | 0.118 |
| fcnf100_test | 1 | 73 | 79.64 | 0.119 |
| fcnf100_test | 2 | 67 | 85.75 | 0.122 |
