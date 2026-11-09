# CT-ScanEval results

| Model (version, settings) | Served model | Trials/task | Pass rate (spread) | Mean score | pass@k | pass^k | Cost per full run | Runtime |
|---|---|---|---|---|---|---|---|---|
| oracle:reference | reference | 1 | 100% (100%-100%) | 1.00 | 100% | 100% | $0.00 | 0s |
| gemini:gemini-3.5-flash-lite | gemini-3.5-flash-lite | 3 | 25% (25%-25%) | 0.46 | 25% | 25% | $0.00 | 204s |
| random:seed0 | random-seed0 | 3 | 21% (0%-38%) | 0.33 | 50% | 0% | $0.00 | 0s |

## Per task (passed trials / trials, mean partial-credit score)

| Task | oracle:reference | gemini:gemini-3.5-flash-lite | random:seed0 |
|---|---|---|---|
| t01_find_large_nodule | 1/1 (1.00) | 0/3 (0.67) | 0/3 (0.22) |
| t02_find_small_nodule | 1/1 (1.00) | 0/3 (0.33) | 0/3 (0.11) |
| t03_find_juxtapleural_nodule | 1/1 (1.00) | 0/3 (0.67) | 0/3 (0.56) |
| t04_no_nodule_hilum | 1/1 (1.00) | 0/3 (0.00) | 1/3 (0.33) |
| t05_no_nodule_lower_lobes | 1/1 (1.00) | 3/3 (1.00) | 1/3 (0.33) |
| t06_nodule_vs_vessels | 1/1 (1.00) | 3/3 (1.00) | 2/3 (0.67) |
| t07_count_nodules | 1/1 (1.00) | 0/3 (0.00) | 0/3 (0.11) |
| t08_measure_nodule | 1/1 (1.00) | 0/3 (0.00) | 1/3 (0.33) |

Run ids: `20261108T175537Z__oracle__reference`, `20261109T093716Z__gemini__gemini-3.5-flash-lite`, `20261109T094040Z__random__seed0`
