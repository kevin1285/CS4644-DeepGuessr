## Per-continent median km (↓)

| Method | Africa | Asia | Europe | North America | Oceania | South America | Unknown |
|---|---|---|---|---|---|---|---|
| Train-mean (sanity) | 7945 | 8112 | 2687 | 5892 | 15746 | 9791 | 2939 |
| Zero-shot CLIP (single) | 1968 | 2284 | 1056 | 2181 | 1262 | 6662 | 1958 |
| Zero-shot CLIP (ensemble) | 1990 | 1822 | 1031 | 1923 | 1276 | 4131 | 1958 |
| Regression (linear) | 3703 | 2779 | 1230 | 1643 | 2388 | 6076 | 3064 |
| Regression (MLP) | 2666 | 1865 | 804 | 940 | 1405 | 3866 | 5000 |
| Geocell linear (one-hot) | 3358 | 1704 | 856 | 839 | 1659 | 7063 | 10201 |
| Geocell linear (smooth tau=75) | 2738 | 1631 | 859 | 843 | 1418 | 6952 | 10142 |
| Geocell MLP (one-hot) | 1754 | 1221 | 633 | 634 | 1210 | 2570 | 10201 |
| Geocell MLP (smooth tau=75) | 1663 | 1156 | 607 | 637 | 1167 | 2402 | 10201 |
| LoRA geocell MLP (smooth tau=75) | 1402 | 983 | 469 | 531 | 1014 | 1412 | 1138 |

## Per-continent country accuracy (↑)

| Method | Africa | Asia | Europe | North America | Oceania | South America | Unknown |
|---|---|---|---|---|---|---|---|
| Train-mean (sanity) | 0.0 | 0.0 | 2.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| Zero-shot CLIP (single) | 16.1 | 22.4 | 20.7 | 45.3 | 70.7 | 7.2 | 0.0 |
| Zero-shot CLIP (ensemble) | 15.3 | 25.3 | 20.8 | 48.7 | 69.9 | 7.0 | 0.0 |
| Regression (linear) | 2.1 | 12.3 | 22.0 | 49.6 | 53.2 | 8.5 | 0.0 |
| Regression (MLP) | 6.9 | 23.1 | 32.7 | 57.0 | 64.4 | 25.7 | 0.0 |
| Geocell linear (one-hot) | 12.5 | 27.0 | 34.2 | 59.5 | 67.5 | 19.5 | 0.0 |
| Geocell linear (smooth tau=75) | 13.2 | 26.6 | 35.2 | 61.3 | 68.6 | 20.9 | 0.0 |
| Geocell MLP (one-hot) | 23.4 | 33.9 | 48.8 | 63.8 | 67.7 | 33.5 | 0.0 |
| Geocell MLP (smooth tau=75) | 24.4 | 35.5 | 49.5 | 65.0 | 68.6 | 33.8 | 0.0 |
| LoRA geocell MLP (smooth tau=75) | 28.2 | 39.3 | 54.6 | 68.2 | 70.9 | 41.7 | 0.0 |
