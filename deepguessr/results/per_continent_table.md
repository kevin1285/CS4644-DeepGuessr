## Per-continent median km (↓)

| Method | Africa | Asia | Europe | North America | Oceania | South America | Unknown |
|---|---|---|---|---|---|---|---|
| Train-mean (sanity) | 8005 | 8119 | 2667 | 5841 | 15646 | 9817 | 2936 |
| Zero-shot CLIP (single) | 1236 | 1129 | 707 | 1599 | 1151 | 2247 | 1957 |
| Zero-shot CLIP (ensemble) | 1298 | 1063 | 732 | 1350 | 1230 | 2251 | 1957 |
| Regression (linear) | 2735 | 1699 | 879 | 1042 | 2150 | 3448 | 1270 |
| Regression (MLP) | 1523 | 1090 | 545 | 584 | 995 | 1652 | 1591 |
| Geocell linear (one-hot) | 1210 | 894 | 514 | 499 | 1005 | 2082 | 1059 |
| Geocell linear (smooth tau=75) | 1166 | 804 | 460 | 452 | 947 | 1647 | 1059 |
| Geocell MLP (one-hot) | 835 | 603 | 313 | 392 | 610 | 1006 | 1310 |
| Geocell MLP (smooth tau=75) | 755 | 547 | 291 | 354 | 570 | 828 | 1368 |
| LoRA geocell MLP (smooth tau=75) | 519 | 400 | 190 | 273 | 482 | 497 | 1310 |

## Per-continent country accuracy (↑)

| Method | Africa | Asia | Europe | North America | Oceania | South America | Unknown |
|---|---|---|---|---|---|---|---|
| Train-mean (sanity) | 0.0 | 0.0 | 3.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| Zero-shot CLIP (single) | 20.8 | 38.2 | 36.0 | 54.4 | 76.5 | 20.3 | 0.0 |
| Zero-shot CLIP (ensemble) | 19.7 | 38.9 | 33.6 | 67.5 | 74.2 | 16.8 | 0.0 |
| Regression (linear) | 4.9 | 22.8 | 34.1 | 57.8 | 61.5 | 26.2 | 0.0 |
| Regression (MLP) | 15.7 | 36.1 | 51.9 | 65.8 | 71.8 | 42.8 | 0.0 |
| Geocell linear (one-hot) | 30.7 | 48.5 | 59.9 | 74.0 | 78.1 | 37.9 | 0.0 |
| Geocell linear (smooth tau=75) | 30.7 | 49.7 | 61.2 | 73.9 | 78.3 | 39.7 | 0.0 |
| Geocell MLP (one-hot) | 36.9 | 52.8 | 66.8 | 74.2 | 79.5 | 52.0 | 0.0 |
| Geocell MLP (smooth tau=75) | 39.2 | 55.5 | 68.8 | 75.8 | 81.8 | 54.1 | 0.0 |
| LoRA geocell MLP (smooth tau=75) | 42.7 | 60.7 | 75.7 | 78.2 | 83.0 | 61.0 | 0.0 |
