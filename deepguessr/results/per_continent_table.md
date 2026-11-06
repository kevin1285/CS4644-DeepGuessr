## Per-continent median km (↓)

| Method | Africa | Asia | Europe | North America | Oceania | South America |
|---|---|---|---|---|---|---|
| Train-mean (sanity) | 8102 | 8203 | 2863 | 5726 | 15683 | 9769 |
| Zero-shot CLIP (single) | 2273 | 2038 | 1097 | 2088 | 1307 | 7790 |
| Zero-shot CLIP (ensemble) | 2253 | 1732 | 1040 | 1793 | 1314 | 5300 |
| Regression (linear) | 3748 | 2965 | 1442 | 1727 | 2557 | 6376 |
| Regression (MLP) | 3786 | 2126 | 1035 | 1120 | 1872 | 5120 |
| Geocell linear (one-hot) | 7378 | 2028 | 1312 | 1096 | 1796 | 8275 |
| Geocell linear (smooth tau=75) | 7302 | 1948 | 1339 | 1113 | 1928 | 10004 |
| Geocell MLP (one-hot) | 3005 | 1674 | 861 | 801 | 1598 | 4965 |
| Geocell MLP (smooth tau=75) | 2822 | 1594 | 835 | 810 | 1531 | 4895 |

## Per-continent country accuracy (↑)

| Method | Africa | Asia | Europe | North America | Oceania | South America |
|---|---|---|---|---|---|---|
| Train-mean (sanity) | 0.0 | 0.0 | 3.4 | 0.0 | 0.0 | 0.0 |
| Zero-shot CLIP (single) | 17.9 | 21.9 | 23.4 | 46.7 | 75.8 | 5.8 |
| Zero-shot CLIP (ensemble) | 15.8 | 24.4 | 23.6 | 50.6 | 74.8 | 5.5 |
| Regression (linear) | 3.2 | 9.5 | 17.9 | 47.6 | 55.8 | 8.1 |
| Regression (MLP) | 3.2 | 15.8 | 28.3 | 55.2 | 65.6 | 20.1 |
| Geocell linear (one-hot) | 5.5 | 12.9 | 11.7 | 50.4 | 67.0 | 22.8 |
| Geocell linear (smooth tau=75) | 6.1 | 13.0 | 14.7 | 51.8 | 68.6 | 14.3 |
| Geocell MLP (one-hot) | 9.5 | 26.8 | 32.8 | 60.9 | 71.7 | 27.2 |
| Geocell MLP (smooth tau=75) | 10.0 | 25.0 | 34.8 | 60.8 | 72.4 | 27.0 |
