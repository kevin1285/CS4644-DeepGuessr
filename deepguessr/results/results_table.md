| Method | Median km ↓ | 25 km ↑ | 200 km ↑ | 750 km ↑ | 2500 km ↑ | Country acc ↑ |
|---|---|---|---|---|---|---|
| Train-mean (sanity) | 6890 | 0.0 | 0.0 | 0.7 | 7.4 | 0.6 |
| Common-country (sanity) | 9069 | 0.0 | 0.1 | 2.9 | 16.6 | 11.9 |
| Zero-shot CLIP (single) | 1934 | 0.2 | 5.2 | 23.8 | 57.5 | 29.6 |
| Zero-shot CLIP (ensemble) | 1763 | **0.2** | **5.3** | 25.2 | 60.2 | 30.5 |
| Regression (linear) | 2700 | 0.0 | 0.9 | 11.7 | 47.4 | 22.0 |
| Regression (MLP) | 1990 | 0.1 | 2.1 | 20.6 | 57.0 | 29.2 |
| Geocell linear (one-hot) | 2273 | 0.1 | 2.7 | 21.3 | 52.6 | 25.2 |
| Geocell linear (smooth tau=75) | 2320 | 0.1 | 2.7 | 21.8 | 52.1 | 25.4 |
| Geocell MLP (one-hot) | 1412 | 0.1 | 4.5 | 31.7 | 64.7 | 36.2 |
| Geocell MLP (smooth tau=75) | **1391** | 0.1 | 4.9 | **32.3** | **65.1** | **36.3** |
