# Released Model Validation History

This file reports the validation curve for the original released MR-IQA 2B training run. Validation used a held-out KONIQ split with 200 valid samples and 8 evaluation shards after each epoch.

| Epoch | SRCC | PLCC | Valid samples | Shards |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8840 | 0.8894 | 200 | 8 |
| 2 | 0.9213 | 0.9302 | 200 | 8 |
| 3 | 0.9318 | 0.9392 | 200 | 8 |
| 4 | 0.9274 | 0.9340 | 200 | 8 |
| 5 | 0.9271 | 0.9409 | 200 | 8 |
| 6 | 0.9249 | 0.9406 | 200 | 8 |
| 7 | 0.9205 | 0.9408 | 200 | 8 |
| 8 | 0.9288 | 0.9465 | 200 | 8 |
| 9 | 0.9307 | 0.9450 | 200 | 8 |
| 10 | 0.9251 | 0.9421 | 200 | 8 |

Best SRCC was reached at epoch 3. The final released checkpoint corresponds to epoch 10.
