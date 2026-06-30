# Fine-tuning study — Cellpose-SAM, whole-cell (TissueNet test, N=300)

Fine-tuned on a 200-image TissueNet train subset, 20 epochs. Delta vs zero-shot.

| model | F1@0.5 | AJI+ | PQ | boundary-F1 | Dice |
|---|---|---|---|---|---|
| cellpose zero-shot | 0.844 | 0.718 | 0.670 | 0.869 | 0.902 |
| cellpose ft lr1e-05 | 0.859 | 0.732 | 0.688 | 0.885 | 0.915 |
| cellpose ft lr5e-05 | 0.843 | 0.715 | 0.671 | 0.869 | 0.905 |

- **lr1e-05**: ΔF1@0.5 = +0.015, ΔAJI+ = +0.014
- **lr5e-05**: ΔF1@0.5 = -0.001, ΔAJI+ = -0.003
