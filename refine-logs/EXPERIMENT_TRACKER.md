# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Sanity check | NewMethod (toy) | ICEWS18-1k | MRR, H@1 | MUST | TODO | Overfit on 1k triples |
| R002 | M0 | Data pipeline | -- | ICEWS18 | Pipeline integrity | MUST | TODO | Verify data loading |
| R003 | M0 | Metric check | TransE | ICEWS18 | MRR, H@1, H@3, H@10 | MUST | TODO | Match published metric computation |
| R004 | M1 | Baseline | TANGO | ICEWS18 | MRR, H@1, H@10 | MUST | TODO | Reproduce from official code |
| R005 | M1 | Baseline | PT2KGC | ICEWS18 | MRR, H@1, H@10 | MUST | TODO | Reproduce from official code |
| R006 | M1 | Baseline | Diffusion-TKG | ICEWS18 | MRR, H@1, H@10 | MUST | TODO | ECML-PKDD 2025 baseline |
| R007 | M2 | Main method | NewMethod (full) | ICEWS18 standard | MRR, H@1, H@3, H@10 | MUST | TODO | Primary result |
| R008 | M2 | Main method | NewMethod (full) | ICEWS18 open-world | MRR, H@1, H@3, H@10 | MUST | TODO | Open-world split |
| R009 | M2 | Main method | NewMethod (full) | GDELT standard | MRR, H@1, H@3, H@10 | MUST | TODO | Cross-dataset validation |
| R010 | M2 | Main method | NewMethod (full) | GDELT open-world | MRR, H@1, H@3, H@10 | MUST | TODO | Open-world GDELT |
| R011 | M2 | Main method | NewMethod (full) | YAGO standard | MRR, H@1, H@3, H@10 | MUST | TODO | Third dataset |
| R012 | M2 | Main method | NewMethod (full) | YAGO open-world | MRR, H@1, H@3, H@10 | MUST | TODO | YAGO open-world |
| R013 | M3 | Ablation | NewMethod w/o selective attention | ICEWS18 OW | MRR, H@1, H@10 | MUST | TODO | Selective attention ablation |
| R014 | M3 | Ablation | NewMethod w/o any attention | ICEWS18 OW | MRR, H@1, H@10 | MUST | TODO | Vanilla diffusion |
| R015 | M3 | Ablation | NewMethod frozen LLM | ICEWS18 OW | MRR, temporal consistency | MUST | TODO | Frozen encoder variant |
| R016 | M3 | Ablation | NewMethod no LLM | ICEWS18 OW | MRR, temporal consistency | MUST | TODO | Pure GNN+diffusion |
| R017 | M3 | Ablation | NewMethod DeBERTa encoder | ICEWS18 OW | MRR, temporal consistency | MUST | TODO | Small LM comparison |
| R018 | M3 | Ablation | Full method | GDELT OW | MRR, H@1, H@10 | MUST | TODO | Cross-dataset ablation |
| R019 | M4 | Simplicity | Overbuilt variant | ICEWS18 OW | MRR, params, latency | NICE | TODO | Extra GNN layers + aux losses |
| R020 | M4 | Simplicity | Minimal variant | ICEWS18 OW | MRR, params, latency | NICE | TODO | Diffusion only |
| R021 | M4 | Failure | Error analysis | ICEWS18 test | Error categories | NICE | TODO | 200 error manual analysis |
