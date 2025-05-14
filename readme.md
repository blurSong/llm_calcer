# A lightweight calculator to evaluate the math/dram complexity of llm inference.

## Usage

```python
from llm_calcer import auto_model, gen_report, print_report

model = auto_model("deepseek-ai/DeepSeek-V3-0324")
report = gen_report(model, 1024, 0, 1, "a16w4")

print_report(report)
```

## Results
```bash
╭──────────────────────────────┬─────────┬─────────────┬─────────┬──────────┬───────────────┬─────────────┬────────────╮
│ Model                        │ Phase   │ Precision   │ Batch   │ Tokens   │ Past Tokens   │ Math TOPS   │ DRAM GBs   │
├──────────────────────────────┼─────────┼─────────────┼─────────┼──────────┼───────────────┼─────────────┼────────────┤
│ deepseek-ai/DeepSeek-V3-0324 │ prefill │ a16w4       │ 1       │ 1024     │ 0             │ 82.1636     │ 335.564    │
├──────────────────────────────┼─────────┼─────────────┼─────────┼──────────┼───────────────┼─────────────┼────────────┤
│ deepseek-ai/DeepSeek-V3-0324 │ decode  │ a16w4       │ 1       │ 1        │ 1024          │ 0.0802429   │ 18.7769    │
╰──────────────────────────────┴─────────┴─────────────┴─────────┴──────────┴───────────────┴─────────────┴────────────╯
```