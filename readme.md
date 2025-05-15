# A lightweight calculator to evaluate the math/dram complexity of llm inference.

## Usage

```python
from llm_calcer import auto_model, calc_inference_complexity

model = auto_model("deepseek-ai/DeepSeek-V3-0324")
calc_inference_complexity(model, 1024, 128, 1, "a16w4")
```

## Result
```bash
╭──────────────────┬───────────────┬─────────────┬─────────┬──────────┬──────────┬─────────────┬────────────╮
│ Model            │ Phase         │ Precision   │ Batch   │ Prompt   │ Output   │ Math TOPs   │ DRAM GBs   │
├──────────────────┼───────────────┼─────────────┼─────────┼──────────┼──────────┼─────────────┼────────────┤
│ DeepSeek-V3-0324 │ prefill       │ a16w4       │ 1       │ 1024     │ 512      │ 82.1636     │ 335.564    │
├──────────────────┼───────────────┼─────────────┼─────────┼──────────┼──────────┼─────────────┼────────────┤
│ DeepSeek-V3-0324 │ decode (once) │ a16w4       │ 1       │ 1024     │ 512      │ 0.0815261   │ 18.7772    │
╰──────────────────┴───────────────┴─────────────┴─────────┴──────────┴──────────┴─────────────┴────────────╯
```