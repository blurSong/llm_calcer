# A lightweight calculator to evaluate the math/dram complexity of llm inference.

## Usage

```python
from llm_calcer import auto_model
model = auto_model("meta-llama/Llama-4-Scout-17B-16E-Instruct")

print("Prefill TOPS: {:.2f}".format(model.calc_inference_tops(1024, 0)))
print("Decode  TOPS: {:.2f}".format(model.calc_inference_tops(1, 1024)))
print("Prefill GBs:  {:.2f}".format(model.calc_inference_dram_gbs(1024, 0, axwy="a16w4")))
print("Decode  GBs:  {:.2f}".format(model.calc_inference_dram_gbs(1, 1024, axwy="a16w4")))

```