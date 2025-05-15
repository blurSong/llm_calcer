import csv
import os
import json
import math
import re
from tabulate import tabulate
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files


def axwy_to_bytes(axwy: str):
    match = re.match(r"a(\d+)w(\d+)", axwy)
    assert match
    ab, wb = match.groups()
    return float(ab) / 8, float(wb) / 8


def get_model_config(path_or_hf_repo: str, cache_dir: str = None):

    if os.path.exists(path_or_hf_repo):
        local_path = os.path.join(path_or_hf_repo, "config.json")
    else:
        if cache_dir:
            local_path = os.path.join(cache_dir, path_or_hf_repo.split("/")[-1])
        else:
            local_path = None
        local_path = hf_hub_download(repo_id=path_or_hf_repo, filename="config.json", local_dir=local_path)

    try:
        with open(local_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise

    return config


def download_model_cache(hf_repo: str, cache_dir: str = None):

    weight_exts = ["bin", "safetensors", "pt", "pth", "ckpt", "npz"]

    local_path = os.path.join(cache_dir, hf_repo.split("/")[-1]) if cache_dir else None
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    files = list_repo_files(hf_repo)
    for f in files:
        if not any(f.endswith(ext) for ext in weight_exts):
            hf_hub_download(repo_id=hf_repo, filename=f, local_dir=local_path)

    return local_path


class llama:
    def __init__(self, config: dict, name: str = None):
        self.config = config
        self.name = name
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"] if "num_key_value_heads" in self.config else self.num_heads
        self.head_dim = self.config["head_dim"] if "head_dim" in self.config else self.hidden_size // self.num_heads
        self.intermediate_size = self.config["intermediate_size"]
        self.vocab_size = self.config["vocab_size"]

    def calc_inference_math_tops(self, tokens: int, past_tokens: int = 0, batch: int = 1, return_break_down: bool = False):
        embedding_macs = self.vocab_size * self.hidden_size * tokens
        lm_head_macs = self.hidden_size * self.vocab_size * tokens

        q_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens
        k_proj_macs = self.hidden_size * self.num_kv_heads * self.head_dim * tokens
        v_proj_macs = self.hidden_size * self.num_kv_heads * self.head_dim * tokens
        out_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens

        attention_tokens = tokens + past_tokens
        attention_qk_macs = self.num_heads * tokens * self.head_dim * attention_tokens
        attention_softmax_macs = self.num_heads * tokens * attention_tokens
        attention_qkv_macs = self.num_heads * tokens * attention_tokens * self.head_dim

        mlp_ffn_macs = self.intermediate_size * self.hidden_size * tokens
        mlp_matdot_macs = self.intermediate_size * tokens

        transformer_block_macs = (
            q_proj_macs
            + k_proj_macs
            + v_proj_macs
            + out_proj_macs
            + attention_qk_macs
            + attention_softmax_macs
            + attention_qkv_macs
            + mlp_ffn_macs * 3
            + mlp_matdot_macs
        )

        model_total_macs = transformer_block_macs * self.num_layers + embedding_macs + lm_head_macs
        model_total_macs = model_total_macs * batch

        if not return_break_down:
            return model_total_macs * 2 / 1e12

        lyr = self.num_layers
        scale = 2 * batch / 1e12
        break_down = {}
        break_down.update({"embedding": embedding_macs * scale})
        break_down.update({"lm_head": lm_head_macs * scale})
        break_down.update({"q_proj": q_proj_macs * lyr * scale})
        break_down.update({"kv_proj": (v_proj_macs + k_proj_macs) * lyr * scale})
        break_down.update({"out_proj": out_proj_macs * lyr * scale})
        break_down.update({"self_attention": (attention_qk_macs + attention_softmax_macs + attention_qkv_macs) * lyr * scale})
        break_down.update({"mlp": (mlp_ffn_macs * 3 + mlp_matdot_macs) * lyr * scale})
        return break_down

    def calc_inference_dram_gbs(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, axwy: str = "a16w4", return_break_down: bool = False
    ):
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.hidden_size * self.vocab_size

        q_proj_params = self.hidden_size * self.num_heads * self.head_dim
        k_proj_params = self.hidden_size * self.num_kv_heads * self.head_dim
        v_proj_params = self.hidden_size * self.num_kv_heads * self.head_dim
        out_proj_params = self.hidden_size * self.num_heads * self.head_dim

        mlp_ffn_params = self.intermediate_size * self.hidden_size

        attention_tokens = tokens + past_tokens
        q_activations = tokens * self.num_heads * self.head_dim
        k_activations = attention_tokens + self.num_kv_heads * self.head_dim
        v_activations = attention_tokens + self.num_kv_heads * self.head_dim

        transformer_block_params = q_proj_params + k_proj_params + v_proj_params + out_proj_params + mlp_ffn_params * 3

        head_and_tail_params = embedding_params + lm_head_params
        transformer_params = transformer_block_params * self.num_layers
        total_activations = (q_activations + k_activations + v_activations) * batch * self.num_layers
        # assume load q activations once a transformer block

        ab, wb = axwy_to_bytes(axwy)
        total_bytes = (transformer_params + head_and_tail_params) * wb + total_activations * ab

        if not return_break_down:
            return total_bytes / 1e9

        lyr = self.num_layers
        scale_a = ab * batch / 1e9
        scale_w = wb / 1e9
        break_down = {}
        break_down.update({"embedding": embedding_params * scale_w})
        break_down.update({"lm_head": lm_head_params * scale_w})
        break_down.update({"q_proj": q_proj_params * lyr * scale_w})
        break_down.update({"kv_proj": (v_proj_params + k_proj_params) * lyr * scale_w})
        break_down.update({"out_proj": out_proj_params * lyr * scale_w})
        break_down.update({"mlp": mlp_ffn_params * 3 * lyr * scale_w})
        break_down.update({"q_activations": q_activations * lyr * scale_a})
        break_down.update({"kv_activations": (v_activations + k_activations) * lyr * scale_a})
        return break_down


class llama4:
    def __init__(self, config: dict, name: str = None):
        # llama4 config contains text_config and vision_config.
        if "text_config" in config:
            self.config = config["text_config"]
        else:
            self.config = config
        self.name = name
        self.num_layers = self.config["num_hidden_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"] if "num_key_value_heads" in self.config else self.num_heads
        self.head_dim = self.config["head_dim"] if "head_dim" in self.config else self.hidden_size // self.num_heads
        self.num_experts_per_tok = self.config["num_experts_per_tok"]
        self.num_experts = self.config["num_local_experts"]
        self.intermediate_size = self.config["intermediate_size"]
        self.intermediate_size_mlp = self.config["intermediate_size_mlp"]
        self.vocab_size = self.config["vocab_size"]
        self.attn_temperature_tuning = self.config["attn_temperature_tuning"]

        interleave_moe_layer_step = self.config["interleave_moe_layer_step"]
        no_rope_step = 4  # FIXME hyper param = attn_temperature_tuning
        self.moe_layers = math.floor(self.num_layers / interleave_moe_layer_step)
        self.no_rope_layers = math.floor(self.num_layers / no_rope_step)
        assert self.moe_layers == len(self.config["moe_layers"])
        assert self.no_rope_layers == self.config["no_rope_layers"].count(0)

    def calc_inference_math_tops(self, tokens: int, past_tokens: int = 0, batch: int = 1, return_break_down: bool = False):
        embedding_macs = self.vocab_size * self.hidden_size * tokens
        lm_head_macs = self.hidden_size * self.vocab_size * tokens

        q_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens
        k_proj_macs = self.hidden_size * self.num_kv_heads * self.head_dim * tokens
        v_proj_macs = self.hidden_size * self.num_kv_heads * self.head_dim * tokens
        out_proj_macs = self.hidden_size * self.num_heads * self.head_dim * tokens

        attention_tokens = tokens + past_tokens
        attn_scales_macs = self.num_heads * self.head_dim * attention_tokens
        attention_qk_macs = self.num_heads * tokens * self.head_dim * attention_tokens
        attention_softmax_macs = self.num_heads * tokens * attention_tokens
        attention_qkv_macs = self.num_heads * tokens * attention_tokens * self.head_dim

        mlp_ffn_macs = self.intermediate_size_mlp * self.hidden_size * tokens
        mlp_matdot_macs = self.intermediate_size_mlp * tokens

        moe_router_macs = self.num_experts * self.hidden_size * tokens
        moe_r_experts_ffn_macs = self.intermediate_size * self.hidden_size * self.num_experts_per_tok * tokens
        moe_r_experts_matdot_macs = self.intermediate_size * self.num_experts_per_tok * tokens
        moe_s_expert_ffn_macs = self.intermediate_size * self.hidden_size * tokens
        moe_s_expert_matdot_macs = self.intermediate_size * tokens

        layer_attention_macs = (
            q_proj_macs
            + k_proj_macs
            + v_proj_macs
            + out_proj_macs
            + attention_qk_macs
            + attention_softmax_macs
            + attention_qkv_macs
        )
        layer_moe_macs = (
            moe_router_macs
            + moe_r_experts_ffn_macs * 3
            + moe_r_experts_matdot_macs
            + moe_s_expert_ffn_macs * 3
            + moe_s_expert_matdot_macs
        )
        layer_mlp_macs = mlp_ffn_macs * 3 + mlp_matdot_macs
        model_total_macs = (
            layer_attention_macs * self.num_layers
            + attn_scales_macs * self.no_rope_layers
            + layer_moe_macs * self.moe_layers
            + layer_mlp_macs * (self.num_layers - self.moe_layers)
            + embedding_macs
            + lm_head_macs
        )
        model_total_macs = model_total_macs * batch

        if not return_break_down:
            return model_total_macs * 2 / 1e12

        lyr, ffn_lyr, moe_lyr, nope_lyr = self.num_layers, self.num_layers - self.moe_layers, self.moe_layers, self.no_rope_layers
        scale = 2 * batch / 1e12
        break_down = {}
        break_down.update({"embedding": embedding_macs * scale})
        break_down.update({"lm_head": lm_head_macs * scale})
        break_down.update({"q_proj": q_proj_macs * lyr * scale})
        break_down.update({"kv_proj": (v_proj_macs + k_proj_macs) * lyr * scale})
        break_down.update({"out_proj": out_proj_macs * lyr * scale})
        break_down.update(
            {
                "attention": (attention_qk_macs + attention_softmax_macs + attention_qkv_macs) * lyr * scale
                + attn_scales_macs * nope_lyr * scale
            }
        )
        break_down.update({"mlp": layer_mlp_macs * ffn_lyr * scale})
        break_down.update({"moe": layer_moe_macs * moe_lyr * scale})
        return break_down

    def calc_inference_dram_gbs(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, axwy: str = "a16w4", return_break_down: bool = False
    ):
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.hidden_size * self.vocab_size

        q_proj_params = self.hidden_size * self.num_heads * self.head_dim
        k_proj_params = self.hidden_size * self.num_kv_heads * self.head_dim
        v_proj_params = self.hidden_size * self.num_kv_heads * self.head_dim
        out_proj_params = self.hidden_size * self.num_heads * self.head_dim

        mlp_ffn_params = self.intermediate_size_mlp * self.hidden_size
        moe_ffn_params = self.intermediate_size * self.hidden_size
        moe_router_params = self.hidden_size * self.num_experts

        attention_tokens = tokens + past_tokens
        q_activations = self.num_heads * tokens * self.head_dim
        k_activations = attention_tokens + self.num_kv_heads * self.head_dim
        v_activations = attention_tokens + self.num_kv_heads * self.head_dim

        layer_attention_params = q_proj_params + k_proj_params + v_proj_params + out_proj_params
        layer_mlp_params = mlp_ffn_params * 3

        # Assume load all experts in the prefill stage and only activated experts in the decode stage
        activated_experts = (self.num_experts if tokens > 1 else self.num_experts_per_tok) + 1
        layer_moe_params = moe_router_params + moe_ffn_params * 3 * activated_experts

        transformer_params = (
            +layer_attention_params * self.num_layers
            + layer_moe_params * self.moe_layers
            + layer_mlp_params * (self.num_layers - self.moe_layers)
        )
        total_activations = (q_activations + k_activations + v_activations) * batch * self.num_layers
        head_and_tail_params = embedding_params + lm_head_params

        ab, wb = axwy_to_bytes(axwy)
        total_bytes = (transformer_params + head_and_tail_params) * wb + total_activations * ab

        if not return_break_down:
            return total_bytes / 1e9

        lyr, ffn_lyr, moe_lyr = self.num_layers, self.num_layers - self.moe_layers, self.moe_layers
        scale_a = ab * batch / 1e9
        scale_w = wb / 1e9
        break_down = {}
        break_down.update({"embedding": embedding_params * scale_w})
        break_down.update({"lm_head": lm_head_params * scale_w})
        break_down.update({"q_proj": q_proj_params * lyr * scale_w})
        break_down.update({"kv_proj": (v_proj_params + k_proj_params) * lyr * scale_w})
        break_down.update({"out_proj": out_proj_params * lyr * scale_w})
        break_down.update({"mlp": layer_mlp_params * ffn_lyr * scale_w})
        break_down.update({"moe": layer_moe_params * moe_lyr * scale_w})
        break_down.update({"q_activations": q_activations * lyr * scale_a})
        break_down.update({"kv_activations": (v_activations + k_activations) * lyr * scale_a})
        return break_down


class deepseek_v3:
    def __init__(self, config: dict, name: str = None):
        # config https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py#L26
        # model https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
        # Deepseek has 2 MLA impls, naive and absorb.
        # Here we use the convenient naive impl to compute tops. But use the kvcache-efficient absorb impl to compute dram gbs.
        self.config = config
        self.name = name
        self.num_layers = self.config["num_hidden_layers"]
        self.first_k_dense_replace = self.config["first_k_dense_replace"]
        self.moe_layer_freq = self.config["moe_layer_freq"]
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.num_kv_heads = self.config["num_key_value_heads"] if "num_key_value_heads" in self.config else self.num_heads
        self.v_head_dim = self.config["v_head_dim"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self.q_lora_rank = self.config["q_lora_rank"] if "q_lora_rank" in self.config else None
        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.intermediate_size = self.config["intermediate_size"]
        self.moe_intermediate_size = self.config["moe_intermediate_size"]
        self.n_routed_experts = self.config["n_routed_experts"]
        self.n_shared_experts = self.config["n_shared_experts"]
        self.num_experts_per_tok = self.config["num_experts_per_tok"]
        self.vocab_size = self.config["vocab_size"]

        self.num_moe_layers = (self.num_layers - self.first_k_dense_replace) / self.moe_layer_freq
        self.num_dense_layers = self.num_layers - self.num_moe_layers

    def calc_inference_math_tops(self, tokens: int, past_tokens: int = 0, batch: int = 1, return_break_down: bool = False):
        embedding_macs = self.vocab_size * self.hidden_size * tokens
        lm_head_macs = self.hidden_size * self.vocab_size * tokens

        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank:
            q_proj_macs = (self.hidden_size * self.q_lora_rank + self.q_lora_rank * self.num_heads * q_head_dim) * tokens
        else:
            q_proj_macs = self.hidden_size * self.num_heads * q_head_dim * tokens

        kv_a_proj_dim = self.kv_lora_rank + self.qk_rope_head_dim
        kv_b_proj_dim = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        kv_a_proj_macs = self.hidden_size * kv_a_proj_dim * tokens
        kv_b_proj_macs = self.kv_lora_rank * kv_b_proj_dim * tokens

        out_proj_macs = self.num_heads * self.v_head_dim * self.hidden_size * tokens

        attention_tokens = tokens + past_tokens
        attention_qk_macs = self.num_heads * tokens * q_head_dim * attention_tokens
        attention_softmax_macs = self.num_heads * tokens * attention_tokens
        attention_qkv_macs = self.num_heads * tokens * attention_tokens * self.v_head_dim

        # mlp and moe
        mlp_ffn_macs = self.intermediate_size * self.hidden_size * tokens
        mlp_matdot_macs = self.intermediate_size * tokens

        moe_gate_macs = self.n_routed_experts * self.hidden_size * tokens
        moe_r_experts_ffn_macs = self.moe_intermediate_size * self.hidden_size * self.num_experts_per_tok * tokens
        moe_r_matdot_macs = self.moe_intermediate_size * self.num_experts_per_tok * tokens
        moe_s_expert_ffn_macs = self.moe_intermediate_size * self.hidden_size * self.n_shared_experts * tokens
        moe_s_expert_matdot_macs = self.moe_intermediate_size * self.n_shared_experts * tokens

        layer_attention_macs = (
            q_proj_macs
            + kv_a_proj_macs
            + kv_b_proj_macs
            + out_proj_macs
            + attention_qk_macs
            + attention_softmax_macs
            + attention_qkv_macs
        )
        layer_mlp_macs = mlp_ffn_macs * 3 + mlp_matdot_macs
        layer_moe_macs = (
            moe_gate_macs + moe_r_experts_ffn_macs * 3 + moe_r_matdot_macs + moe_s_expert_ffn_macs * 3 + moe_s_expert_matdot_macs
        )

        model_total_macs = (
            layer_attention_macs * self.num_layers
            + layer_mlp_macs * self.num_dense_layers
            + layer_moe_macs * self.num_moe_layers
            + embedding_macs
            + lm_head_macs
        )
        model_total_macs = model_total_macs * batch

        if not return_break_down:
            return model_total_macs * 2 / 1e12

        lyr, den_lyr, moe_lyr = self.num_layers, self.num_dense_layers, self.num_moe_layers
        scale = 2 * batch / 1e12
        break_down = {}
        break_down.update({"embedding": embedding_macs * scale})
        break_down.update({"lm_head": lm_head_macs * scale})
        break_down.update({"q_proj": q_proj_macs * lyr * scale})
        break_down.update({"kv_proj": (kv_a_proj_macs + kv_b_proj_macs) * lyr * scale})
        break_down.update({"out_proj": out_proj_macs * lyr * scale})
        break_down.update({"attention": (attention_qk_macs + attention_softmax_macs + attention_qkv_macs) * lyr * scale})
        break_down.update({"mlp": layer_mlp_macs * den_lyr * scale})
        break_down.update({"moe": layer_moe_macs * moe_lyr * scale})
        return break_down

    def calc_inference_dram_gbs(
        self, tokens: int, past_tokens: int = 0, batch: int = 1, axwy: str = "a16w4", return_break_down: bool = False
    ):
        embedding_params = self.vocab_size * self.hidden_size
        lm_head_params = self.hidden_size * self.vocab_size

        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.q_lora_rank:
            q_proj_params = self.hidden_size * self.q_lora_rank + self.q_lora_rank * self.num_heads * q_head_dim
        else:
            q_proj_params = self.hidden_size * self.num_heads * q_head_dim
        kv_a_proj_params = self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim)
        kv_b_proj_params = self.kv_lora_rank * self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        out_proj_params = self.hidden_size * self.num_heads * self.v_head_dim

        mlp_ffn_params = self.intermediate_size * self.hidden_size
        moe_ffn_params = self.moe_intermediate_size * self.hidden_size
        moe_router_params = self.hidden_size * self.n_routed_experts

        # absorb mla
        attention_tokens = tokens + past_tokens
        q_activations = tokens * self.num_heads * q_head_dim
        kv_activations = attention_tokens * self.kv_lora_rank
        k_pe_activations = attention_tokens * self.qk_rope_head_dim

        # Assume load all experts in the prefill stage and only activated experts in the decode stage
        activated_experts = (self.n_routed_experts if tokens > 1 else self.num_experts_per_tok) + self.n_shared_experts
        layer_moe_params = moe_router_params + moe_ffn_params * 3 * activated_experts
        layer_mlp_params = mlp_ffn_params * 3
        layer_attention_params = q_proj_params + kv_b_proj_params + kv_a_proj_params + out_proj_params

        transformer_params = (
            layer_attention_params * self.num_layers
            + layer_mlp_params * self.num_dense_layers
            + layer_moe_params * self.num_moe_layers
        )
        total_activations = (q_activations + kv_activations + k_pe_activations) * batch * self.num_layers
        head_and_tail_params = embedding_params + lm_head_params

        ab, wb = axwy_to_bytes(axwy)
        total_bytes = (transformer_params + head_and_tail_params) * wb + total_activations * ab
        if not return_break_down:
            return total_bytes / 1e9

        lyr, den_lyr, moe_lyr = self.num_layers, self.num_dense_layers, self.num_moe_layers
        scale_a = ab * batch / 1e9
        scale_w = wb / 1e9
        break_down = {}
        break_down.update({"embedding": embedding_params * scale_w})
        break_down.update({"lm_head": lm_head_params * scale_w})
        break_down.update({"q_proj": q_proj_params * lyr * scale_w})
        break_down.update({"kv_proj": (kv_a_proj_params + kv_b_proj_params) * lyr * scale_w})
        break_down.update({"out_proj": out_proj_params * lyr * scale_w})
        break_down.update({"mlp": layer_mlp_params * den_lyr * scale_w})
        break_down.update({"moe": layer_moe_params * moe_lyr * scale_w})
        break_down.update({"q_activations": q_activations * lyr * scale_a})
        break_down.update({"kv_activations (absorb)": (kv_activations + k_pe_activations) * lyr * scale_a})
        return break_down


def auto_model(path_or_hf_repo: str, cache_dir: str = None, custom_config: dict = None):
    config = get_model_config(path_or_hf_repo, cache_dir)
    model_type = config.get("model_type", None)

    # special config postprocess
    if model_type == "llama4":
        config = config["text_config"]

    if custom_config:
        config.update(custom_config)

    name = path_or_hf_repo.split("/")[-1]

    if model_type == "llama":
        return llama(config, name)
    elif model_type == "llama4":
        return llama4(config, name)
    elif model_type == "deepseek_v3":
        return deepseek_v3(config, name)
    else:
        raise NotImplementedError(f"Unsupported model: {model_type}")


def calc_inference_complexity(
    model,
    prompt: int,
    output: int = 0,
    batch: int = 1,
    axwy: str = "a16w4",
    verbose: bool = False,
):
    header = ["Model", "Phase", "Precision", "Batch", "Prompt", "Output"]
    pvalues = [model.name, "prefill", axwy, batch, prompt, output]
    dvaules = [model.name, "decode (once)", axwy, batch, prompt, output]

    # p
    pmath = model.calc_inference_math_tops(prompt, 0, batch, verbose)
    pdram = model.calc_inference_dram_gbs(prompt, 0, batch, axwy, verbose)

    # d. use output/2 for average
    past_token = prompt + output / 2
    dmath = model.calc_inference_math_tops(1, past_token, batch, verbose)
    ddram = model.calc_inference_dram_gbs(1, past_token, batch, axwy, verbose)

    if verbose:
        math_header, dram_header = list(pmath.keys()), list(pdram.keys())
        header += [f"{mh.capitalize()} TOPs" for mh in math_header] + [f"{dh.capitalize()} GBs" for dh in dram_header]
        pvalues += list(pmath.values()) + list(pdram.values())
        dvaules += list(dmath.values()) + list(ddram.values())
    else:
        header += ["Math TOPs", "DRAM GBs"]
        pvalues += [pmath, pdram]
        dvaules += [dmath, ddram]

    table = [header, pvalues, dvaules]
    if verbose:
        name = f"results/{model.name}_{axwy}_in{prompt}_out{output}_b{batch}.csv"
        with open(name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(table)
        print(f"Report saved to {name}")
    else:
        print(tabulate(table, headers="firstrow", tablefmt="rounded_grid", stralign="left", numalign="left"))


def test_llms():
    hf_repos = [
        "mlx-community/Meta-Llama-3.1-405B-4bit",
        "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
        "deepseek-ai/DeepSeek-V3-0324",
    ]
    for hf_repo in hf_repos:
        model = auto_model(hf_repo, "models")
        calc_inference_complexity(model, prompt=1024, output=512, batch=1, axwy="a16w4", verbose=True)


if __name__ == "__main__":
    test_llms()
