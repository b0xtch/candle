# Generates `engram_reference.safetensors`, the fixture used by `tests/engram.rs` to check that
# `candle_nn::engram` matches the official DeepSeek Engram demo implementation bit-for-bit
# (hashing) and numerically (module outputs).
#
#   pip install torch numpy sympy tokenizers safetensors
#   python engram_reference.py /path/to/Engram/engram_demo_v1.py
#
# The reference file is https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
# It downloads a HF tokenizer and imports `transformers`; both are stubbed out here with a small
# synthetic vocabulary so that the fixture can be produced offline. The real `tokenizers`
# normalizer from the reference is still used to build the compressed vocabulary.
import importlib.util
import sys
import types

import numpy as np
import torch
from safetensors.torch import save_file

# A synthetic vocabulary exercising the tokenizer-compression rules: case and accent folding,
# whitespace collapsing, the lone-space sentinel, empty strings and undecodable byte tokens.
WORDS = [
    "<pad>", "<s>", "</s>", " ", "  ", "\n", "\t", " \n ", "",
    "a", "A", " a", " A", "á", "ä", "Á", "e", "E", " é", "è", "ê",
    "the", "The", " the", " The", "THE", " Great", " great", "Alexander", " Alexander",
    "horse", " horse", " Horse", "B", "uce", "phal", "us", ".", ",", "Ｇｒｅａｔ", "ﬁ", "fi",
    "�", "��", "x�", "Straße", "strasse", "İstanbul", "istanbul", "Ω", "ω",
]
VOCAB = WORDS + [f"tok{i}" for i in range(len(WORDS), 160)] + [f" Tok{i % 40}" for i in range(40)]


class FakeTokenizer:
    def __len__(self):
        return len(VOCAB)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB[i] for i in ids)

    def convert_ids_to_tokens(self, tid):
        return f"<raw:{tid}>"


class AutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeTokenizer()


sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
spec = importlib.util.spec_from_file_location("engram_demo_v1", sys.argv[1])
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

CASES = {
    # Multi-branch (hyper-connection) layout as in the paper: 2- and 3-grams.
    "mhc": dict(
        engram=dict(engram_vocab_size=[61, 53], max_ngram_size=3, n_embed_per_ngram=8,
                    n_head_per_ngram=2, layer_ids=[1, 3], pad_id=2, seed=0, kernel_size=4),
        backbone=dict(hidden_size=16, hc_mult=2, vocab_size=len(VOCAB), num_layers=4),
    ),
    # Single residual stream (hc_mult = 1), what standard decoder-only models use: 2/3/4-grams.
    "single": dict(
        engram=dict(engram_vocab_size=[37, 41, 43], max_ngram_size=4, n_embed_per_ngram=12,
                    n_head_per_ngram=3, layer_ids=[0, 2], pad_id=0, seed=7, kernel_size=3),
        backbone=dict(hidden_size=8, hc_mult=1, vocab_size=len(VOCAB), num_layers=3),
    ),
}

out = {}
torch.manual_seed(299792458)
np.random.seed(0)
for name, case in CASES.items():
    ref.engram_cfg = ref.EngramConfig(tokenizer_name_or_path="fake", **case["engram"])
    ref.backbone_config = ref.BackBoneConfig(**case["backbone"])
    hc, hidden = case["backbone"]["hc_mult"], case["backbone"]["hidden_size"]
    b, t = 2, 13
    input_ids = torch.randint(0, len(VOCAB), (b, t), dtype=torch.int64)
    hidden_states = torch.randn(b, t, hc, hidden)
    out[f"{name}.input_ids"] = input_ids
    out[f"{name}.hidden_states"] = hidden_states
    for layer_id in case["engram"]["layer_ids"]:
        engram = ref.Engram(layer_id=layer_id)
        with torch.no_grad():
            # The demo leaves the norms at their default (ones) initialization; randomize them so
            # that the test also checks that every weight is used where it should be.
            for p_name, p in engram.named_parameters():
                if "norm" in p_name:
                    p.copy_(1.0 + 0.5 * torch.randn_like(p))
            output = engram(hidden_states=hidden_states, input_ids=input_ids)
        hm = engram.hash_mapping
        prefix = f"{name}.layer{layer_id}"
        out[f"{name}.lookup"] = torch.from_numpy(hm.compressed_tokenizer.lookup_table.copy())
        out[f"{prefix}.multipliers"] = torch.from_numpy(hm.layer_multipliers[layer_id].copy())
        out[f"{prefix}.head_sizes"] = torch.tensor(
            [x for y in hm.vocab_size_across_layers[layer_id] for x in y], dtype=torch.int64)
        out[f"{prefix}.hash_ids"] = torch.from_numpy(hm.hash(input_ids)[layer_id].copy())
        out[f"{prefix}.output"] = output
        for p_name, p in engram.state_dict().items():
            out[f"{prefix}.engram.{p_name}"] = p.detach().clone().contiguous()

save_file({k: v.contiguous() for k, v in out.items()}, "engram_reference.safetensors")
for k, v in sorted(out.items()):
    if v.numel() <= 8:
        print(k, v.tolist())
