"""Execute the installed LM wrapper's mask forwarding using a CPU-only sentinel.

This isolates its actual Python control flow, without loading model weights or
claiming to measure GPU attention outputs. The source hash makes it reproducible.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runtimes/art/.venv/lib/python3.12/site-packages/unsloth/models/llama.py'


def main():
    source=SOURCE.read_text()
    node=next(n for n in ast.parse(source).body
              if isinstance(n,ast.FunctionDef) and n.name=='CausalLM_fast_forward')
    code='from __future__ import annotations\n'+ast.get_source_segment(source,node)
    supplied=object()
    generated=object()
    captured={}

    class StopProbe(Exception):
        pass

    class Decoder:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            raise StopProbe

    namespace={'HAS_XFORMERS':True,
               'xformers':SimpleNamespace(attn_bias=SimpleNamespace(LowerTriangularMask=lambda:generated))}
    exec(compile(code,str(SOURCE),'exec'),namespace)
    model=SimpleNamespace(model=Decoder(), config=SimpleNamespace(
        output_attentions=False, output_hidden_states=False, use_return_dict=True))
    try:
        namespace['CausalLM_fast_forward'](None)(model,causal_mask=supplied)
    except StopProbe:
        pass
    result={'source':str(SOURCE.relative_to(ROOT)),
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'probe':'Installed CausalLM wrapper control flow, sentinel decoder, HAS_XFORMERS=True',
            'custom_mask_preserved':captured.get('causal_mask') is supplied,
            'custom_mask_replaced_by_lower_triangle':captured.get('causal_mask') is generated,
            'limit':'No weights, no GPU attention computation, no task-performance claim'}
    (ROOT/'experiments/retail-grpo-repair/attention_forwarding_diagnostic.json').write_text(
        json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
