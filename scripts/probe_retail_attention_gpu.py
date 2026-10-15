"""Small random Qwen2: verify the actual Unsloth/xFormers packed forward on GPU0."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ['UNSLOTH_RETURN_HIDDEN_STATES']='1'
import unsloth
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
import peft.tuners.lora.model as lora_implementation
from mini_customer_agent.retail.art_numerics import (
    install_plain_weight_dispatch,install_packed_attention,install_tensor_attention_bias)

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:
        raise RuntimeError('Probe must use GPU0 only')
    install_plain_weight_dispatch(lora_implementation)
    directory=ROOT/'.cache/attention-probe-qwen2'
    if not (directory/'config.json').exists():
        torch.manual_seed(47)
        config=Qwen2Config(vocab_size=151936,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                          num_attention_heads=4,num_key_value_heads=2,initializer_range=.15,
                          max_position_embeddings=2048,tie_word_embeddings=True,use_cache=False)
        small=Qwen2ForCausalLM(config)
        small.save_pretrained(directory)
        AutoTokenizer.from_pretrained('/home/zxl/models/Qwen2.5-3B-Instruct',local_files_only=True).save_pretrained(directory)
        del small
    model,_=unsloth.FastLanguageModel.from_pretrained(str(directory),max_seq_length=256,
                    dtype=torch.bfloat16,load_in_4bit=False,local_files_only=True)
    model=unsloth.FastLanguageModel.get_peft_model(model,r=2,lora_alpha=4,lora_dropout=0,
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                    use_gradient_checkpointing=True,random_state=47)
    model.for_training(use_gradient_checkpointing=True)
    model.config.use_cache=False
    trainer=SimpleNamespace(model=model)
    def calculate(dtype,trainer,input_ids,causal_mask,forward_kwargs):
        return trainer.model(input_ids=input_ids,causal_mask=causal_mask,**forward_kwargs).logits
    implementation=SimpleNamespace(calculate_logprobs=calculate)
    import unsloth.models.llama as llama_implementation
    install_tensor_attention_bias(llama_implementation)
    install_packed_attention(implementation)
    prefix=[4,6,8,9,10,11,12,13];a=list(range(20,36));b=list(range(40,48))
    ids=torch.tensor([prefix+a+b],device='cuda')
    separate_ids=torch.tensor([prefix+b],device='cuda')
    groups=torch.tensor([0]*len(prefix)+[1]*len(a)+[2]*len(b),device='cuda')
    length=ids.shape[1]
    allow=((groups[:,None]==groups[None,:])|(groups[None,:]==0)) & torch.ones(length,length,dtype=torch.bool,device='cuda').tril()
    mask=torch.zeros(1,length,length,dtype=torch.bfloat16,device='cuda').masked_fill(~allow,float('-inf'))
    positions=torch.tensor([list(range(len(prefix)+len(a)))+list(range(len(prefix),len(prefix)+len(b)))],device='cuda')
    results={}
    for gradients in (False,True):
        with torch.set_grad_enabled(gradients):
            separate=model(input_ids=separate_ids).logits[:,-len(b):]
            implementation._retail_attention_enabled=False
            wrong=implementation.calculate_logprobs(None,trainer,ids,mask,{'position_ids':positions})[:,-len(b):]
            implementation._retail_attention_enabled=True
            corrected=implementation.calculate_logprobs(None,trainer,ids,mask,{'position_ids':positions})[:,-len(b):]
            old=(wrong.detach().float()-separate.detach().float()).abs().max().item()
            new=(corrected.detach().float()-separate.detach().float()).abs().max().item()
            assert new < .03 and new < old*.1, (gradients,old,new)
            result={'uncorrected_max_abs_error':old,'corrected_max_abs_error':new}
            if gradients:
                corrected.float().square().mean().backward()
                grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                assert grads and all(torch.isfinite(g).all().item() for g in grads)
                assert any(g.abs().max().item()>0 for g in grads)
                result['finite_nonzero_adapter_gradients']=True
            results['training' if gradients else 'reference_no_grad']=result
    output={'passed':True,'random_tiny_model':True,'gpu':torch.cuda.get_device_name(0),
            'results':results,'limit':'Kernel/forward/backward correctness test; not a trained task-performance result'}
    (ROOT/'experiments/retail-grpo-repair/attention_gpu_probe.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
