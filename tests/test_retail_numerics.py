import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires training runtime; runs on CPU')
class NumericalTests(unittest.TestCase):
    def test_mask_preservation_isolates_sibling_branches_and_cleans_up(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from mini_customer_agent.retail.art_numerics import install_packed_attention
        torch.manual_seed(3)
        config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                          num_attention_heads=4,num_key_value_heads=2,initializer_range=.2,
                          max_position_embeddings=64,use_cache=False)
        config._attn_implementation='eager'
        reference=Qwen2ForCausalLM(config).eval()
        class Decoder(torch.nn.Module):
            def forward(self,input_ids,causal_mask,position_ids):
                return reference(input_ids,attention_mask=causal_mask[:,None],position_ids=position_ids).logits
        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model=Decoder()
            def get_base_model(self):
                return self
            def forward(self,input_ids,causal_mask,position_ids):
                # Reproduce the pinned Unsloth wrapper's mask replacement.
                triangle=torch.zeros_like(causal_mask).masked_fill(
                    torch.ones_like(causal_mask,dtype=torch.bool).triu(1),float('-inf'))
                return self.model(input_ids,causal_mask=triangle,position_ids=position_ids)
        model=Outer()
        trainer=SimpleNamespace(model=model)
        prefix=[4,6,8];a=list(range(10,22));b=[30,31]
        ids=torch.tensor([prefix+a+b]);length=ids.shape[1]
        groups=torch.tensor([0]*len(prefix)+[1]*len(a)+[2]*len(b))
        allow=((groups[:,None]==groups[None,:])|(groups[None,:]==0)) & torch.ones(length,length,dtype=torch.bool).tril()
        mask=torch.zeros(length,length).masked_fill(~allow,float('-inf'))[None]
        pos=torch.tensor([list(range(len(prefix)+len(a)))+list(range(len(prefix),len(prefix)+len(b)))])
        def calculate(dtype,trainer,input_ids,causal_mask,forward_kwargs):
            return trainer.model(input_ids,causal_mask=causal_mask,**forward_kwargs)
        implementation=SimpleNamespace(calculate_logprobs=calculate)
        with torch.no_grad():
            separate=reference(torch.tensor([prefix+b])).logits[:,-len(b):]
            wrong=implementation.calculate_logprobs(None,trainer,ids,mask,{'position_ids':pos})[:,-len(b):]
            install_packed_attention(implementation)
            corrected=implementation.calculate_logprobs(None,trainer,ids,mask,{'position_ids':pos})[:,-len(b):]
        torch.testing.assert_close(corrected,separate,rtol=1e-5,atol=1e-5)
        self.assertGreater((wrong-separate).abs().max().item(),1e-3)
        self.assertEqual(len(model.model._forward_pre_hooks),0)
        with self.assertRaises(TypeError):
            implementation.calculate_logprobs(None,trainer,ids,mask,{'unsupported':True})
        self.assertEqual(len(model.model._forward_pre_hooks),0)

    def test_plain_weights_skip_optional_quantizer_but_subclasses_do_not(self):
        import torch
        from mini_customer_agent.retail.art_numerics import install_plain_weight_dispatch
        calls = []
        def original(target, *args, **kwargs):
            calls.append(target)
            return 'native'
        implementation = SimpleNamespace(dispatch_torchao=original)
        install_plain_weight_dispatch(implementation)
        self.assertIsNone(implementation.dispatch_torchao(torch.nn.Linear(2, 2), 'adapter'))
        self.assertEqual(calls, [])
        class QuantizedTensor(torch.Tensor):
            pass
        target = SimpleNamespace(weight=torch.ones(2, 2).as_subclass(QuantizedTensor))
        self.assertEqual(implementation.dispatch_torchao(target, 'adapter'), 'native')
        self.assertEqual(calls, [target])

    def test_bfloat16_logprob_normalization_uses_float32(self):
        import torch
        from mini_customer_agent.retail.logprobs import checkpointed_logprobs, probability_diagnostics
        states=torch.tensor([[[20.,19.,18.,17.],[25.,24.,23.,22.]]],dtype=torch.bfloat16)
        head=torch.eye(4,dtype=torch.bfloat16)
        targets=torch.tensor([[0,1]])
        reference=torch.log_softmax((states@head).double(),-1).gather(-1,targets[...,None]).squeeze(-1)
        legacy,_=checkpointed_logprobs(head,states,targets,1)
        stable,_=checkpointed_logprobs(head,states,targets,1,fp32_output=True)
        self.assertLess((stable.double()-reference).abs().max().item(),2e-6)
        self.assertGreater((legacy.double()-reference).abs().max().item(),.005)
        states.requires_grad_(True)
        head.requires_grad_(True)
        stable,_=checkpointed_logprobs(head,states,targets,1,fp32_output=True)
        observed_grads=torch.autograd.grad(-stable.sum(),(states,head))
        dense=torch.log_softmax((states@head).float(),-1).gather(-1,targets[...,None]).squeeze(-1)
        reference_grads=torch.autograd.grad(-dense.sum(),(states,head))
        for observed,expected in zip(observed_grads,reference_grads):
            torch.testing.assert_close(observed,expected,rtol=.01,atol=.004)
        inputs={'logprobs':torch.tensor([[float('nan'),-.2,-.3]]),
                'assistant_mask':torch.tensor([[False,True,True]])}
        aligned=torch.tensor([[ -.2,-.3,0.]])
        self.assertEqual(probability_diagnostics(inputs,aligned)['old_policy_ratio_outside_clip'],0)

    def test_packed_branches_match_separate_qwen_forwards(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from mini_customer_agent.retail.art_numerics import install_packed_positions
        torch.manual_seed(42)
        config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                          num_attention_heads=4,num_key_value_heads=2,initializer_range=.2,
                          max_position_embeddings=64,use_cache=False)
        config._attn_implementation='eager'
        model=Qwen2ForCausalLM(config).eval()
        prefix=[4,6,8];a=list(range(10,22));b=[30,31]
        ids=torch.tensor([prefix+a+b]);length=ids.shape[1]
        groups=torch.tensor([0]*len(prefix)+[1]*len(a)+[2]*len(b))
        allow=((groups[:,None]==groups[None,:])|(groups[None,:]==0)) & torch.ones(length,length,dtype=torch.bool).tril()
        mask=torch.zeros(length,length).masked_fill(~allow,float('-inf'))[None,None]
        pos=torch.tensor([list(range(len(prefix)+len(a)))+list(range(len(prefix),len(prefix)+len(b)))])
        implementation=SimpleNamespace()
        def calculate(dtype,trainer,input_ids,causal_mask,forward_kwargs,*args,**kwargs):
            return model(input_ids,attention_mask=causal_mask,**forward_kwargs).logits
        implementation.calculate_logprobs=calculate
        def factory(trainer):
            def loss(model,inputs):
                return implementation.calculate_logprobs(None,None,inputs['tokens'],mask,{})
            return loss
        implementation.get_compute_loss_fn=factory
        with torch.no_grad():
            separate=model(torch.tensor([prefix+b])).logits[:,-len(b):]
            wrong=implementation.get_compute_loss_fn(None)(model,{'tokens':ids,'input_pos':pos})[:,-len(b):]
            install_packed_positions(implementation)
            correct=implementation.get_compute_loss_fn(None)(model,{'tokens':ids,'input_pos':pos})[:,-len(b):]
        torch.testing.assert_close(correct,separate,rtol=1e-5,atol=1e-5)
        self.assertGreater((wrong-separate).abs().max().item(),1e-4)


if __name__=='__main__':
    unittest.main()
