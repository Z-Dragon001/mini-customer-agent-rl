"""Pass ART's existing packed token positions through to the model's RoPE.

The pinned ART shared Unsloth path prepares input_pos but does not forward it.
This optional adapter retains the upstream objective and attention mask.
"""
from contextvars import ContextVar


def install_plain_weight_dispatch(implementation):
    """Avoid probing an optional quantizer for ordinary, unquantized weights.

    The pinned PEFT 0.21.2 checks TorchAO's version before the tensor type. The
    locked runtime's TorchAO is older; ordinary BF16 LoRA does not use it.
    Tensor subclasses (including actual TorchAO weights) retain native checks.
    """
    import torch
    if getattr(implementation, '_retail_plain_weight_dispatch', False):
        return
    original = implementation.dispatch_torchao

    def dispatch(target, *args, **kwargs):
        base = target.get_base_layer() if hasattr(target, 'get_base_layer') else target
        if type(getattr(base, 'weight', None)) in (torch.Tensor, torch.nn.Parameter):
            return None
        return original(target, *args, **kwargs)

    implementation.dispatch_torchao = dispatch
    implementation._retail_plain_weight_dispatch = True


def install_packed_positions(implementation):
    if getattr(implementation, '_retail_positions_installed', False):
        return
    positions = ContextVar('retail_packed_positions', default=None)
    original_factory = implementation.get_compute_loss_fn
    original_calculate = implementation.calculate_logprobs

    def get_compute_loss_fn(trainer):
        original = original_factory(trainer)
        def compute_loss(model, inputs, *args, **kwargs):
            token = positions.set(inputs['input_pos'])
            try:
                return original(model, inputs, *args, **kwargs)
            finally:
                positions.reset(token)
        return compute_loss

    def calculate_logprobs(*args, **kwargs):
        if not getattr(implementation, '_retail_positions_enabled', True):
            return original_calculate(*args, **kwargs)
        current = positions.get()
        if current is None:
            raise RuntimeError('Packed positions missing from trainer inputs')
        if len(args) >= 5:
            args = list(args)
            args[4] = {**args[4], 'position_ids': current.to(args[2].device)}
        else:
            kwargs['forward_kwargs'] = {**kwargs.get('forward_kwargs', {}),
                                       'position_ids': current.to(kwargs['input_ids'].device)}
        return original_calculate(*args, **kwargs)

    implementation.get_compute_loss_fn = get_compute_loss_fn
    implementation.calculate_logprobs = calculate_logprobs
    implementation._retail_positions_installed = True


def install_packed_attention(implementation):
    """Preserve ART's branch mask after Unsloth's outer LM wrapper replaces it.

    The hook is scoped to one forward and removed even on failure. Decoder
    layers receive the original mask before gradient checkpointing captures it.
    No shared-prefix branch can attend a sibling branch through a plain triangle.
    """
    if getattr(implementation, '_retail_attention_installed', False):
        return
    original = implementation.calculate_logprobs

    def calculate_logprobs(*args, **kwargs):
        if not getattr(implementation, '_retail_attention_enabled', True):
            return original(*args, **kwargs)
        trainer = args[1] if len(args) > 1 else kwargs['trainer']
        mask = args[3] if len(args) > 3 else kwargs['causal_mask']
        base = trainer.model.get_base_model()
        decoder = base.model
        forwarded = False

        def preserve(module, positional, keywords):
            nonlocal forwarded
            keywords = {**keywords, 'causal_mask': mask}
            forwarded = True
            return positional, keywords

        handle = decoder.register_forward_pre_hook(preserve, with_kwargs=True)
        try:
            result = original(*args, **kwargs)
            if not forwarded:
                raise RuntimeError('Packed attention mask did not reach the decoder')
            return result
        finally:
            handle.remove()

    implementation.calculate_logprobs = calculate_logprobs
    implementation._retail_attention_installed = True


def install_tensor_attention_bias(implementation):
    """Expand ART's [B,Q,K] bias for the locked xFormers GQA tensor layout.

    expand() keeps shared storage instead of allocating a dense mask per head.
    Fail explicitly if a future backend would silently ignore this branch mask.
    """
    if getattr(implementation, '_retail_bias_installed', False):
        return
    from dataclasses import replace
    import torch
    original = implementation.run_attention

    def run_attention(*, config, context, Q, K, V):
        mask = context.causal_mask
        if isinstance(mask, torch.Tensor) and mask.ndim == 3:
            if config.backend != 'xformers' or context.seq_info is not None:
                raise RuntimeError('ART branch masks currently require the pinned xFormers path')
            if config.n_groups != 1 and not context.requires_grad:
                bias = mask[:, None, None].expand(-1, config.n_kv_heads, config.n_groups, -1, -1)
            else:
                bias = mask[:, None].expand(-1, context.n_heads, -1, -1)
            context = replace(context, causal_mask=bias)
        return original(config=config, context=context, Q=Q, K=K, V=V)

    implementation.run_attention = run_attention
    implementation._retail_bias_installed = True


def replay_probability_audit(implementation, trainer, request, config, experimental_config):
    """Read-only replay of the control's first packed batch, before this run's update."""
    import hashlib
    import json
    from pathlib import Path
    import torch
    from art.preprocessing.pack import packed_tensors_from_dir
    from .logprobs import probability_diagnostics

    descriptor_path = Path(request['descriptor'])
    descriptor = json.loads(descriptor_path.read_text())
    packed = packed_tensors_from_dir(**descriptor)
    batch = {key: value[:1] for key, value in packed.items()}
    result = {'source_descriptor': str(descriptor_path),
              'descriptor_sha256': hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
              'before_first_reported_update': True, 'variants': {}}
    original_setting = getattr(implementation, '_retail_positions_enabled', True)
    try:
        for enabled in [False, True]:
            implementation._retail_positions_enabled = enabled
            inputs = {**batch, 'config': config, '_config': experimental_config,
                      'return_new_logprobs': True}
            # Native return-new-logprobs bypasses the loss, backward and update.
            shifted = implementation.get_compute_loss_fn(trainer)(trainer.model, inputs)
            unshifted = torch.nn.functional.pad(shifted[:, 1:], (0, 1))
            metrics = probability_diagnostics({key: value.to(shifted.device)
                                               for key, value in batch.items() if isinstance(value, torch.Tensor)},
                                              unshifted, experimental_config.get('epsilon', .2))
            result['variants']['bf16_fp32_positions' if enabled else 'bf16_fp32'] = metrics
            del shifted, unshifted
    finally:
        implementation._retail_positions_enabled = original_setting
    return result


def replay_attention_audit(implementation, trainer, request, config, experimental_config):
    """Paired read-only 3B forwards on every frozen control sequence, pre-update."""
    import hashlib
    import json
    from pathlib import Path
    import torch
    from art.preprocessing.pack import packed_tensors_from_dir
    from .logprobs import probability_diagnostics
    path=Path(request['descriptor'])
    packed=packed_tensors_from_dir(**json.loads(path.read_text()))
    output={'source_descriptor':str(path),'descriptor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'before_first_reported_update':True,'sequences':[],
            'note':'BF16/FP32/positions fixed in both variants; only branch attention preservation differs. '
                   'After native cold-start warmup, no backward or optimizer update in this replay.'}
    previous=getattr(implementation,'_retail_attention_enabled',True)
    try:
        for index in range(packed['tokens'].shape[0]):
            batch={key:value[index:index+1] for key,value in packed.items()}
            variants={}
            for enabled in (False,True):
                implementation._retail_attention_enabled=enabled
                shifted=implementation.get_compute_loss_fn(trainer)(trainer.model,
                    {**batch,'config':config,'_config':experimental_config,'return_new_logprobs':True})
                unshifted=torch.nn.functional.pad(shifted[:,1:],(0,1))
                metrics=probability_diagnostics(
                    {k:v.to(shifted.device) for k,v in batch.items() if isinstance(v,torch.Tensor)},
                    unshifted,experimental_config.get('epsilon',.2))
                variants['preserved' if enabled else 'overwritten']=metrics
                del shifted,unshifted
            output['sequences'].append({'index':index,'variants':variants})
    finally:
        implementation._retail_attention_enabled=previous
    return output
