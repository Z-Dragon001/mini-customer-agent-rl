"""Narrow compatibility fixes for the locked ART 0.5.16 / Unsloth runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Unsloth must patch transformers before the training service imports it.
import unsloth  # noqa: F401
from art.local import LocalBackend
from art.unsloth.service import UnslothService

# Install in both the parent and the spawned model-service interpreter.
import peft.tuners.lora.model as _lora_implementation
from .art_numerics import install_plain_weight_dispatch
install_plain_weight_dispatch(_lora_implementation)


class RetailUnslothService(UnslothService):
    def __init__(self, *args, persist_optimizer=False, stable_logprobs=False, packed_positions=False,
                 packed_attention=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.persist_optimizer = persist_optimizer
        self._optimizer_restored = False
        self.stable_logprobs = stable_logprobs
        self.packed_positions = packed_positions
        self.packed_attention = packed_attention

    async def _train_shared(self, disk_packed_tensors, config, _config, verbose=False):
        import importlib
        from .logprobs import checkpointed_logprobs, probability_diagnostics
        implementation = importlib.import_module("art.unsloth.train")
        # Upstream chunks the projection, but autograd retains every chunk's
        # vocabulary-sized intermediates until backward. Recompute them instead.
        from functools import partial
        implementation._calculate_logprobs = partial(checkpointed_logprobs, fp32_output=self.stable_logprobs)
        if self.packed_positions:
            from .art_numerics import install_packed_positions
            install_packed_positions(implementation)
        if self.packed_attention:
            from .art_numerics import install_packed_attention, install_tensor_attention_bias
            install_packed_attention(implementation)
            install_tensor_attention_bias(importlib.import_module('unsloth.models.llama'))
        diagnostic_dir = Path(self.output_dir) / 'first_training_batch'
        if not diagnostic_dir.exists():
            import shutil
            shutil.copytree(disk_packed_tensors['dir'], diagnostic_dir)
            (diagnostic_dir/'descriptor.json').write_text(json.dumps({**disk_packed_tensors,
                'dir': str(diagnostic_dir)}, indent=2)+'\n')
        if not getattr(self, '_probability_diagnostics_installed', False):
            original_loss = implementation.loss_fn

            def measured_loss(inputs, new_logprobs, ref_logprobs, entropies, experimental_config):
                result = original_loss(inputs, new_logprobs, ref_logprobs, entropies, experimental_config)
                diagnostics = probability_diagnostics(inputs, new_logprobs, experimental_config.get('epsilon', .2))
                for key, value in diagnostics.items():
                    self._state.trainer._metrics['train'][key].append(value)
                attention_request = Path(self.output_dir) / 'attention_audit_request.json'
                if (diagnostics and config.learning_rate > 1e-9 and self.packed_attention and attention_request.exists()
                        and not (Path(self.output_dir)/'attention_audit.json').exists()
                        and not getattr(self, '_attention_audit_done', False)):
                    from art.local.checkpoints import get_last_checkpoint_dir
                    if int(Path(get_last_checkpoint_dir(self.output_dir)).name) != 1:
                        raise RuntimeError('Initial attention audit is missing after training already advanced')
                    from .art_numerics import replay_attention_audit
                    self._attention_audit_done = True
                    audit = replay_attention_audit(implementation, self._state.trainer,
                        json.loads(attention_request.read_text()), config, experimental_config)
                    (Path(self.output_dir)/'attention_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
                    print('attention_audit=' + json.dumps(audit), flush=True)
                request_path = Path(self.output_dir) / 'numerical_audit_request.json'
                if (diagnostics and self.stable_logprobs and self.packed_positions
                        and not (Path(self.output_dir)/'numerical_audit.json').exists()
                        and not getattr(self, '_numerical_audit_done', False) and request_path.exists()):
                    from .art_numerics import replay_probability_audit
                    self._numerical_audit_done = True
                    audit = replay_probability_audit(implementation, self._state.trainer,
                                                     json.loads(request_path.read_text()), config, experimental_config)
                    (Path(self.output_dir)/'numerical_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
                    print('numerical_audit=' + json.dumps(audit), flush=True)
                return result

            implementation.loss_fn = measured_loss
            self._probability_diagnostics_installed = True
        resumed_optimizer = False
        if self.persist_optimizer and not self._optimizer_restored:
            from art.local.checkpoints import get_last_checkpoint_dir
            from .optimizer_state import restore_optimizer_state
            latest = Path(get_last_checkpoint_dir(self.output_dir))
            saved = latest / 'retail_optimizer.pt'
            if saved.exists():
                self._last_training_mode = restore_optimizer_state(
                    saved, self._state.trainer.optimizer, expected_step=int(latest.name))
                resumed_optimizer = True
                print(json.dumps({'event': 'optimizer_restored', 'step': int(latest.name)}), flush=True)
            elif int(latest.name) > 1:
                raise RuntimeError('Missing optimizer state for resumed RL checkpoint')
            self._optimizer_restored = True
        # ART's first-call warmup performs an optimizer step (even at 1e-9).
        # On resume it would decay the restored Adam moments once more. Start
        # with the real batch instead; preserve ordinary cold-start behavior.
        service_module = importlib.import_module('art.unsloth.service')
        original_process = service_module.process_train_batch

        async def process_without_resume_warmup(*args, **kwargs):
            if resumed_optimizer:
                kwargs['warmup'] = False
            async for metric in original_process(*args, **kwargs):
                yield metric

        if resumed_optimizer:
            service_module.process_train_batch = process_without_resume_warmup
        try:
            async for metrics in super()._train_shared(disk_packed_tensors, config, _config, verbose=verbose):
                yield metrics
        finally:
            service_module.process_train_batch = original_process
        if self.persist_optimizer:
            from art.local.checkpoints import get_last_checkpoint_dir
            from .optimizer_state import save_optimizer_state
            latest = Path(get_last_checkpoint_dir(self.output_dir))
            save_optimizer_state(latest / 'retail_optimizer.pt', self._state.trainer.optimizer,
                                 step=int(latest.name), mode='rl')
            print(json.dumps({'event': 'optimizer_saved', 'step': int(latest.name)}), flush=True)

    async def train_sft(self, batches, verbose=False):
        # Unsloth's GRPOTrainer.__init__ ends with for_inference(). ART's
        # standalone SFT loop calls only train(), which leaves checkpointing off.
        model = self._state.peft_model
        model.for_training(use_gradient_checkpointing=True)
        model.config.use_cache = False
        layers = [m for m in model.modules() if hasattr(m, "gradient_checkpointing")]
        active = sum(bool(m.gradient_checkpointing) for m in layers)
        diagnostic = {"event": "sft_training_mode", "checkpointing_modules": len(layers),
                      "active_checkpointing_modules": active, "training": model.training}
        print(json.dumps(diagnostic), flush=True)
        if not layers or active != len(layers) or not model.training:
            raise RuntimeError("SFT gradient checkpointing did not activate")
        async for metrics in super().train_sft(batches, verbose=verbose):
            yield metrics


class RetailLocalBackend(LocalBackend):
    def __init__(self, *args, persist_optimizer=False, stable_logprobs=False, packed_positions=False,
                 packed_attention=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.persist_optimizer = persist_optimizer
        self.stable_logprobs = stable_logprobs
        self.packed_positions = packed_positions
        self.packed_attention = packed_attention

    async def _get_service(self, model):
        # Keep subprocess ownership local; upstream's global pkill can affect
        # another ART experiment running under the same account.
        from art.dev.get_model_config import get_model_config
        from art.dev.validate import is_dedicated_mode
        from art.utils.output_dirs import get_model_dir
        from mp_actors import move_to_child_process

        if model.name not in self._services:
            output_dir = get_model_dir(model=model, art_path=self._path)
            config = get_model_config(base_model=model.base_model, output_dir=output_dir,
                                      config=model._internal_config)
            if is_dedicated_mode(config) or config.get("tinker_args") is not None:
                raise ValueError("RetailLocalBackend supports the single-GPU Unsloth runtime only")
            os.environ["IMPORT_UNSLOTH"] = "1"
            service = RetailUnslothService(model_name=model.name, base_model=model.base_model,
                                           config=config, output_dir=output_dir,
                                           persist_optimizer=self.persist_optimizer,
                                           stable_logprobs=self.stable_logprobs,
                                           packed_positions=self.packed_positions,
                                           packed_attention=self.packed_attention)
            if not self._in_process:
                service = move_to_child_process(service, process_name="retail-model-service")
            self._services[model.name] = service
        return self._services[model.name]
