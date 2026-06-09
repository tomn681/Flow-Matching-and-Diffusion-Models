from __future__ import annotations


def test_training_utils_reexports_owner_modules() -> None:
    from utils import checkpointing, config_io, distributed, runtime_env, training_utils

    assert training_utils.load_json_config is config_io.load_json_config
    assert training_utils.save_json_config is config_io.save_json_config
    assert training_utils.allocate_run_dir is config_io.allocate_run_dir
    assert training_utils.set_seed is runtime_env.set_seed
    assert training_utils.resolve_device is runtime_env.resolve_device
    assert training_utils.resolve_batch_size is runtime_env.resolve_batch_size
    assert training_utils.summarize_model is runtime_env.summarize_model
    assert training_utils.safe_torch_load is checkpointing.safe_torch_load
    assert training_utils.latest_checkpoint is checkpointing.latest_checkpoint
    assert training_utils.save_checkpoint is checkpointing.save_checkpoint
    assert training_utils.maybe_load_checkpoint is checkpointing.maybe_load_checkpoint
    assert training_utils.setup_distributed is distributed.setup_distributed
    assert training_utils.is_distributed is distributed.is_distributed
    assert training_utils.is_main_process is distributed.is_main_process


def test_diffusion_utils_reexports_owner_modules() -> None:
    from utils.model_utils import diffusion_loading, diffusion_runtime, diffusion_utils

    assert diffusion_utils.build_diffusion_model is diffusion_loading.build_diffusion_model
    assert diffusion_utils.warn_attention_conditioning_shape is diffusion_loading.warn_attention_conditioning_shape
    assert diffusion_utils.encode_diffusion_batch is diffusion_runtime.encode_diffusion_batch
    assert diffusion_utils.decode_diffusion_batch is diffusion_runtime.decode_diffusion_batch
    assert diffusion_utils.prepare_diffusion_visual_batch is diffusion_runtime.prepare_diffusion_visual_batch
