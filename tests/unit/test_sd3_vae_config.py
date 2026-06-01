from __future__ import annotations

from pathlib import Path

from models import ModelFactory
from utils import load_json_config


def test_sd3_vae_config_builds_and_uses_asymmetric_decoder_depth() -> None:
    cfg_path = Path("configs/sd3_vae.json")
    cfg = load_json_config(cfg_path)
    model = ModelFactory.build(cfg)

    encoder_blocks_per_stage = [len(stage.blocks) for stage in model.encoder.downs]
    decoder_blocks_per_stage = [len(stage.blocks) for stage in model.decoder.ups]

    assert all(count == 2 for count in encoder_blocks_per_stage)
    assert all(count == 4 for count in decoder_blocks_per_stage)

