from __future__ import annotations

import torch
from torch.utils.data import Dataset

from losses.intermediate import IntermediateFeatureLoss
from models import ModelFactory
from models.vae import ConvPoolAlignment, VectorProjectionAlignment
from training import VAETrainer
from utils.model_utils.vae_utils import build_vae_model


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {"target": torch.rand(1, 32, 32, generator=generator)}


def _config(levels) -> dict:
    return {
        "model": {
            "model_type": "vae",
            "latent_type": "monai",
            "in_channels": 1,
            "out_channels": 1,
            "resolution": 32,
            "channels": [32, 64, 64],
            "num_res_blocks": 1,
            "attention_levels": [False, False, False],
            "latent_channels": 4,
            "norm_num_groups": 32,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "spatial_dims": 2,
        },
        "training": {"intermediate_supervision": {"default_loss": "l1", "levels": levels}},
    }


def test_sparse_identity_supervision_captures_only_requested_levels() -> None:
    model = ModelFactory.build(_config([0.5, None, {"weight": 0.25, "loss": "mse"}]))
    output = model(torch.randn(1, 1, 32, 32), sample_posterior=False)

    assert output.intermediate_features is not None
    assert set(output.intermediate_features) == {0, 2}
    assert output.intermediate_features[0].encoder.shape == (1, 32, 32, 32)
    assert output.intermediate_features[2].encoder.shape == (1, 64, 8, 8)
    assert output.intermediate_features[0].encoder.shape == output.intermediate_features[0].decoder.shape
    assert [spec.loss for spec in model.intermediate_supervision] == ["l1", "mse"]


def test_conv_pool_projection_is_trainable_and_has_requested_shape() -> None:
    cfg = _config(
        [
            {
                "weight": 0.5,
                "alignment": {"type": "conv_pool", "out_channels": 7, "pool_size": 3},
            },
            None,
            None,
        ]
    )
    model = ModelFactory.build(cfg)
    assert isinstance(model.intermediate_alignments["0"], ConvPoolAlignment)

    output = model(torch.randn(2, 1, 32, 32), sample_posterior=False)
    pair = output.intermediate_features[0]
    assert pair.encoder.shape == pair.decoder.shape == (2, 7, 3, 3)

    loss = IntermediateFeatureLoss(level=0, loss="l1")
    value = loss.compute(context={"intermediate_features": output.intermediate_features})
    value.backward()
    assert model.intermediate_alignments["0"].encoder_projection.conv.weight.grad is not None
    assert model.intermediate_alignments["0"].decoder_projection.conv.weight.grad is not None


def test_vector_projection_round_trips_through_state_dict() -> None:
    cfg = _config(
        [
            None,
            {
                "weight": 0.5,
                "loss": "cosine",
                "alignment": {"type": "vector_projection", "out_dim": 11},
            },
            None,
        ]
    )
    model = ModelFactory.build(cfg)
    assert isinstance(model.intermediate_alignments["1"], VectorProjectionAlignment)
    state = model.state_dict()

    restored = ModelFactory.build(cfg)
    restored.load_state_dict(state)
    output = restored(torch.randn(2, 1, 32, 32), sample_posterior=False)
    pair = output.intermediate_features[1]
    assert pair.encoder.shape == pair.decoder.shape == (2, 11)


def test_absent_supervision_preserves_default_model_output_contract() -> None:
    cfg = _config([None, None, None])
    cfg["training"].pop("intermediate_supervision")
    model = ModelFactory.build(cfg)
    output = model(torch.randn(1, 1, 32, 32), sample_posterior=False)

    assert model.intermediate_supervision == ()
    assert len(model.intermediate_alignments) == 0
    assert output.intermediate_features is None


def test_eval_builder_keeps_projection_state_but_disables_capture() -> None:
    cfg = _config([0.5, None, None])
    model = build_vae_model(cfg, torch.device("cpu"), set_eval=True)

    assert len(model.intermediate_alignments) == 1
    assert model.intermediate_supervision
    output = model(torch.randn(1, 1, 32, 32), sample_posterior=False)
    assert output.intermediate_features is None


def test_level_count_must_match_vae_depth() -> None:
    cfg = _config([0.5, None])
    try:
        ModelFactory.build(cfg)
    except ValueError as exc:
        assert "exactly one entry per VAE stage" in str(exc)
    else:
        raise AssertionError("Expected invalid intermediate supervision depth to fail.")


def test_vae_trainer_optimizes_and_reports_intermediate_projection(tmp_path) -> None:
    cfg = _config(
        [
            {
                "weight": 0.5,
                "alignment": {"type": "conv_pool", "out_channels": 8, "pool_size": 2},
            },
            None,
            None,
        ]
    )
    cfg["training"].update(
        {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-4,
            "save_every": 1,
            "output_dir": str(tmp_path / "run"),
            "input_normalize": "positive",
            "manual_device": "cpu",
            "save_images": False,
            "kl_weight": 0.0,
            "codebook_weight": 0.0,
        }
    )
    trainer = VAETrainer(cfg)
    dataset = _TinyDataset()
    trainer._setup(dataset, val_dataset=dataset)

    projection_parameters = set(map(id, trainer.model.intermediate_alignments.parameters()))
    optimizer_parameters = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert projection_parameters <= optimizer_parameters

    batch = {"target": torch.stack([dataset[0]["target"], dataset[1]["target"]])}
    metrics = trainer._training_step(batch, epoch=1)
    assert "intermediate_l1_l1" in metrics
    assert metrics["intermediate_l1_l1"] >= 0.0
