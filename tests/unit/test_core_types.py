import torch

from core.types import ModelOutput, NoisyBatch, TrainingState


def test_model_output_defaults() -> None:
    reconstruction = torch.zeros(1, 1, 4, 4)
    output = ModelOutput(reconstruction=reconstruction)

    assert output.reconstruction.shape == (1, 1, 4, 4)
    assert output.posterior is None
    assert output.codebook_loss is None
    assert output.auxiliary == {}


def test_noisy_batch_fields() -> None:
    noisy = torch.randn(2, 1, 8, 8)
    target = torch.randn(2, 1, 8, 8)
    timesteps = torch.randint(0, 10, (2,))

    batch = NoisyBatch(noisy=noisy, target=target, timesteps=timesteps)

    assert batch.noisy.shape == batch.target.shape
    assert batch.timesteps.shape == (2,)


def test_training_state_defaults_and_payloads() -> None:
    state = TrainingState(
        epoch=1,
        global_step=5,
        model_state={"w": 1},
        optimizer_state={"lr": 1e-4},
        metrics={"loss": 0.5},
    )

    assert state.config is None
    assert state.extra == {}
    assert state.metrics["loss"] == 0.5
