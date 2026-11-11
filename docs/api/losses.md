# Losses API

::: src.losses

## Boundary Note

- `src.losses` exposes trainer-facing loss components, composition, and
  registration.
- `src.nn.losses` contains the tensor-level primitives those components build
  on, such as hinge/WGAN math and small discriminator modules.

## VAE Intermediate Supervision

`IntermediateFeatureLoss` compares one aligned encoder/decoder feature pair. The
VAE trainer creates one component per active entry in
`training.intermediate_supervision.levels`, so each level has its own weight,
criterion, and metric key (for example `intermediate_l1_l1` or
`intermediate_l3_mse`). Projection modules live on the VAE and the loss component
only performs the configured tensor distance.
