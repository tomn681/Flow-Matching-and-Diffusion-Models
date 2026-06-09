# Losses API

::: src.losses

## Boundary Note

- `src.losses` exposes trainer-facing loss components, composition, and
  registration.
- `src.nn.losses` contains the tensor-level primitives those components build
  on, such as hinge/WGAN math and small discriminator modules.
