# Multi-Modal Conditioning

This guide documents the Phase N conditioning paths for text-only and image+text generation.

## Text-Only Conditioning

Use `training.conditioning: "text"` when prompts should become cross-attention context directly.

Expected config shape:

```json
{
  "training": {
    "conditioning": "text",
    "text_encoder": {
      "kind": "clip",
      "model_name": "openai/clip-vit-large-patch14"
    }
  }
}
```

At runtime, `GenerativeTrainer` builds a `TextConditioningAdapter` on the resolved device and feeds prompt strings from dataset samples under either `text` or `prompt`.

## Chain Conditioning

Use `training.conditioning: "chain"` when combining channel-concatenated inputs with cross-attention context.

Typical example:

- `concatenate`: low-level image-like conditioning concatenated into the model input
- `attention`: precomputed attention tensor
- `text`: prompt strings encoded into cross-attention embeddings

Expected sample keys:

- `image`: fallback concat-style conditioning
- `concat_cond`: explicit concat branch
- `attn_cond`: explicit attention branch
- `text` or `prompt`: raw prompt strings

If `text_encoder` is configured, the default chain path composes:

- `concatenate`
- `attention`
- `text`

## Direct Adapter Usage

Two text-conditioning surfaces exist intentionally:

- `TextConditioningAdapter`: accepts raw strings and encodes them
- registry `"text"` adapter: accepts pre-encoded embedding tensors only

Use the class or `build_text_conditioning_adapter(...)` when you want raw text handled inside the conditioning layer. Use the registry adapter only when text embeddings are already prepared upstream.

## Performance Note

If the same prompts are reused repeatedly, pre-encoding text once and passing embeddings directly avoids repeated tokenizer/model overhead.
