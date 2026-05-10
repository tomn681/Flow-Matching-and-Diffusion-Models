from __future__ import annotations

import random


def select_visual_indices(ds, count: int, seed: int | None = None) -> list[int]:
    total = len(ds)
    if total <= 0:
        return []
    rng = random.Random(seed)
    indices = []
    if hasattr(ds, "data") and isinstance(getattr(ds, "data"), list):
        cases = {}
        for idx, row in enumerate(ds.data):
            case_id = row.get("Case") or row.get("case") or row.get("case_id")
            if case_id is None:
                continue
            cases.setdefault(case_id, []).append(idx)
        if cases:
            case_ids = list(cases.keys())
            rng.shuffle(case_ids)
            for case_id in case_ids[:count]:
                indices.append(rng.choice(cases[case_id]))
    if not indices:
        indices = list(range(total))
        rng.shuffle(indices)
        indices = indices[:count]
    return indices
