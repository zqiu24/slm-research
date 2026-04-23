# Champion history

Append-only log of champion promotions. The "six months later" view: a new
team member reads this file top-to-bottom to understand the research
trajectory without needing to ask anyone.

Format per entry:

```
## YYYY-MM — champion-vN (<experiment-name>)
- Promoted from: <PR link>
- Config hash: <16 hex>
- W&B report: <link>
- Primary gain: <metric delta vs previous champion>
- See: docs/experiments/<experiment-name>.md
```

---

## (pending first champion)

- Initial baseline: plain AdamW + standard SwiGLU/GQA/RMSNorm architecture,
  per-family defaults in `configs/base/family/`. See `configs/experiments/champion.yaml`.
- No promotion event recorded yet; this entry will be appended once the
  first monthly review selects a winner.
