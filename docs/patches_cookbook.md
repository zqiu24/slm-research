# Patches cookbook

When to reach for `src/patches/`: only when Megatron's ModuleSpec system
cannot express what you need (new residual connections, low-level layer
rewrites, kernel-selection changes). If you can implement a variant as an
`nn.Module` wired up via `build_spec`, do that instead — it's cleaner and
doesn't touch upstream code.

## Writing a patch

1. Create `src/patches/<name>.py`.
2. Docstring must cite the upstream function, Megatron SHA, and the
   rationale:
   ```python
   """
   PATCH: <name>
   Modifies: <fully.qualified.symbol>
   Upstream SHA ref: <megatron-sha> (line ~NNN)
   Rationale: <one paragraph>
   Required by: experiments tagged family:<family>
   """
   ```
3. Register:
   ```python
   from src.patches._registry import register_patch

   @register_patch(name="<name>", targets=("<fully.qualified.symbol>",))
   def apply():
       import <upstream>
       <upstream>.<symbol> = <replacement>
   ```
4. Declare a `tests/numerics/test_<name>_neutral.py` test demonstrating
   that the patch is a no-op when its feature is not activated.
5. List the patch in each experiment that needs it under
   `experiment.patches`.

## Conflict handling

Two patches that declare overlapping `targets` raise `PatchConflict` at
registration time. If you need the combined effect, write a single
combined patch.

## Hashing

`apply_patches(names)` returns `patch_set_hash = blake2s(sorted(name:source_sha))`.
This hash is recorded on every run so the exact monkey-patched state is
reproducible.
