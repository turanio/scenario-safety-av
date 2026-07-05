# Technology Decision Summary

The current source of truth is `notes/00_TECHNOLOGY_DECISION_DOCUMENT.md`.

Initial implementation decisions:

- Build the framework and deterministic baselines first.
- Keep the package CPU-only and testable without external datasets.
- Use highway-env as the first prototype simulator later.
- Keep simulator-specific code behind adapter interfaces.
- Treat CARLA as the final validation target, not an early blocker.
- Treat cVMD/cVMDx as the primary diffusion candidate after baseline validation.
- Treat MotionDiffuser as a literature reference unless runnable code and usable weights are verified.
