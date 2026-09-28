# Jase Conversational Interpreter V27 — pretrain audit

## Freeze and provenance

- Historical champion: Qwen3.5-9B V26 best adapter at `training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c/best_adapter`, unchanged.
- Base model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- V37 conversational blind panel was committed first (`90b3eef`): 120 scenarios, 150 user turns, SHA-256 `1f44fa9512dc1c2b8b5947da406f86083f48d98f3fc499998d450d98794e55e1`, zero exact overlap with 209,612 prior rows, maximum prior token Jaccard 0.583333. Gold is agent-authored and **not** independently human reviewed.
- V27 source and split hashes are in `data/v27_conversational/manifest.json`. The corpus excludes all benchmark requests as sources. Against V37 and earlier panels, exact overlap is zero, maximum token Jaccard 0.555556.

## Architecture

An opt-in Stage A proposal updates a versioned GoalDraft/ConversationState. The capability Registry, not the model, computes required slots and discovery. Stage B verbalizes a structured clarification plan. Deterministic finalization validates GoalContract 1.3 before the existing fail-closed Planner. Confirmation binds a material digest and is invalidated by corrections. Consequential provider execution remains disabled. Independent goals open separate drafts; dependent multi-goal orchestration remains a known limitation to evaluate.

## Corpus and loss

3,780 train and 655 validation examples. Train categories: non-action 650, incomplete 540, contextual 530, correction/cancellation/confirmation 550, result selection 180, independent multi-goal 180, complete 650, clarification wording 500. Some examples are projected from vetted, registry-resolvable V26 contracts; others are new structured-state contrasts. The generated portions remain templated in places and have not received independent human review.

Full-corpus tokenizer audit: Qwen2Tokenizer at pinned revision, max 422 train / 432 validation tokens, zero over 512, no truncation. Completion masking verified on every row: user message, structured state, and capability feedback have `-100` labels; only assistant semantic output or Stage B wording contributes to loss. Exact cross-split prompt overlap is zero. `scripts/audit_v27_conversational.py` records the mask preview and distribution.

## Training decision

Controlled internal-validation pilots compare 60-step warm start from V26 and 60-step base initialization on identical data/seed. The base candidate is selected only if its best internal validation loss is below 90% of the warm-start pilot; otherwise the warm start preserves the established semantics. **V37 is not used for pilot or checkpoint choice.** Full configuration remains NF4 4-bit, BF16, LoRA r16/alpha32/dropout .05 on the last 16 text layers, batch 1, accumulation 4, max length 512, learning rate 6e-6, 900 optimizer steps with validation/checkpoints every 150. Each model configuration requires its own smoke, backward, adapter reload and inference before full training. The checkpoint contains optimizer/scheduler/RNG for resume. Disk free before launch: about 81 GB; no old cache or adapter will be removed.

## Known limitations

The training set does not establish production quality. Existing GoalContract 1.3 and the Registry do not yet represent every conditional requirement or dependent goal. Generated contextual supervision uses structured state snapshots, not full production conversation logs. V37 should be evaluated once after training, with historical V33–V36 used only as regression diagnostics. Human review, provider binding and material false READY remain decisive gates.
