# Train search vertical slice: baseline congelata

Registrata prima di modificare contratto, Planner, Registry o provider. Working tree pulito; HEAD `4fb0c254317810c985d147ea0ea402a3a28daa77`. Nessun processo Python attivo alla verifica. Nessun training è autorizzato in questo milestone.

| Componente | Ultimo commit del file | SHA-256 del file |
| --- | --- | --- |
| Interpreter GoalContract 1.3 | `c59d5a843aa728b87831b6b17194d4e8ba178f99` | `0cf500656a0ce5e1374b5fec65d0bbbc444823c064074e79f74c31f843d4ee0e` |
| Schema 1.3 | `c59d5a843aa728b87831b6b17194d4e8ba178f99` | `d501feaeee016fe777e4d8b75d5a25eb1238be6b2d829fa41a006b66a06e5561` |
| Prompt Qwen V26 | `e3533166affd1bf47e99ea8314648b671b6aa0e4` | `effface5bd777a7f2d6f5caa2a1e83b4f37e5ba3a0193b8bdee6fc23426e71ba` |
| Planner 1.3 | `e70a9e5aac7565833f61ba1846ede6e9dad6a601` | `941d408995133ecf8fc404e6d4580f047149836d06d55f8e115d88bf5967ecc2` |
| Handoff mock | `e70a9e5aac7565833f61ba1846ede6e9dad6a601` | `0e92e1a4ad5baa77b5d06e019d9c3db5defe6fbcb3f4d3cec2a0acc35a87b9c3` |
| Guard routing V5 | `36c41585ef627b52534a67c11436e4f54d9f15f0` | `1b57cd3c80db7684299c58d05209e2de6af4e8e54e6b3abae247f5cb9ab40754` |
| Registry loader | `e70a9e5aac7565833f61ba1846ede6e9dad6a601` | `fcf772c4d7ce5c3540b511a80ba2b4be383183d498d60fcc58e6279593c9eba0` |
| Registry V3 | `e70a9e5aac7565833f61ba1846ede6e9dad6a601` | `9a8d2e384e3418974955dc712372eb813564d7da4544281871d1d70ed61d5560` |

## Benchmark congelati

| Pannello | Commit | SHA-256 |
| --- | --- | --- |
| V33_RAW | `d3b95261910fade3e512276e15748409cc731483` | `4818c5ea0b35aa0c5b6ff467ca1656b4e7313ed87d431975db18e2cb8b566f37` |
| V34 | `861b144ea000b071e5d9c15a2b59870e0ac5a7b0` | `b4a86432b691eb64220b64a24be29d29ee8dbd9b7b59d2d0fd75e2e900ff7734` |
| V35 | `3626f225ee1af41e1d3d8b03cdeab574ca9683c6` | `c0e4d08208406cc4d1198ee84b432d6ee8fc5eaf678ad641dbcbf3074fb41ac7` |
| V36 | `c152c0965d118e08cc3d5b96462f6be952d50b16` | `af40145327eb755a2496344df858f5ec26264a881a72b099ab73ee7640f5078d` |

## Adapter champion

`training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c/best_adapter/adapter_model.safetensors`: SHA-256 `1cdfe10bc7f05599072bb2b6763487b6124a339baab1063743c25ec18e56dfc5`, 86,594,112 byte. Run `20260927T061719Z-780ce06c` completo; modello base `Qwen/Qwen3.5-9B` revisione `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Adapter e benchmark esistenti non saranno modificati. Tutte le azioni con effetti esterni restano disabilitate (`execution_permitted=false`).
