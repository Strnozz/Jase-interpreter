# Granite 4.1 8B challenger — stato prima dei pesi

Data: 2026-09-27. Nessun training Granite era iniziato al momento di questo audit.

## Provenienza congelata

- Challenger instruct ufficiale: `ibm-granite/granite-4.1-8b`, revisione `1504002f650e656a0a3789d99574df12e3e94ed0`, licenza Apache-2.0. Config `configs/models/granite41-8b-v26-challenger.json`; `AutoModelForCausalLM`, chat template nativo, NF4/bf16, LoRA r16/alpha32/dropout0,05 sulle ultime 16 layer **solo dopo verifica dei nomi reali**.
- Champion: Qwen3.5-9B V26 best step 1200, run `20260927T061719Z-780ce06c`, SHA modello `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Invariato.
- Corpus semantico identico al V26: 6026 train (`54c5e512f5b837b0e9b9d9d35841df5ee968fcee179b6f3095a5dfa1bdce0a6d`) e 1087 valid (`cc748b1330cac27264faccf040d440bd17e627723889ab2ac2c772bfce1ec934`). Split senza duplicati esatti o overlap normalizzato secondo l'audit completo; nessuna modifica dei messaggi/gold.
- Tokenizer Granite sull'intero corpus: train media 382,02, p99 489, max 491 token; valid media 381,83, p99 489, max 489. Zero oltre 512, prefix system/user mascherato e target assistant attivo. Audit `training/audits/granite41-8b-v26-token-audit.json`.
- V33 blind: **100 casi**, congelato prima di scaricare i pesi/training Granite in commit `d3b9526`, SHA `4818c5ea0b35aa0c5b6ff467ca1656b4e7313ed87d431975db18e2cb8b566f37`. Venti famiglie da cinque casi; zero overlap esatto, massimo Jaccard token 0,6 con corpora/pannelli precedenti e 0,315789 all'interno. LF imposto solo su V33 per mantenere l'hash tra checkout Windows. V32 resta solo regressione storica.
- Review queue V33: `benchmarks/v33/human_review_queue.jsonl`, **100/100 ancora da revisionare da un umano indipendente**; 19 casi segnalati per ontologia/interpretazione. Guard V5 sul gold: 89 ACCEPT, 9 HOLD_UNSUPPORTED_ACTION, 2 HOLD_MISSING_REQUIRED_INFORMATION. Questi HOLD non implicano errore gold. Ogni correzione futura deve creare una nuova versione, non modificare V33 congelato.

## Regressione e limiti del confronto

Il refactor Qwen mantiene esattamente i 124 LoRA target V26 e genera gli stessi tre output raw su V32 `001/005/021`. I test Qwen/Granite del template e loss masking passano. L'evaluator condiviso conserva raw output, greedy `do_sample=false`, massimo 480 nuovi token, metriche V1.3/Guard preesistenti e segnala separatamente validità JSON/schema. Nessun semantic repair viene applicato ai risultati.

**Limite causale principale:** V26 è warm-start da V25; Granite parte da base instruct. Pur usando lo stesso corpus V26 e 1500 step nominali, la storia di apprendimento non è identica. Il confronto misura candidati operativi, non dimostra che eventuali differenze siano causate soltanto dal foundation model. Il costo di training cumulativo Qwen deve essere distinto dal solo run V26.

Al lancio: RTX 4080 SUPER 16 GB, PyTorch `2.9.1+cu128`, Transformers `5.17.0`, PEFT `0.21.0`, bitsandbytes `0.50.2`, CUDA 12.8 (da manifest V26; riconfermare nel run Granite). Circa 97 GB liberi prima del download. I quattro shard ufficiali sommano 17.583.228.152 byte; adapter/checkpoint e cache restano fuori Git. Il runner staccato `scripts/run_granite41_challenger.py` blocca il full training se snapshot, GPU, bf16, corpus, V33, vanilla JSON, smoke backward o reload adapter non passano.
