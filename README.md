# Jase Interpreter 2B — Starter Kit

> Pipeline multi-model NVIDIA: [fase iniziale](docs/MULTIMODEL_PHASE1.md) e [training Jase 9B](docs/JASE_9B_TRAINING.md).
> La baseline MLX 2B resta invariata; il nuovo runner QLoRA usa un ambiente
> `.venv-nvidia` separato e configurazioni in `configs/models/`.

Obiettivo: specializzare `Qwen3.5-2B` affinché trasformi una richiesta umana in un **Jase GoalContract v1** JSON, senza tentare di conoscere a priori tutti gli oggetti del mondo.

Il principio è:

```text
utente → Jase Interpreter → GoalContract → Entity Resolution → Capability Resolver → Planner → Execution
```

L'Interpreter capisce **cosa** vuole l'utente. Non decide **come** eseguirlo.

---

## 0. Requisiti Mac

Questo percorso usa MLX e richiede un Mac Apple Silicon (M1/M2/M3/M4/M5 o successivo).

Controlla:

```bash
uname -m
```

Deve restituire:

```text
arm64
```

Consigliato Python 3.11 o 3.12.

---

## 1. Avvio rapido sul Mac

Apri Terminale nella cartella di questo progetto e lancia:

```bash
bash scripts/bootstrap_mac.sh
```

Lo script:

1. crea `.venv`;
2. installa MLX-LM e dipendenze;
3. genera 3000 esempi train, 300 validation e 500 test;
4. valida tutti i GoalContract contro JSON Schema.

Il modello MLX usato sul Mac è:

```text
mlx-community/Qwen3.5-2B-4bit
```

### Baseline prima del training

```bash
source .venv/bin/activate
python scripts/infer_mlx.py "Trovami un panino con la mortadella a Milano sotto i 10 euro"
```

Salva qualche risultato: ci servirà per confrontare il modello prima/dopo.

### Benchmark baseline

Per partire senza aspettare troppo:

```bash
python scripts/eval_mlx.py --limit 50
```

### Fine-tuning

Profilo standard:

```bash
bash scripts/train_mac.sh
```

Se il Mac ha poca memoria o va in memory pressure:

```bash
bash scripts/train_mac_lowmem.sh
```

Gli adapter finiscono in:

```text
adapters/jase-v1/
```

### Prova il modello fine-tuned

```bash
python scripts/infer_mlx.py \
  --adapter adapters/jase-v1 \
  "Mi trovi uno Sphynx femmina a Milano sotto i 1200 euro?"
```

### Benchmark dopo il training

```bash
python scripts/eval_mlx.py \
  --adapter adapters/jase-v1 \
  --limit 100
```

Confronta almeno:

- `json_parse_rate`
- `schema_valid_rate`
- `first_action_rate`
- `first_target_rate`
- `exact_contract_rate`

`exact_contract_rate` è volutamente severa: due JSON semanticamente equivalenti possono non essere byte-per-byte identici.

---

## 2. Il GoalContract v1

Schema:

```text
schema/goal_contract_v1.schema.json
```

Esempio:

```json
{
  "schema_version": "1.0",
  "goals": [
    {
      "action": "find",
      "raw_action": "mi trovi",
      "target": {
        "name": "gatto",
        "type": "animal",
        "raw": "gatto"
      },
      "attributes": [
        {
          "property": "breed",
          "operator": "eq",
          "value": "Sphynx"
        }
      ],
      "constraints": [
        {
          "property": "location",
          "operator": "near",
          "value": "Milano"
        },
        {
          "property": "price",
          "operator": "lte",
          "value": 1200,
          "currency": "EUR"
        }
      ],
      "preferences": []
    }
  ],
  "clarification": {
    "required": false,
    "reason": null,
    "missing_fields": []
  }
}
```

### Perché è open-world

`target.name`, `target.type`, `action`, `property` e `value` non sono cataloghi chiusi.

Quindi il modello può ricevere una cosa mai vista prima:

```text
"Cercami un Zorba X7 nero sotto i 400 euro"
```

ed estrarre comunque struttura + valore senza che `Zorba X7` fosse noto durante il training.

Gli enum sono usati quasi solo per gli **operatori logici** (`lte`, `gte`, `contains`, ecc.), perché quelli sono una grammatica stabile.

---

## 3. Come viene creato il dataset

Generatore:

```text
scripts/generate_dataset.py
```

Rigenerazione completa:

```bash
python scripts/generate_dataset.py \
  --train 10000 \
  --valid 1000 \
  --test 1500 \
  --seed 42
```

Il test set contiene intenzionalmente:

- target non presenti nel train;
- nomi di prodotti inventati;
- negazioni;
- preferenze morbide;
- budget;
- luoghi;
- tempo;
- richieste ambigue;
- multi-goal.

Questo è essenziale: vogliamo misurare **generalizzazione strutturale**, non memoria.

Valida sempre dopo ogni generazione:

```bash
python scripts/validate_dataset.py
```

---

## 4. Formato training

Ogni riga dei file JSONL è una conversazione:

```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"{...GoalContract...}"}
]}
```

MLX viene eseguito con `mask_prompt: true`: il loss viene calcolato sulla risposta finale, non sul system/user prompt.

---

## 5. Aumentare automaticamente la varietà linguistica

La ground truth NON va affidata al teacher. Il teacher deve solo parafrasare la richiesta; il GoalContract originale resta invariato.

Script:

```text
scripts/augment_with_teacher.py
```

Funziona con qualunque endpoint OpenAI-compatible.

Esempio:

```bash
export TEACHER_BASE_URL="http://127.0.0.1:8080/v1"
export TEACHER_MODEL="nome-modello-teacher"
export TEACHER_API_KEY=""

python scripts/augment_with_teacher.py \
  --input data/train.jsonl \
  --output data/train_augmented.jsonl \
  --variants 3 \
  --limit 500
```

Prima di usare `train_augmented.jsonl`, controlla a campione che le parafrasi non abbiano modificato i vincoli.

### Strategia consigliata

Fase 1:

```text
3k-10k esempi deterministici
```

Fase 2:

```text
+ parafrasi teacher
```

Fase 3:

```text
+ casi reali Jase anonimizzati e corretti
```

Fase 4:

```text
error mining → nuovi esempi mirati → nuovo adapter
```

---

## 6. API locale Jase Interpreter

Dopo il training:

```bash
source .venv/bin/activate
python scripts/serve_mlx.py --adapter adapters/jase-v1
```

Endpoint:

```text
POST http://127.0.0.1:8001/interpret
```

Test:

```bash
curl -X POST http://127.0.0.1:8001/interpret \
  -H 'Content-Type: application/json' \
  -d '{"query":"Mi serve un elettricista a Milano stasera, massimo 100 euro"}'
```

La API rifiuta automaticamente output che non sono JSON validi o non rispettano lo schema.

---

## 7. Passaggio alla RTX 4080 Super

Per la GPU NVIDIA è incluso:

```text
scripts/train_nvidia.sh
```

Su Linux/WSL con CUDA:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-nvidia.txt
bash scripts/train_nvidia.sh
```

Il training NVIDIA usa `ms-swift` + LoRA standard su `Qwen/Qwen3.5-2B`.

Per un 2B e una 4080 Super da 16 GB partirei da LoRA standard, non QLoRA. Se necessario si può aggiungere BNB 4-bit più avanti.

---

## 8. Cosa NON fare

Non fare continual-learning automatico direttamente sulle richieste degli utenti.

Non aggiungere una richiesta al dataset solo perché l'esecuzione è riuscita.

Non lasciare che l'Interpreter scelga tool/API concreti.

Non mettere PII raw nel training set.

Non trasformare `target`, prodotti, professioni, razze, cibi, città ecc. in enum giganteschi.

Non giudicare il modello solo dal training loss: serve il test held-out.

---

## 9. Roadmap consigliata

### Milestone A — Proof of concept

- GoalContract v1 congelato
- 3k train
- baseline Qwen
- primo adapter MLX
- benchmark 100 esempi

### Milestone B — Generalizzazione

- 10k-30k train
- parafrasi
- slang/errori grammaticali
- inglese
- multi-goal
- negazioni
- entità inventate
- richieste incomplete

### Milestone C — Jase integration

```text
Interpreter
   ↓
Schema validator
   ↓
Entity resolver
   ↓
Capability resolver
   ↓
Planner
```

### Milestone D — Data flywheel

```text
Jase traffic
   ↓
anonymization
   ↓
failures / corrections
   ↓
reviewed gold data
   ↓
new training round
```

---

## 10. File principali

```text
Jase Interpreter
├── SYSTEM_PROMPT.txt
├── schema/
│   └── goal_contract_v1.schema.json
├── configs/
│   ├── mlx_lora.yaml
│   └── mlx_lora_lowmem.yaml
├── scripts/
│   ├── bootstrap_mac.sh
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   ├── infer_mlx.py
│   ├── eval_mlx.py
│   ├── augment_with_teacher.py
│   ├── serve_mlx.py
│   ├── train_mac.sh
│   ├── train_mac_lowmem.sh
│   └── train_nvidia.sh
├── data/
├── adapters/
└── outputs/
```

La prima cosa da fare è **non modificare ancora il modello**: genera il dataset, esegui il baseline su 50 esempi e salva i numeri. Poi allena l'adapter e confronta gli stessi 50/100 esempi.
