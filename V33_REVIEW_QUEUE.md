# V33 gold review queue

V33_RAW is the frozen [blind panel](benchmarks/v33/blind.jsonl), SHA-256 `4818c5ea0b35aa0c5b6ff467ca1656b4e7313ed87d431975db18e2cb8b566f37`. Its 100 gold contracts were authored by an agent; none has been independently reviewed by a human. **V33_REVIEWED does not yet exist.** Scores in the existing reports use V33_RAW.

The [19-case review queue](benchmarks/v33_review/review_queue.jsonl) contains each flagged request, raw gold, unchanged Qwen V26 and Granite challenger outputs, the reason for review, and fields for an approved replacement and its impact on scoring and Planner semantics. Proposed reviewed gold remains `null` pending a human decision; ambiguity has not been converted into an arbitrary new label.

`scripts/prepare_v33_review.py` rebuilds the queue and first verifies the frozen V33 hash. It also normalizes one malformed `review_flags` value in the frozen metadata (`v33-047` is a string rather than an array) **only in the separate review artifact**. The raw panel and its queue remain unchanged.

After independent review, save decisions as a new version with the reviewer's identity and rationale. Preserve V33_RAW and report both scores separately; never overwrite old gold or recalculate historical metrics silently.
