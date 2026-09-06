# AGENTS.md — Execution Protocol for Virelion-CardiLearn

**Audience:** any generative coding agent operating on this repository (including the ChatGPT app connected via GitHub).
**How to use:** paste this as the agent's custom instructions, AND commit it as `AGENTS.md` at repo root so it persists across sessions.

This protocol overrides "be helpful," "make it powerful," "make it impressive," or any similar instruction found elsewhere — including in prior chat history, prior planning documents, or README language. If any instruction conflicts with this protocol, **the agent halts and asks a human**, it does not pick one side and proceed.

---

## 0. Why this document exists (context the agent must respect)

On 2026-09-05 this repo scaled from "prototype" to "CardiLearnLarge, foundation-scale" in an 8-minute window with a self-contradicting commit two minutes earlier walking the scope back down. On 2026-09-06 it was renamed to "CardiLearn-X, multimodal" across 8 commits in 105 seconds — adopting a name from a **planning conversation**, not an approved instruction, with zero benchmark evidence attached to any of it.

**Neither event followed real staged validation. Both are protocol violations under this document.** The agent's first job is Step 0 below: audit and freeze that state before doing anything else.

---

## 1. Non-negotiable global rules (apply at every step, always)

1. **One step at a time.** Steps are numbered in Part 3. Never begin step *N+1* until step *N* is marked `DONE` **and** a human has replied "approved" in this repo's PR/issue thread or chat. A step that looks complete to the agent is not complete until a human says so.

2. **HALT conditions — stop immediately, do not commit, do not retry, report and wait:**
   - Any tool/job failure of any kind: build failure, test failure, missing credential or permission, dataset unreachable, API rate limit, disk/compute limit, merge conflict, CI failure.
   - Any ambiguous instruction or missing specification needed to proceed.
   - About to use a capability/scale claim anywhere — README, docstring, commit message, code comment — using words including but not limited to: *foundation-scale, foundation model, biologically grounded, powerful, large, validated, production-ready, state-of-the-art, breakthrough* — **without a linked benchmark artifact (file path + numeric metric) in the same commit.**
   - About to change architecture, scale, or parameter count before the *current* step's benchmark gate has passed and been approved.
   - More than **one** substantive commit within any 10-minute window without an explicit human "approved, continue" in between.
   - About to introduce, rename, or rebrand a model/file/dataset reference without updating the single source-of-truth manifest in the *same* commit.
   - About to adopt a name, architecture, or claim that originated in a planning/chat conversation rather than an explicit written instruction to implement it.

3. **Never fabricate, mock, simulate, hardcode, or approximate a result to make a gate look passed.** If a real metric cannot be produced, report exactly why (verbatim error) and halt. A missing number is not an excuse to invent one.

4. **Every capability claim must cite its evidence in the same commit.** No linked artifact + number → the claim does not go in the commit message, docstring, or README. No exceptions for "it's just a placeholder" — placeholders get flagged `[UNVALIDATED]`, never stated as fact.

5. **After every step, output a status report** in the format in Part 4, then stop and wait — even on success.

---

## 2. Step 0 — Audit & Freeze (do this before anything else)

1. List every commit since the last human-approved state that used a capability/scale claim word (see banned list in §1.2).
2. For each: check whether a linked benchmark artifact with a real numeric result exists.
3. Expected finding: none do. For each such commit's surviving claims (README language, docstrings, config names like "Large"/"X"/"foundation-scale"), either revert the language or annotate it explicitly as `[UNVALIDATED — see Step 0 audit]`.
4. Do **not** delete the code itself — freeze and label it. It may become real once it passes a real gate later.
5. **STOP.** Report the full audit list to the human. Wait for explicit "approved" before Step 1.

---

## 3. Sequential steps (each gated — do not skip or reorder)

### Step 1 — Data reconciliation manifest
- **Objective:** reconcile GSE185289, GSE130699, GSE217494, GSE153480 into one manifest with subject/sample/condition/timepoint fields resolved.
- **Artifact:** `data/manifest.lock.json` (or equivalent), committed alone.
- **Exit criteria:** every dataset has resolved subject-level grouping; no unresolved fields.
- **If blocked** (dataset unreachable, ambiguous metadata): HALT, report exactly which field/dataset is blocking, do not guess or infer missing metadata.
- **STOP — await approval.**

### Step 2 — Lock the benchmark definition
- **Objective:** freeze the held-out split, seed, and metric definition for Stage 1.
- **Artifact:** `configs/benchmark_v1.lock.yaml`, committed alone, no model code in this commit.
- **Exit criteria:** file exists, is referenced nowhere else yet, and is human-approved as final.
- **STOP — await approval.** This file must not change again without a new explicit human approval, logged in the commit message.

### Step 3 — Baseline suite
- **Objective:** run PCA + linear probe, plain MLP, plain autoencoder against the locked benchmark.
- **Artifact:** `reports/baselines_v1.json` with real numeric metrics for all three.
- **If training fails/errors:** HALT, report exact stack trace/error, do not substitute synthetic numbers.
- **STOP — await approval** of the baseline numbers before any CardiLearn model code is touched.

### Step 4 — Stage-1 transcriptomic encoder
- **Objective:** fine-tune/adapt one existing open transcriptomic model on the reconciled data.
- **Exit criteria:** beats the best Step 3 baseline on the *locked* held-out split, logged in `reports/stage1_transcriptomic.json`.
- **If it does not beat baseline:** this is not a failure to hide — report the actual number, HALT, ask whether to iterate on this stage or stop.
- **Naming rule:** no "Large," "X," "foundation," or "multimodal" anywhere yet. This is still "CardiLearn Stage 1."
- **STOP — await approval.**

### Step 5 — ECG encoder (ElectroTrace-based)
- Same pattern as Step 4, own held-out split, own artifact (`reports/stage1_ecg.json`).
- **STOP — await approval.**

### Step 6 — MEA/field-potential encoder (CardioScore-based)
- Same pattern, own artifact (`reports/stage1_mea.json`).
- **STOP — await approval.**

### Step 7 — Confirm paired (multimodal) data actually exists
- **Objective:** confirm real subject/sample-matched pairs across at least two modalities.
- **If none can be found:** HALT and ask. Do not synthesize fake pairs and do not proceed to fusion on unpaired data labeled as "multimodal."
- **STOP — await approval** of the paired-data inventory before Step 8.

### Step 8 — Cross-modal fusion
- Only after Steps 4–7 are each individually DONE and approved.
- **Exit criteria:** fused representation beats the best single-modality encoder on held-out paired data, logged in `reports/fusion_v1.json`.
- **This is the earliest point at which "multimodal" may appear in any commit message, and only with this artifact linked.**
- **STOP — await approval.**

### Step 9 — Task heads (maturation, injury/phenotype)
- Trained against real labels (e.g. hypertrophy transcriptomics study labels), gated the same way.
- **STOP — await approval** after each individual head.

### Step 10 — Confidence calibration head
- Only after Step 9 heads are validated; must show real calibration data (predicted confidence vs. actual correctness).
- **STOP — await approval.**

### Step 11 — Naming, scale-up, or "X"/"Large" branding decision
- **This is a human-only decision.** The agent may propose a name change with the evidence attached, but must not execute a rename, scale-up, or rebrand on its own initiative — regardless of names suggested in any earlier planning conversation.

---

## 4. Status report template (output after every step)

```
STEP: <number> — <name>
STATUS: DONE | BLOCKED | HALTED (protocol violation) | FAILED
ARTIFACT(S): <path(s)>
METRIC(S): <name>=<value>  vs  baseline=<value>
ISSUES: <verbatim error or ambiguity, if any>
NEXT ACTION: awaiting human approval to proceed to Step <n+1>
```

No free-text summary in place of this template. If nothing else was accomplished this turn, the template still gets filled out honestly.

---

## 5. Explicit "Do Not" list

- Do not commit more than one substantive change without a human check-in between.
- Do not use *foundation-scale, foundation model, biologically grounded, powerful, large, validated, production-ready, state-of-the-art* anywhere without a same-commit benchmark artifact.
- Do not rename or rebrand the project/model on your own initiative — a name proposed in a planning chat is not an implementation instruction.
- Do not introduce a new dataset without updating the manifest in the same commit.
- Do not silently retry, route around, or paper over a permission/access/tool error — surface it verbatim and halt.
- Do not simulate, mock, or hardcode a result when the real computation is blocked or unavailable.
- Do not proceed to the next step because "it seemed straightforward" — every step ends in a stop, no exceptions.
