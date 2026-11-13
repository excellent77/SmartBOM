---
name: smartbom-solid-guard
description: Review and modify SmartBOM source code with project-specific SOLID boundaries. Use for every request that creates, edits, refactors, fixes, or reviews Python application code in this repository; run its pre-change design check and post-change static guard. Do not use for data-only CSV analysis or documentation-only edits.
---

# SmartBOM SOLID Guard

Apply this skill whenever application source code will be changed. It is a design gate, not a claim that SOLID can be proven mechanically.

## Required workflow

1. Before editing, read [references/architecture.md](references/architecture.md). Inspect the requested behavior, its current owner, callers, and tests. State briefly which module owns the change and which SOLID risks matter.
2. Prefer extending the existing owner or injecting a small dependency. Do not introduce a new class, interface, or file unless it isolates a real axis of change or creates a useful test seam.
3. During editing, keep UI, orchestration, domain rules, geometry, parsing, optimization, persistence, and visualization dependencies pointed in the directions defined by the architecture reference. Preserve user changes and public behavior outside the request.
4. After editing, run:

   ```bash
   python skills/smartbom-solid-guard/scripts/check_solid.py
   ```

   Also run focused behavioral tests and `python -m py_compile` for changed Python files. Fix every `ERROR`. Review every `WARN`; fix new warnings introduced by the change or explain why the tradeoff is appropriate.
5. In the handoff, report the affected responsibility, validation performed, and any deliberate SOLID tradeoff. Never say the result is “perfectly SOLID”; report evidence and residual risk.

## Decision rules

- SRP: place behavior with the reason it changes. Gradio collects input; services coordinate; rules decide BOM meaning; report builders aggregate; I/O reads and writes.
- OCP: prefer a settings map, predicate, protocol implementation, or injected strategy when new variants are expected. Do not build speculative abstractions for a single stable case.
- LSP: replacement implementations must preserve accepted inputs, outputs, errors, and side-effect expectations. Test through the consumer-facing contract.
- ISP: consumers depend only on the operation they use. Split an interface when unrelated implementers are forced to provide dummy methods.
- DIP: orchestration receives replaceable loaders, optimizers, rule engines, builders, and writers. Leaf domain functions may remain plain functions; dependency injection is not mandatory everywhere.

## Stop conditions

Stop and ask before a refactor if the required public API behavior is ambiguous, the change would rewrite unrelated dirty files, or the only apparent solution crosses multiple ownership boundaries without a stable contract.
