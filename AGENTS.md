# SmartBOM agent instructions

## Mandatory SOLID guard

For every task that creates, edits, refactors, fixes, or reviews Python application code in this repository, the main agent must use the project skill at:

`.agents/skills/smartbom-solid-guard/SKILL.md`

Read that file completely before making source-code changes and follow its pre-change, during-change, and post-change workflow. This requirement also applies to small fixes. It does not apply to data-only CSV comparisons or documentation-only edits.

Do not claim that SOLID compliance is mathematically guaranteed. Report the checks performed, new warnings, justified tradeoffs, and residual risks.
