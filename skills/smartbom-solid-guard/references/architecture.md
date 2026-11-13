# SmartBOM architecture boundaries

Use this reference before changing production Python code.

## Ownership map

| Area | Owner | Must not own |
|---|---|---|
| UI and user events | `connectivity_bom_main.py` | geometry, parsing, BOM business rules |
| Batch entry point | `connectivity_bom_batch.py`, `smartbom_bom/batch.py` | duplicated analysis rules |
| Use-case orchestration | `smartbom_bom/service.py` | Gradio components, DXF details, rule implementations |
| Configuration | `smartbom_bom/config.py` | workflow or I/O |
| Domain models | `smartbom_bom/models.py` | framework dependencies and file access |
| Geometry | `geometry.py`, `grouping.py` | text formats, UI, CSV reporting |
| Text parsing/matching | `text.py` | report writing and UI |
| Component classification | `predicates.py` | mutation-heavy workflow |
| Tube optimization | `optimizer.py` | presentation and persistence |
| BOM domain rules | `rules.py` | Gradio, Plotly, file access |
| Human review rules | `review.py` | Gradio widgets and drawing code |
| Aggregation/report data | `reporting.py` | file-system output |
| File input/output | `io.py`, `dxf_reader.py` | BOM policy |
| Visualization | `visualization.py` | mutation of analysis results |

`Test.py` and `systemv1/` are legacy/experimental unless the request explicitly targets them. Do not copy new production behavior into them automatically.

## Dependency direction

The intended flow is:

```text
UI / batch -> application service -> domain services and rules -> models/config
                                  -> loader/writer protocols -> I/O adapters
UI -> visualization (read-only result rendering)
```

Production package modules must not import UI entry points. Domain and application modules must not import Gradio or Plotly. Only the I/O boundary should import `dxf_reader`. Visualization may import Plotly but must not change domain state.

## Change checklist

Before editing:

- Identify the single owner of the requested behavior.
- Search all callers and construction sites.
- Decide whether the request adds a case, a policy, a replaceable strategy, or a new use case.
- Record behavior that must remain unchanged.

After editing:

- SRP: Does each changed unit still have one coherent reason to change?
- OCP: Can the next expected component type/format be added without another conditional scattered across modules?
- LSP: Can existing injected implementations still replace the changed abstraction?
- ISP: Did any consumer acquire methods or data it does not need?
- DIP: Did a high-level policy begin constructing or importing a low-level framework detail?
- Are errors raised at the boundary that can explain and recover from them?
- Do focused tests cover the new rule and a regression case?

## Acceptable tradeoffs

Small pure functions are preferred over ceremonial classes. A conditional is acceptable when the domain is closed and local. Duplication may be temporarily safer than a shared abstraction when two concepts only look similar; document the reason and revisit after a third concrete use case.
