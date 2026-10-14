# SmartBOM BOM 分析核心

此套件由原本單一的 `connectivity_OR_bom.py` 拆分而來。公開入口是
`BomApplicationService`；舊檔案只保留相容匯入，不再承載業務邏輯。

## 模組責任

| 模組 | 單一責任 | 主要擴充點 |
|---|---|---|
| `config.py` | 集中參數與顏色映射 | `BomSettings` |
| `models.py` | 無流程依賴的領域資料 | `Geometry`、`EntityGroup`、`BomAnalysisResult` |
| `io.py` | DXF/DWG/CSV 載入與 CSV 輸出 | `VectorDataLoader`、`ReportWriter` Protocol |
| `geometry.py` | 幾何轉換、距離、聚類、去重 | `GeometryService` |
| `text.py` | 文字清理、規格解析、最近文字配對 | `TextSpecificationService` |
| `predicates.py` | 元件類型判斷 | 小型純函式 |
| `grouping.py` | 元件群組與鄰接圖 | `EntityGroupingService` |
| `optimizer.py` | Tube 配對及尺寸 ILP | `TubeAssignmentOptimizer` |
| `rules.py` | Gauge、Hose、Reducer、材質等領域規則 | `BomRuleEngine` |
| `reporting.py` | BOM 聚合與稽核明細 | `BomReportBuilder` |
| `review.py` | Reducer 分段、風險排序、人工修正與 BOM 重建 | `ConnectivityReviewService` |
| `visualization.py` | 原圖、連通塊覆蓋色與 hover 資訊 | `build_debug_figure` |
| `service.py` | 串聯完整使用案例 | `BomApplicationService` |

## SOLID 對應

- SRP：每個模組只處理一類變化，I/O、幾何、最佳化與報表彼此分離。
- OCP：新增 Loader、Writer 或服務實作時，不必改動應用流程。
- LSP：符合 Protocol 的 Loader／Writer 可直接替換預設實作。
- ISP：載入與輸出是兩個小介面，呼叫端不依賴不需要的方法。
- DIP：應用服務透過建構式接收依賴，測試可注入記憶體替身。

## 使用方式

推薦入口：

~~~python
from connectivity_bom_main import main

main("./data/10_單線圖_HALYLP2200_SG.dxf")
~~~

只分析、不寫檔：

~~~python
from smartbom_bom import BomApplicationService

service = BomApplicationService.create_default()
result = service.run("./data/10_單線圖_HALYLP2200_SG.dxf", write_reports=False)
print(result.bom)
~~~

自訂容許值：

~~~python
from smartbom_bom import BomApplicationService, BomSettings

settings = BomSettings(connection_tolerance=0.25, duplicate_tolerance=0.005)
service = BomApplicationService.create_default(settings)
result = service.run("drawing.dxf")
~~~

## 維護原則

1. 顏色與固定參數只放在 `config.py`。
2. 新文字格式只修改 `text.py`，不要讓報表模組解析字串。
3. 新工程元件規則放在 `rules.py`，不要混入幾何距離程式。
4. 更換求解器只修改或替換 `TubeAssignmentOptimizer`。
5. 新輸出格式實作 `ReportWriter`，不要改核心分析流程。
6. 人工覆核規則只放在 `review.py`；Gradio 不直接修改 `EntityGroup`。
7. 視覺樣式與 Plotly trace 只放在 `visualization.py`，不得反向影響分析結果。

## 人工覆核工作流

`connectivity_bom_main.py` 採用兩階段流程：

1. 上傳檔案後執行自動分析，但不立即發布 BOM。
2. 以 Reducer／Reducer-Union 為邊界建立連通塊。
3. 依文字證據差異數排序，並標示小型連通塊。
4. 使用者只修改連通塊的尺寸與規格。
5. `ConnectivityReviewService` 將修正值回寫至受連通塊控制的元件。
6. 固定 Gauge、Gauge Gasket、R.Tee 與 Hose 規則保持不變。
7. 重新計算 Reducer 尺寸、排除 `顏色_XXX` 元件後才建立最終 BOM。
