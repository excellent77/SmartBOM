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
| `batch.py` | 發現單線圖並批次產生報表 | `run_directory` |

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

批次處理 `./data/data_new` 內所有單線圖：

~~~bash
python connectivity_bom_batch.py
~~~

每張圖會在原資料夾產生 `<檔名>_bom.csv`、`<檔名>_details.csv` 與
`<檔名>_adjacency.csv`。也可在 Python 內指定其他資料夾：

~~~python
from smartbom_bom import run_directory

summary = run_directory("./data/data_new", progress=print)
print(summary)
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

1. 上傳檔案後先列出本次圖面所有元件顏色。
2. 使用者輸入元件名稱，並由下拉選單選擇 `Tube類／Reducer類／一般元件`；每列完成後才能分析。
3. 確認對照後才執行幾何分組、文字配對與鄰接分析；Tube類不參與一般元件配對，Reducer類作為尺寸連通塊邊界。
4. 依文字證據差異數排序，並標示小型連通塊。
5. 使用者只修改連通塊的尺寸與規格。
6. `ConnectivityReviewService` 將修正值回寫至受連通塊控制的元件。
7. Gauge 與 Gauge Gasket 維持 `1/4"`；Hose 維持長度文字規則。
8. R.Tee、Union R.Tee 與 VCR R.Tee 改由 `Reducer類` 統一處理，不再固定為 `Xx1/4"`。
9. Reducer類可連接任意多個方向，但所有方向必須剛好歸納為兩種尺寸，最終尺寸為這兩種尺寸的標準化組合。
