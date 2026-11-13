# SmartBOM

SmartBOM 讀取 DXF、DWG 或既有向量 CSV，依幾何連接、顏色元件分類與文字規格產生配管 BOM，並提供 Gradio 人工覆核介面。

## 快速開始

環境需求：Python 3.11、`requirements.txt` 套件。讀取 DWG 時還需要系統安裝 `ODAFileConverter` 與 `xvfb-run`；DXF、CSV 不需要 ODA 轉檔。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python connectivity_bom_main.py
```

批次分析 `data/data_new`：

```bash
python connectivity_bom_batch.py
```

## 使用流程

1. 上傳 DXF、DWG 或向量 CSV。
2. 為本次圖面的每種顏色填寫元件名稱，並選擇 `Tube類`、`Reducer類` 或 `一般元件`。
3. 系統建立元件組、鄰接矩陣、文字配對與 Reducer 分段連通塊。
4. 在互動圖與覆核表確認每個連通塊的尺寸、規格。
5. 按下確認後輸出 `_bom.csv`、`_details.csv`、`_adjacency.csv`。

## 正式程式結構

```text
connectivity_bom_main.py       Gradio UI 與事件綁定
connectivity_bom_batch.py      批次執行入口
dxf_reader.py                  DXF/DWG 轉換與向量資料抽取
columns.py                     dxf_reader 共用欄位名稱
smartbom_bom/
  config.py                    常數、容許值、預設顏色映射
  models.py                    領域資料模型
  io.py                        載入器與 CSV writer
  geometry.py                  幾何轉換、距離、去重
  grouping.py                  元件聚類與鄰接圖
  text.py                      文字清理、規格解析與配對
  optimizer.py                 Tube 文字／尺寸整數規劃
  predicates.py                元件類型與固定規則判斷
  rules.py                     BOM 尺寸、材質、數量規則
  reporting.py                 BOM 與群組明細聚合
  review.py                    人工覆核與 Reducer 尺寸重建
  visualization.py             Plotly 互動圖
  service.py                   完整分析用例協調器
  batch.py                     批次檔案探索與執行
```

依賴方向固定為：

```text
UI／batch → BomApplicationService → domain services → models/config
                                 → loader/writer → dxf_reader/CSV
UI → visualization（唯讀呈現）
```

## 程式化使用

```python
from smartbom_bom import BomApplicationService, BomSettings

settings = BomSettings(connection_tolerance=0.25)
service = BomApplicationService.create_default(settings)
result = service.run("drawing.dxf", write_reports=True)

print(result.bom)
print(result.details)
print(result.adjacency)
```

公開入口集中在 `smartbom_bom/__init__.py`。新增功能時請優先擴充既有服務，不要從外部直接依賴底線開頭的內部函式。

## 維護規則

- 顏色、容許值與固定參數放在 `config.py`。
- 新文字格式放在 `text.py`；報表層不解析原始文字。
- Gauge、Hose、Reducer、材質與數量規則放在 `rules.py`。
- 人工修正與確認後重建規則放在 `review.py`。
- Gradio 只收集／呈現資料，不實作 BOM 領域規則。
- Plotly 樣式只放在 `visualization.py`，不得修改分析結果。
- 所有 Python 修改須遵循 `.agents/skills/smartbom-solid-guard/SKILL.md`。

## 驗證

```bash
python -m py_compile connectivity_bom_main.py connectivity_bom_batch.py dxf_reader.py smartbom_bom/*.py
python .agents/skills/smartbom-solid-guard/scripts/check_solid.py --all
```

`gen_data.py` 與 `symbols/` 是可選的影像辨識合成資料工具，不參與 SmartBOM 執行流程。
