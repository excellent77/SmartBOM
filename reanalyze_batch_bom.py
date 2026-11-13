"""重新比較 data_new 正確 BOM 與最新批次 `_bom.csv`。"""

from collections import defaultdict
import csv
from decimal import Decimal
from pathlib import Path
import re

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill


ROOT = Path("./data/data_new")
OUTPUT = ROOT / "BOM比較分析_重新批次"
FIELDS = ("品名", "尺寸", "材質", "數量")


def number(path: Path) -> int:
    return int(re.match(r"(\d+)_", path.name).group(1))


def clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def norm_name(value) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", clean(value).casefold())


def norm_size(value) -> str:
    parts = [
        re.sub(r"\s+", "", part).replace("″", '"').replace("”", '"').casefold()
        for part in re.split(r"[xX×]", clean(value)) if part.strip()
    ]
    return "x".join(sorted(parts)) if len(parts) > 1 else "".join(parts)


def norm_material(value) -> str:
    return clean(value).upper()


def quantity(value) -> Decimal:
    text = clean(value).upper().replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", re.sub(r"\s*M$", "", text))
    return Decimal(match.group()) if match else Decimal(0)


def read_answer(path: Path) -> list[dict]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    header_at = next(i for i, row in enumerate(rows) if all(field in row for field in FIELDS))
    header = list(rows[header_at])
    positions = {field: header.index(field) for field in FIELDS}
    result = []
    for row in rows[header_at + 1:]:
        if positions["品名"] >= len(row) or not clean(row[positions["品名"]]):
            continue
        result.append({field: row[index] if index < len(row) else None for field, index in positions.items()})
    workbook.close()
    return result


def read_prediction(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream) if clean(row.get("品名"))]


def aggregate(rows: list[dict], include_material: bool):
    totals = defaultdict(Decimal)
    display = {}
    for row in rows:
        key = (norm_name(row["品名"]), norm_size(row["尺寸"]))
        if include_material:
            key += (norm_material(row["材質"]),)
        totals[key] += quantity(row["數量"])
        display.setdefault(key, {field: clean(row[field]) for field in FIELDS[:3]})
    return dict(totals), display


def fmt(value) -> str:
    if value is None:
        return ""
    return str(int(value)) if value == value.to_integral() else format(value.normalize(), "f")


def compare_mode(answers: dict[int, Path], predictions: dict[int, Path], include_material: bool):
    summaries, details = [], []
    for drawing in sorted(answers):
        answer_rows = read_answer(answers[drawing])
        prediction_path = predictions.get(drawing)
        prediction_rows = read_prediction(prediction_path) if prediction_path else []
        expected, expected_display = aggregate(answer_rows, include_material)
        predicted, predicted_display = aggregate(prediction_rows, include_material)
        exact = missing = extra = mismatch = 0
        for key in sorted(set(expected) | set(predicted)):
            expected_quantity, predicted_quantity = expected.get(key), predicted.get(key)
            if expected_quantity is not None and predicted_quantity is not None:
                if expected_quantity == predicted_quantity:
                    exact += 1
                    continue
                kind, mismatch = "數量不符", mismatch + 1
            elif expected_quantity is not None:
                kind, missing = "AI缺少", missing + 1
            else:
                kind, extra = "AI多出", extra + 1
            reference = expected_display.get(key) or predicted_display.get(key)
            details.append({
                "圖號": drawing, "差異類型": kind,
                "品名": reference["品名"], "尺寸": reference["尺寸"],
                "材質": reference["材質"] if include_material else "(忽略)",
                "正確數量": fmt(expected_quantity), "AI數量": fmt(predicted_quantity),
                "正確答案內容": " | ".join(expected_display.get(key, {}).values()),
                "AI預測內容": " | ".join(predicted_display.get(key, {}).values()),
            })
        expected_count, predicted_count = len(expected), len(predicted)
        precision = exact / predicted_count if predicted_count else 0
        recall = exact / expected_count if expected_count else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        summaries.append({
            "圖號": drawing, "正確答案檔案": answers[drawing].name,
            "AI預測檔案": prediction_path.name if prediction_path else "",
            "狀態": "已比較" if prediction_path else "缺少AI預測",
            "正確答案列數": expected_count, "AI預測列數": predicted_count,
            "完全正確列數": exact, "差異列數": missing + extra + mismatch,
            "AI缺少列數": missing, "AI多出列數": extra,
            "數量不符列數": mismatch,
            "Precision": precision, "Recall": recall, "F1": f1,
        })
    return summaries, details


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_sheet(workbook: Workbook, title: str, rows: list[dict]) -> None:
    sheet = workbook.create_sheet(title)
    headers = list(rows[0])
    sheet.append(headers)
    for row in rows:
        sheet.append([row[header] for header in headers])
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width = min(
            max(len(str(cell.value or "")) for cell in column) + 2, 45
        )


def metrics(summaries: list[dict]) -> dict:
    compared = [row for row in summaries if row["狀態"] == "已比較"]
    expected = sum(row["正確答案列數"] for row in compared)
    predicted = sum(row["AI預測列數"] for row in compared)
    exact = sum(row["完全正確列數"] for row in compared)
    precision, recall = exact / predicted, exact / expected
    return {
        "可比較圖面": len(compared), "正確答案列": expected, "AI預測列": predicted,
        "完全正確列": exact,
        "AI缺少": sum(row["AI缺少列數"] for row in compared),
        "AI多出": sum(row["AI多出列數"] for row in compared),
        "數量不符": sum(row["數量不符列數"] for row in compared),
        "Precision": precision, "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall),
        "完全正確圖面": [row["圖號"] for row in compared if row["差異列數"] == 0],
    }


def main() -> None:
    answers = {number(path): path for path in ROOT.glob("*BOM表*.xlsx")}
    predictions = {number(path): path for path in ROOT.glob("*_bom.csv")}
    full_summary, full_details = compare_mode(answers, predictions, True)
    simple_summary, simple_details = compare_mode(answers, predictions, False)
    OUTPUT.mkdir(exist_ok=True)
    write_csv(OUTPUT / "完整規格_逐圖摘要.csv", full_summary)
    write_csv(OUTPUT / "完整規格_差異明細.csv", full_details)
    write_csv(OUTPUT / "忽略材質_逐圖摘要.csv", simple_summary)
    write_csv(OUTPUT / "忽略材質_差異明細.csv", simple_details)
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_sheet(workbook, "完整規格摘要", full_summary)
    add_sheet(workbook, "完整規格差異", full_details)
    add_sheet(workbook, "忽略材質摘要", simple_summary)
    add_sheet(workbook, "忽略材質差異", simple_details)
    workbook.save(OUTPUT / "重新批次BOM比較.xlsx")
    full_metrics, simple_metrics = metrics(full_summary), metrics(simple_summary)
    report = [
        "# 重新批次 BOM 比較", "", "## 完整規格（含材質）", "",
        *[f"- {key}：{value:.2%}" if key in {"Precision", "Recall", "F1"} else f"- {key}：{value}" for key, value in full_metrics.items()],
        "", "## 忽略材質", "",
        *[f"- {key}：{value:.2%}" if key in {"Precision", "Recall", "F1"} else f"- {key}：{value}" for key, value in simple_metrics.items()],
    ]
    (OUTPUT / "重新批次BOM比較.md").write_text("\n".join(report), encoding="utf-8")
    print("FULL", full_metrics)
    print("IGNORE_MATERIAL", simple_metrics)


if __name__ == "__main__":
    main()
