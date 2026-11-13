"""批次處理 data/data_new 內所有單線圖的執行入口。"""

from pathlib import Path

from smartbom_bom.batch import run_directory


DATA_DIRECTORY = Path("./data/data_new")


def main(directory: str | Path = DATA_DIRECTORY) -> None:
    """產生各單線圖對應的三份 CSV 報表，並列印執行摘要。"""
    summary = run_directory(directory, progress=print)
    succeeded = int((summary["狀態"] == "成功").sum())
    failed = int((summary["狀態"] == "失敗").sum())
    print(f"\n批次處理完成：成功 {succeeded} 份，失敗 {failed} 份。")
    if failed:
        columns = ["輸入檔案", "錯誤"]
        print(summary.loc[summary["狀態"] == "失敗", columns].to_string(index=False))


if __name__ == "__main__":
    main()
