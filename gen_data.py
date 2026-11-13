"""使用 symbols/ 圖示產生可選的 YOLO 合成訓練資料。"""

from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFont


SYMBOL_DIRECTORY = Path("symbols")
OUTPUT_DIRECTORY = Path("synth_dataset")
IMAGE_COUNT = 1000
IMAGE_SIZE = (800, 800)


def _draw_background_noise(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    """加入管線、格線與文字，模擬工程圖背景。"""
    for _ in range(random.randint(15, 30)):
        start_x, start_y = random.randint(0, width), random.randint(0, height)
        length = random.randint(50, 300)
        angle = math.radians(random.choice([0, 30, 90, 150, 210, 270, 330]))
        end = (int(start_x + length * math.cos(angle)), int(start_y + length * math.sin(angle)))
        draw.line(
            [(start_x, start_y), end],
            fill=random.choice([(255, 165, 0), (150, 150, 150), (100, 200, 255)]),
            width=random.randint(1, 3),
        )

    if width >= 300 and height >= 300 and random.random() > 0.5:
        grid_x = random.randint(100, width - 200)
        grid_y = random.randint(100, height - 200)
        for index in range(random.randint(3, 8)):
            draw.line([(grid_x, grid_y + index * 20), (grid_x + 200, grid_y + index * 20)], fill=(200, 200, 200))
            draw.line([(grid_x + index * 40, grid_y), (grid_x + index * 40, grid_y + 150)], fill=(200, 200, 200))

    font = ImageFont.load_default()
    labels = ["FL+4350", "D.V", "C.V", '1/2" PN2 Tube', "3185mm", "QTY: 5"]
    for _ in range(random.randint(10, 20)):
        draw.text(
            (random.randint(0, width), random.randint(0, height)),
            random.choice(labels),
            fill=random.choice([(0, 0, 0), (50, 50, 50)]),
            font=font,
        )


def generate_dataset(
    symbol_directory: Path = SYMBOL_DIRECTORY,
    output_directory: Path = OUTPUT_DIRECTORY,
    image_count: int = IMAGE_COUNT,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> None:
    """產生合成圖片及同名 YOLO 標註檔。"""
    symbols = sorted(symbol_directory.glob("*.png"))
    if not symbols:
        raise FileNotFoundError(f"找不到元件圖示：{symbol_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    class_map = {path.name: index for index, path in enumerate(symbols)}
    width, height = image_size

    for image_index in range(image_count):
        canvas = Image.new("RGB", image_size, "white")
        _draw_background_noise(ImageDraw.Draw(canvas), width, height)
        canvas = canvas.convert("RGBA")
        annotations = []

        for _ in range(random.randint(10, 25)):
            symbol_path = random.choice(symbols)
            with Image.open(symbol_path) as source:
                symbol = source.convert("RGBA")
            symbol = symbol.rotate(random.choice([0, 30, 90, 150, 180, 210, 270, 330]), expand=True)
            scale = random.uniform(0.4, 0.9)
            symbol = symbol.resize(
                (max(1, int(symbol.width * scale)), max(1, int(symbol.height * scale))),
                Image.Resampling.LANCZOS,
            )
            x = random.randint(0, width - symbol.width)
            y = random.randint(0, height - symbol.height)
            canvas.paste(symbol, (x, y), symbol)
            annotations.append(
                f"{class_map[symbol_path.name]} "
                f"{(x + symbol.width / 2) / width:.6f} {(y + symbol.height / 2) / height:.6f} "
                f"{symbol.width / width:.6f} {symbol.height / height:.6f}"
            )

        stem = f"synth_{image_index:04d}"
        canvas.convert("RGB").convert("L").convert("RGB").save(output_directory / f"{stem}.jpg")
        (output_directory / f"{stem}.txt").write_text("\n".join(annotations), encoding="utf-8")


def main() -> None:
    generate_dataset()
    print(f"完成：{OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
