import os
import random
from PIL import Image, ImageDraw, ImageFont
import math

# --- 設定區 ---
SYMBOL_DIR = "symbols"        
OUTPUT_DIR = "synth_dataset"  
NUM_IMAGES = 1000              
IMG_SIZE = (800, 800)         


os.makedirs(OUTPUT_DIR, exist_ok=True)


symbol_files = sorted([f for f in os.listdir(SYMBOL_DIR) if f.endswith('.png')])
class_map = {name: i for i, name in enumerate(symbol_files)}

# ==========================================
# 輔助函數：畫隨機干擾背景 (Noise)
# ==========================================
def draw_background_noise(draw, width, height):
    for _ in range(random.randint(15, 30)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        length = random.randint(50, 300)
        angle_deg = random.choice([0, 30, 90, 150, 210, 270, 330])
        angle_rad = math.radians(angle_deg)
        x2 = int(x1 + length * math.cos(angle_rad))
        y2 = int(y1 + length * math.sin(angle_rad))
        
        color = random.choice([(255, 165, 0), (150, 150, 150), (100, 200, 255)])
        thickness = random.randint(1, 3)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)


    if random.random() > 0.5: 
        grid_start_y = random.randint(100, height - 200)
        grid_start_x = random.randint(100, width - 200)
        for i in range(random.randint(3, 8)):
            draw.line([(grid_start_x, grid_start_y + i*20), (grid_start_x + 200, grid_start_y + i*20)], fill=(200, 200, 200), width=1)
            draw.line([(grid_start_x + i*40, grid_start_y), (grid_start_x + i*40, grid_start_y + 150)], fill=(200, 200, 200), width=1)

    font = ImageFont.load_default()
    for _ in range(random.randint(10, 20)):
        x, y = random.randint(0, width), random.randint(0, height)
        text = random.choice(["FL+4350", "D.V", "C.V", "1/2\" PN2 Tube", "3185mm", "QTY: 5"])
        draw.text((x, y), text, fill=random.choice([(0,0,0), (50,50,50)]), font=font)


# ==========================================
# 主生成迴圈
# ==========================================
for img_idx in range(NUM_IMAGES):
    canvas = Image.new("RGB", IMG_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    

    draw_background_noise(draw, IMG_SIZE[0], IMG_SIZE[1])
    
    labels = []
    canvas_rgba = canvas.convert("RGBA")

    for _ in range(random.randint(10, 25)):
        s_name = random.choice(symbol_files)
        s_img = Image.open(os.path.join(SYMBOL_DIR, s_name)).convert("RGBA")
        
        angle = random.choice([0, 30, 90, 150, 180, 210, 270, 330])
        s_img = s_img.rotate(angle, expand=True) 
        
        scale = random.uniform(0.4, 0.9) 
        new_size = (int(s_img.width * scale), int(s_img.height * scale))
        
        if new_size[0] == 0 or new_size[1] == 0: continue
            
        s_img = s_img.resize(new_size, Image.Resampling.LANCZOS)

        x = random.randint(0, IMG_SIZE[0] - s_img.width)
        y = random.randint(0, IMG_SIZE[1] - s_img.height)

        canvas_rgba.paste(s_img, (x, y), s_img)

        x_center = (x + s_img.width / 2) / IMG_SIZE[0]
        y_center = (y + s_img.height / 2) / IMG_SIZE[1]
        w_norm = s_img.width / IMG_SIZE[0]
        h_norm = s_img.height / IMG_SIZE[1]
        
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        labels.append(f"{class_map[s_name]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    final_image = canvas_rgba.convert("RGB")

    final_image = final_image.convert("L").convert("RGB")

    # 儲存結果
    base_name = f"synth_{img_idx:04d}"
    final_image.save(os.path.join(OUTPUT_DIR, f"{base_name}.jpg"))
    with open(os.path.join(OUTPUT_DIR, f"{base_name}.txt"), "w") as f:
        f.write("\n".join(labels))

print(f"完成！請到 {OUTPUT_DIR} 資料夾查看資料。")