import cv2
import os
import glob
import numpy as np
import random

# Paths
img_dir = "./sample/images"
label_dir = "./sample/labels"
out_img_dir = "./result/images"
out_label_dir = "./result/labels"

custom_bg_dir = "./background/sample-background.JPG"
canvas_size = 640

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

all_images = glob.glob(os.path.join(img_dir, "*.jpg"))

def load_yolo_labels(label_path):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                boxes.append([int(cls), x, y, w, h])
    return boxes

def save_yolo_labels(label_path, boxes):
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def crop_object(img, box):
    """Crop object tightly from YOLO box"""
    h, w = img.shape[:2]
    cls, x, y, bw, bh = box
    abs_x = int((x - bw/2) * w)
    abs_y = int((y - bh/2) * h)
    abs_w = int(bw * w)
    abs_h = int(bh * h)
    cropped = img[abs_y:abs_y+abs_h, abs_x:abs_x+abs_w]
    return cropped, cls

def create_packed_mosaic(idx, max_objects=30, target_objects=50):
    chosen_imgs = random.sample(all_images, min(max_objects, len(all_images)))
    bg_images = glob.glob(custom_bg_dir)

    def get_background():
        if bg_images:
            bg_path = random.choice(bg_images)
            bg = cv2.imread(bg_path)
            bg = cv2.resize(bg, (canvas_size, canvas_size))
            return bg
        else:
            # fallback: white canvas if no bg found
            return np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    
    canvas = get_background()
    new_boxes = []

    x_cursor, y_cursor = 0, 0
    row_height = 0

    for img_path in chosen_imgs:
        lbl_path = os.path.join(label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        img = cv2.imread(img_path)
        boxes = load_yolo_labels(lbl_path)
        if not boxes:
            continue

        box = random.choice(boxes)
        obj, cls = crop_object(img, box)
        if obj.size == 0:
            continue

        h, w = obj.shape[:2]
        # --- Normalize object size ---
        scale = target_objects / max(h, w)       # make the largest side = TARGET_OBJECT
        scale *= random.uniform(0.9, 1.1)     # small random jitter
        new_w, new_h = int(w * scale), int(h * scale)
        obj_resized = cv2.resize(obj, (new_w, new_h))

        # If object doesn't fit horizontally, move to next row
        if x_cursor + new_w > canvas_size:
            x_cursor = 0
            y_cursor += row_height
            row_height = 0

        # If canvas is full, stop
        if y_cursor + new_h > canvas_size:
            break

        # Paste object
        canvas[y_cursor:y_cursor+new_h, x_cursor:x_cursor+new_w] = obj_resized

        # Label relative to canvas
        rel_x = (x_cursor + new_w/2) / canvas_size
        rel_y = (y_cursor + new_h/2) / canvas_size
        rel_bw = new_w / canvas_size
        rel_bh = new_h / canvas_size
        new_boxes.append([cls, rel_x, rel_y, rel_bw, rel_bh])

        # Update cursor
        x_cursor += new_w
        row_height = max(row_height, new_h)

    # Save outputs
    out_img_path = os.path.join(out_img_dir, f"packed_{idx}.jpg")
    out_lbl_path = os.path.join(out_label_dir, f"packed_{idx}.txt")

    cv2.imwrite(out_img_path, canvas)
    save_yolo_labels(out_lbl_path, new_boxes)

    print(f"✅ Saved {out_img_path} with {len(new_boxes)} objects")

# Progressive loop for packed mosaics
def run_packed(total=1000, step=100, start=50, decrement=10):
    """
    total      = total mosaics to generate
    step       = change parameters every this many images
    start      = initial target_objects size
    decrement  = how much to decrease target_objects each step
    """
    target_objects = start
    for i in range(total):
        # every 'step' images, increase target_objects
        if i % step == 0 and i != 0:
            target_objects -= decrement
        create_packed_mosaic(i, max_objects=1000, target_objects=target_objects)

# Example usage:
# Run 1000 packed mosaics (target objects grows: 50 → 60 → 70 …)
run_packed(total=5, step=1, start=200, decrement=30)

# # Example: Generate 10 packed mosaics
# for i in range(10):
#     create_packed_mosaic(i, max_objects=100, target_objects=85)
