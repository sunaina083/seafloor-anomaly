"""
tile_dataset.py

Slices full 640x640 sonar images into 80x80 tiles and labels each tile as
normal or anomalous from Roboflow YOLO-format bounding-box labels.
"""

import argparse
from pathlib import Path

from PIL import Image


DATASET_ROOT = Path("../../data/raw")
OUTPUT_ROOT = Path("../../data/tiles")
TILE_SIZE = 64
IMAGE_SIZE = 640
IGNORE_CLASS_IDS = {1}  # fish are treated as normal/background


def yolo_box_to_pixels(box, image_size=IMAGE_SIZE):
	"""Convert a normalized YOLO box to pixel coordinates."""
	class_id, x_center, y_center, width, height = box
	x_center *= image_size
	y_center *= image_size
	width *= image_size
	height *= image_size
	return (
		class_id,
		x_center - width / 2,
		y_center - height / 2,
		x_center + width / 2,
		y_center + height / 2,
	)



def overlap_fraction(tile_box, obj_box, tile_size=TILE_SIZE):
    """
    Compute what fraction of the TILE's area is covered by the object box.
    Returns a value from 0.0 (no overlap) to 1.0 (object fully covers tile).
    """
    tx_min, ty_min, tx_max, ty_max = tile_box
    ox_min, oy_min, ox_max, oy_max = obj_box

    inter_x_min = max(tx_min, ox_min)
    inter_y_min = max(ty_min, oy_min)
    inter_x_max = min(tx_max, ox_max)
    inter_y_max = min(ty_max, oy_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0  # no overlap at all

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    tile_area = tile_size * tile_size
    return inter_area / tile_area


MIN_OVERLAP_FRACTION = 0.40  # a tile must have at least 15% of its area covered
                               # by an anomaly box to count as "anomalous" —
                               # avoids over-labeling tiles that just barely
                               # brush a box edge

def load_labels(label_path):
	"""Read a YOLO label file into pixel-coordinate bounding boxes."""
	boxes = []
	if not label_path.exists():
		return boxes

	with label_path.open("r") as label_file:
		for line in label_file:
			parts = line.strip().split()
			if not parts:
				continue
			class_id = int(parts[0])
			x_center, y_center, width, height = map(float, parts[1:5])
			boxes.append(
				yolo_box_to_pixels((class_id, x_center, y_center, width, height))
			)
	return boxes


def tile_image(image_path, label_path, output_root, tile_size=TILE_SIZE):
	"""Tile one image, save each tile, and return normal/anomalous counts."""
	image = Image.open(image_path).convert("L")
	boxes = load_labels(label_path)
	anomaly_boxes = [box for box in boxes if box[0] not in IGNORE_CLASS_IDS]

	n_tiles_per_side = IMAGE_SIZE // tile_size
	normal_count = anomalous_count = 0

	for row in range(n_tiles_per_side):
		for col in range(n_tiles_per_side):
			x0, y0 = col * tile_size, row * tile_size
			tile_box = (x0, y0, x0 + tile_size, y0 + tile_size)
			is_anomalous = any(
                overlap_fraction(tile_box, (bx0, by0, bx1, by1)) >= MIN_OVERLAP_FRACTION
                for (_, bx0, by0, bx1, by1) in anomaly_boxes
            )
			label_folder = "anomalous" if is_anomalous else "normal"
			output_dir = output_root / label_folder
			output_dir.mkdir(parents=True, exist_ok=True)
			image.crop(tile_box).save(
				output_dir / f"{image_path.stem}_r{row}_c{col}.png"
			)
			if is_anomalous:
				anomalous_count += 1
			else:
				normal_count += 1

	return normal_count, anomalous_count


def main(split):
	images_dir = DATASET_ROOT / split / "images"
	labels_dir = DATASET_ROOT / split / "labels"
	output_dir = OUTPUT_ROOT / split

	if not images_dir.exists():
		raise FileNotFoundError(
			f"Couldn't find {images_dir} — check DATASET_ROOT and the dataset structure."
		)

	image_paths = sorted(images_dir.glob("*.jpg"))
	print(f"Found {len(image_paths)} images in {split} split.")
	total_normal = total_anomalous = 0
	for image_path in image_paths:
		normal, anomalous = tile_image(
			image_path, labels_dir / f"{image_path.stem}.txt", output_dir
		)
		total_normal += normal
		total_anomalous += anomalous

	total = total_normal + total_anomalous
	percentage = total_anomalous / total if total else 0
	print(
		f"Done. {split}: {total_normal} normal tiles, "
		f"{total_anomalous} anomalous tiles ({percentage:.1%} anomalous)."
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Tile sonar images and label them as normal or anomalous."
	)
	parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
	main(parser.parse_args().split)
