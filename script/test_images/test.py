#!/usr/bin/env python3
import json
import os
import argparse
from PIL import Image, ImageDraw, ImageFont


def load_json(path):
	with open(path, 'r') as f:
		return json.load(f)


def resolve_image_path(image_path, json_dir):
	if os.path.isabs(image_path):
		if os.path.exists(image_path):
			return image_path
	# try relative to json dir
	candidate = os.path.join(json_dir, os.path.basename(image_path))
	if os.path.exists(candidate):
		return candidate
	# fallback: join with json dir
	candidate2 = os.path.join(json_dir, image_path)
	if os.path.exists(candidate2):
		return candidate2
	raise FileNotFoundError(f'image not found: {image_path}')


def draw_boxes_on_image(image_path, infos, out_path=None):
	img = Image.open(image_path).convert('RGB')
	draw = ImageDraw.Draw(img)
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None

	# colors
	colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 165, 0)]

	i = 0
	for key in sorted(infos.keys()):
		if not key.startswith('info'):
			continue
		info = infos[key]
		bbox = info.get('bbox', [])
		score = info.get('score', 0)
		name = info.get('name', '')
		if not bbox or len(bbox) < 4:
			i += 1
			continue
		x1, y1, x2, y2 = bbox[:4]
		color = colors[i % len(colors)]
		# rectangle
		draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
		# label
		label = f"{name} {score:.2f}" if name else f"{score:.2f}"
		if font:
			try:
				text_size = font.getsize(label)
			except Exception:
				# fallback
				text_size = (len(label) * 6, 11)
		else:
			text_size = (len(label) * 6, 11)
		text_bg = [x1, max(0, y1 - text_size[1] - 4), x1 + text_size[0] + 4, y1]
		draw.rectangle(text_bg, fill=(0, 0, 0))
		draw.text((x1 + 2, max(0, y1 - text_size[1] - 2)), label, fill=color, font=font)
		i += 1

	if out_path is None:
		base, ext = os.path.splitext(image_path)
		out_path = f"{base}_boxed{ext}"
	img.save(out_path)
	return out_path


def main():
	parser = argparse.ArgumentParser(description='Draw boxes from info.json')
	parser.add_argument('--json', '-j', default='info.json', help='Path to info.json')
	args = parser.parse_args()

	json_path = args.json
	if not os.path.isabs(json_path):
		json_path = os.path.abspath(json_path)
	if not os.path.exists(json_path):
		raise SystemExit(f'json file not found: {json_path}')

	data = load_json(json_path)
	json_dir = os.path.dirname(json_path)
	image_path = data.get('image_path')
	if not image_path:
		raise SystemExit('image_path missing in json')
	image_path = resolve_image_path(image_path, json_dir)

	out = draw_boxes_on_image(image_path, data, out_path=None)
	print(f'Wrote annotated image: {out}')


if __name__ == '__main__':
	main()
