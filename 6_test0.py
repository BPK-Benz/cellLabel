import os
import cv2
import json
import numpy as np
import colorsys
from utils.make_coco import *

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# make color dictioanary
def make_colors():
	variation = 46
	hues = np.linspace(0, 1, variation+1)[:-1]
	colors = [[int(255*c) for c in colorsys.hsv_to_rgb(h, 1, 1)] for h in hues]
	return colors
colors = make_colors()

def load_coco(coco_path):
	f = open(coco_path)
	return json.load(f)

# category id -> [category of that category id]
def maps_categories(coco):
	cats = {}
	for cat in coco['categories']:
		cats[cat['id']] = cat
	return cats

# image id -> [anns of that image id]
def maps_annotations(coco):
	anns = {}
	for ann in coco['annotations']:
		if ann['image_id'] in anns:
			anns[ann['image_id']] += [ann]
		else:
			anns[ann['image_id']] = [ann]
	return anns

if __name__ == '__main__':


	config = '../../configs/mask_rcnn/mask_rcnn_r50_fpn_1x_cell.py'
	checkpoint = '../../work_dirs/mask_rcnn_r50_fpn_1x_cell/latest.pth'
	model = init_detector(config, checkpoint, device='cuda:0')

	# make path
	root = ''
	coco_path = os.path.join(root, "test.json")

	# init parameters
	if os.path.exists(coco_path):
		coco = load_coco(coco_path)  
	else:
		print('Error: COCO file not exists.')
		exit()
	cats = maps_categories(coco)
	anns = maps_annotations(coco)
	total = len(coco["images"])
	index = 0

	# main loop
	while True:

		# get image data
		image_id = coco["images"][index]["id"]
		image_path = root + coco["images"][index]["file_name"]
		image = cv2.imread(image_path)

		# # draw ground truth
		# for ann in anns[image_id]:

		# 	# get label data
		# 	bbox = ann["bbox"]
		# 	category_id = ann["category_id"]
		# 	name = cats[category_id]['name']

		# 	# show bbox and image
		# 	x, y, w, h = bbox
		# 	# color = colors[category_id]
		# 	color = (0, 255, 0)
		# 	cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
		# 	cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


		# draw detected
		img = mmcv.imread(image_path)
		results, masks = inference_detector(model, img)
		for i in range(3):
			result = results[i]
			category_id = i + 1
			for box in result:
				x1, y1, x2, y2, c = box
				if c < 0.3: continue
				x1, y1, x2, y2 = [int(j) for j in [x1, y1, x2, y2]]
				name = cats[category_id]['name']
				# color = colors[category_id]
				color = (0, 0, 255)
				color = (0, 255, 0)
				cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
				cv2.putText(image, name, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

		# display
		cv2.imshow('display', image)
		output_path = 'data/output_test/' + str(index) + '.png'
		cv2.imwrite(output_path, image)

		# user feedback
		key = cv2.waitKey(0)
		if key in [ord('q')]:
			break
		elif key in [ord("a")]: 
			index = np.clip(index-1, 0, total-1)
		elif key in [ord("d")]: 
			index = np.clip(index+1, 0, total-1)
		elif key in [ord("f")]: 
			index = np.clip(index+50, 0, total-1)
