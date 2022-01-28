import os
import cv2
import json
import numpy as np

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def load_coco(coco_path):
    with open(coco_path) as file:
        coco = json.load(file)
    return coco

def save_coco(coco_path, coco):
    with open(coco_path, 'w') as outfile:
        json.dump(coco, outfile)

def read_anno(coco):
	maps = {}
	for i, annotation in enumerate(coco['annotations']):
		image_id = annotation['image_id']
		if not image_id in maps:
			maps[image_id] = [i]
		else:
			maps[image_id].append(i)
	print('[ Finished Read Annotations ]')
	return maps

def check_border(x, y, w, h, img_w, img_h):
	margin = 5
	return int(
		x < margin or
		y < margin or
		x + w > img_w - margin or
		y + h > img_h - margin
	)

def condition(annotation=None):
    if not annotation:
        return [
            {
                "supercategory": 'cell_fused',
                "id": 1,
                "name": 'divide_cell',
            },
            {
                "supercategory": 'cell_fused',
                "id": 2,
                "name": 'not_divided_cell',
            },
        ]
    else:

        channel = annotation['channel']
        divide = annotation['divide']
        border = annotation['border']
        infect = annotation['infect']

        if channel == 'cell':
            if divide:
                return 1
            else:
                return 2
        else:
            return 0


if __name__ == "__main__":

	# load ground truth
	gts_path = 'groundtruth.json'
	gts = load_coco(gts_path)
	gts_maps = read_anno(gts)

	# make prediction
	pds_path = 'cnn_prediction.json'
	pds = gts.copy()
	pds['annotations'] = []

	# model
	config = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell_2Class_GT.py'
	checkpoint = 'source/latest.pth'
	model = init_detector(config, checkpoint, device='cuda:0')

	categories = condition()
	scale = 3/4
	count_annotation = 0
	images = gts['images']
	total = len(images)

	# loop for each images
	for index in range(total):

		print('[ Processing {} of {} | {} ]'.format(index, total, gts['images'][index]['file_name']))

		# load image
		image_path = gts['images'][index]['file_name']
		cv2image = cv2.imread(image_path)

		image = mmcv.imread(image_path)
		image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
		results = inference_detector(model, image)

		img_h = gts['images'][index]['height']
		img_w = gts['images'][index]['width']

		for c in range(len(results)):

			for result in results[c]:

				x, y, w, h, s = result
				x = int(result[0] / scale)
				y = int(result[1] / scale)
				w = int((result[2] - result[0]) / scale)
				h = int((result[3] - result[1]) / scale)
				s = str(result[4])
				pds['annotations'].append({
					'id': count_annotation,
					'image_id': gts['images'][index]['id'],
					'channel': 'cell',
					'divide': int(c == 1),
					'infect': '',
					'border': check_border(x, y, w, h, img_w, img_h),
					'bbox': [x, y, w, h],
					'category_id': c+1,
					'score': s
				})
				count_annotation += 1

	save_coco(pds_path, pds)
	print('[ Finish! ]')
