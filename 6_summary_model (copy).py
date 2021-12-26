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

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

if __name__ == "__main__":

	annotations = [
		{
			'name': 'groundtruth',
			'type': 'label',
			'dataset': [
				'S1/Plate_03/Testing_set',
			],
			'label_path': 'groundtruth.json',
		},
		{
			'name': 'image_processing',
			'type': 'label',
			'dataset': [
				'S1/Plate_03/Testing_set',
			],
			'label_path': 'image_processing.json',
		},
		{
			'name': 'faster-rcnn resnet101 2-classes',
			'type': 'model',
			'dataset': [
				'S1/Plate_03/Testing_set',
			],
			'model_path': 'source/latest.pth',
			'config_path': '../../configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell_2Class_GT.py',
		},
	]

	gts = load_coco(annotations[0]['label_path'])
	gts_maps = read_anno(gts)

	# p1s = load_coco(annotations[1]['label_path'])
	# p1s_maps = read_anno(p1s)

	config = annotations[2]['config_path']
	checkpoint = annotations[2]['model_path']
	model = init_detector(config, checkpoint, device='cuda:0')
	print('[ Finished Load Model ]')

	threshold_iou = 0.5
	scale = 3/4
	confident = 0.5

	images = gts['images']
	total = len(images)

	scores = {
		'border': {
				0: { 'tp': 0, 'fp': 0, 'fn': 0 },
				1: { 'tp': 0, 'fp': 0, 'fn': 0 },
		},
		'divide': {
				0: { 'tp': 0, 'fp': 0, 'fn': 0 },
				1: { 'tp': 0, 'fp': 0, 'fn': 0 },
		},
		'infect': {
				'non-infected': { 'tp': 0, 'fp': 0, 'fn': 0 },
				'cytocell': { 'tp': 0, 'fp': 0, 'fn': 0 },
				'nuccell': { 'tp': 0, 'fp': 0, 'fn': 0 },
		},
	}

	# loop for each images
	for index in range(total):

		print('[ Processing {} of {} | {} ]'.format(index, total, gts['images'][index]['file_name']))

		# get groundtruth labels of 1 image
		gt = []
		if gts['images'][index]['id'] in gts_maps:
			for j in gts_maps[gts['images'][index]['id']]:
				gt.append(gts['annotations'][j])

		# # get image processing labels of 1 image
		# p1 = []
		# if p1s['images'][index]['id'] in p1s_maps:
		# 	for j in p1s_maps[p1s['images'][index]['id']]:
		# 		p1.append(p1s['annotations'][j])

		# load image
		image_path = gts['images'][index]['file_name']
		cv2image = cv2.imread(image_path)

		image = mmcv.imread(image_path)
		image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
		results = inference_detector(model, image)
		map_features = {
			0: 1, 
			1: 0
		}
		
		# loop for each features
		for feature in scores:

			if feature != 'divide': continue

			# loop for each category
			for category in scores[feature]:

				filtered_gt = [o for o in gt if o[feature] == category]
				filtered_p1 = [o/scale for o in results[map_features[category]] if o[4] > confident]

				matched_gt = []
				matched_p1 = []

				for i in range(len(filtered_gt)):
					bbox1 = {
						'x1': filtered_gt[i]['bbox'][0],
						'y1': filtered_gt[i]['bbox'][1],
						'x2': filtered_gt[i]['bbox'][0] + filtered_gt[i]['bbox'][2],
						'y2': filtered_gt[i]['bbox'][1] + filtered_gt[i]['bbox'][3],
					}
					for j in range(len(filtered_p1)):
						bbox2 = {
							'x1': filtered_p1[j][0],
							'y1': filtered_p1[j][1],
							'x2': filtered_p1[j][2],
							'y2': filtered_p1[j][3],
						}
						iou = get_iou(bbox1, bbox2)
						if iou > threshold_iou:
							matched_gt += [i]
							matched_p1 += [j]

				tps = set(matched_gt)
				fps = [j for j in range(len(filtered_p1)) if not j in matched_p1]
				fns = [i for i in range(len(filtered_gt)) if not i in matched_gt]
				scores[feature][category]['tp'] += len(tps)
				scores[feature][category]['fp'] += len(fps) + len(matched_gt) - len(tps)
				scores[feature][category]['fn'] += len(fns)

				# # double check visualization: start
				# if category != 1: continue

				# for i in range(len(filtered_gt)):
				# 	x1 = filtered_gt[i]['bbox'][0]
				# 	y1 = filtered_gt[i]['bbox'][1]
				# 	x2 = filtered_gt[i]['bbox'][0] + filtered_gt[i]['bbox'][2]
				# 	y2 = filtered_gt[i]['bbox'][1] + filtered_gt[i]['bbox'][3]
				# 	color = (0, 150, 255)
				# 	cv2.rectangle(cv2image, (x1, y1), (x2, y2), color, 3)

				# for j in range(len(filtered_p1)):
				# 	x1 = int(filtered_p1[j][0])
				# 	y1 = int(filtered_p1[j][1])
				# 	x2 = int(filtered_p1[j][2])
				# 	y2 = int(filtered_p1[j][3])
				# 	c = str(filtered_p1[j][4])[:5]
				# 	color = (255, 150, 0)
				# 	cv2.rectangle(cv2image, (x1, y1), (x2, y2), color, 2)
				# 	cv2.putText(cv2image, c, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 1, cv2.LINE_AA)

				# cv2image = cv2.resize(cv2image, (0, 0), fx=scale, fy=scale)
				# cv2.imshow('cv2image', cv2image)
				# key = cv2.waitKey(0)
				# if key in [ord('q'), 27]: exit()
				# # double check visualization: end

	# loop for each features
	for feature in scores:

		print('-'*80)
		print(feature)

		# loop for each category
		for category in scores[feature]:

			count_tp = scores[feature][category]['tp']
			count_fp = scores[feature][category]['fp']
			count_fn = scores[feature][category]['fn']
			precision = count_tp / (count_tp + count_fp) if count_tp else 0
			recall = count_tp / (count_tp + count_fn) if count_tp else 0
			f1 = 2 * precision * recall / (precision + recall) if precision else 0

			print('\t', category)
			print('\t\tTrue Positive  =', count_tp)
			print('\t\tFalse Positive =', count_fp)
			print('\t\tFalse Negative =', count_fn)
			print('\t\tPrecision      =', round(precision, 3))
			print('\t\tRecall         =', round(recall, 3))
			print('\t\tF1 score       =', round(f1, 3))
