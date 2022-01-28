import os
import numpy as np
import cv2
import json
import time
import pandas as pd
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import colorsys
from scipy.spatial import distance

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

def checksum(scores):
	total = 0
	for name in scores:
		total += scores[name]['tp']
		total += scores[name]['fn']
	return total

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

def make_colors(variation):
    hues = np.linspace(0, 1, variation+1)[:-1]
    variation = [[int(255*c) for c in colorsys.hsv_to_rgb(h, 1, 1)] for h in hues]
    colors = [[0, 0, 0]]
    colors += variation
    colors = [c[::-1] for c in colors]
    return colors
colors = make_colors(len(condition()))


class cellEval: # managing folders for source and target 
	def __init__(self):

		# ground truth
		gts_path = 'groundtruth.json'
		self.gts = load_coco(gts_path)
		self.gts_maps = read_anno(self.gts)

		# image processing
		pds_path = 'cnn_prediction.json'
		self.pds = load_coco(pds_path)
		self.pds_maps = read_anno(self.pds)

		self.total = len(self.gts['images'])
		self.index = 0

		# window
		self.scale = 0.9
		self.mouse = 0, 0

		self.showing = {
			'gt': True,
			'pd': True,
		}

		# load first image
		self.load_image()

	def load_image(self):

		image_data = self.gts['images'][self.index]
		image_path = image_data['file_name']

		self.image_id = image_data['id']
		self.image = cv2.imread(image_path)

		self.print_eval()
		
		self.display()

	def print_eval(self):

		index = self.index
		gts = self.gts
		gts_maps = self.gts_maps
		pds = self.pds
		pds_maps = self.pds_maps

		categories = condition()
		scores = {}
		for c in categories:
			scores[c['name']] = { 'tp': 0, 'fp': 0, 'fn': 0 }
	
		threshold_iou = 0.5

		# get all sample
		gt = []
		if gts['images'][index]['id'] in gts_maps:
			for j in gts_maps[gts['images'][index]['id']]:
				d = gts['annotations'][j]
				d['class'] = condition(d)
				gt.append(d)

		# get all predict
		pd = []
		if pds['images'][index]['id'] in pds_maps:
			for j in pds_maps[pds['images'][index]['id']]:
				d = pds['annotations'][j]
				if 'score' in d:
					d['class'] = d['category_id']
				else:
					d['class'] = condition(d)
				pd.append(d)

		for c in categories:

			filtered_gt = [o for o in gt if o['class'] == c['id']]
			filtered_pd = [o for o in pd if o['class'] == c['id']]

			matched_gt = []
			matched_pd = []

			for i in range(len(filtered_gt)):
				bbox1 = {
					'x1': filtered_gt[i]['bbox'][0],
					'y1': filtered_gt[i]['bbox'][1],
					'x2': filtered_gt[i]['bbox'][0] + filtered_gt[i]['bbox'][2],
					'y2': filtered_gt[i]['bbox'][1] + filtered_gt[i]['bbox'][3],
				}
				for j in range(len(filtered_pd)):
					bbox2 = {
						'x1': filtered_pd[j]['bbox'][0],
						'y1': filtered_pd[j]['bbox'][1],
						'x2': filtered_pd[j]['bbox'][0] + filtered_pd[j]['bbox'][2],
						'y2': filtered_pd[j]['bbox'][1] + filtered_pd[j]['bbox'][3],
					}
					iou = get_iou(bbox1, bbox2)
					if iou >= threshold_iou:
						matched_gt += [i]
						matched_pd += [j]

			tps = set(matched_gt)
			fps = [j for j in range(len(filtered_pd)) if not j in matched_pd]
			fns = [i for i in range(len(filtered_gt)) if not i in matched_gt]

			count_tp = len(tps)
			count_fp = len(fps) + len(matched_gt) - len(tps)
			count_fn = len(fns)

			scores[c['name']]['tp'] += count_tp
			scores[c['name']]['fp'] += count_fp
			scores[c['name']]['fn'] += count_fn

			if count_tp > 0:
				precision = count_tp / (count_tp + count_fp)
				recall = count_tp / (count_tp + count_fn)
				f1 = 2 * precision * recall / (precision + recall)
			else:
				precision = 0
				recall = 0
				f1 = 0

			print('\t', c['name'])
			if (count_tp + count_fn):
				print('\t\tTrue Positive  =', count_tp)
				print('\t\tFalse Positive =', count_fp)
				print('\t\tFalse Negative =', count_fn)
				print('\t\tPrecision      =', round(precision, 3))
				print('\t\tRecall         =', round(recall, 3))
				print('\t\tF1 score       =', round(f1, 3))
			else:
				print('No sample')

		print('total_instances', len(self.gts_maps[self.image_id]))
		print('checksum_scores', checksum(scores))
		print('-'*80)

	def display(self):

		canvas = self.image.copy()

		if self.showing['gt']:

			for i in self.gts_maps[self.image_id]:

				d = self.gts['annotations'][i]
				c = condition(d)
				color = colors[c]
				x, y, w, h = d['bbox']
				cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 1)

		if self.showing['pd']:

			for i in self.pds_maps[self.image_id]:

				d = self.pds['annotations'][i]
				if 'score' in d:
					c = d['category_id']
				else:
					c = condition(d)
				color = colors[c]
				x, y, w, h = d['bbox']
				cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 1)

		self.update_tk(canvas)

	def update_tk(self, imagecv):

		# to tkinter image
		imagecv = cv2.resize(imagecv, (0,0), fx=self.scale, fy=self.scale)
		imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2RGB)
		im = Image.fromarray(imagecv)
		imagetk = ImageTk.PhotoImage(image=im)
		panel.configure(image=imagetk)
		panel.image = imagetk


def onKeyPress(event):

	# Exit
	if event.keysym == 'Escape':
		root.destroy()
		exit()

	# Change image
	elif event.char in ['a', 'A', 'ฟ', 'ฤ'] or event.keysym == 'Left':
		image.index = sorted([0, image.index-1, image.total-1])[1]
		image.load_image()
	elif event.char in ['d', 'D', 'ก', 'ฏ'] or event.keysym == 'Right':
		image.index = sorted([0, image.index+1, image.total-1])[1]
		image.load_image()

	elif event.char in [' ']:
		image.showing['gt'] = not image.showing['gt']
		image.showing['pd'] = not image.showing['gt']
		if image.showing['gt']: print('groundtruth')
		if image.showing['pd']: print('prediction')
		image.display()

	elif event.char in ['2', '@', '/', '๑']:
		image.showing['pd'] = not image.showing['pd']
		image.display()

	elif event.char in ['2', '@', '/', '๑']:
		image.showing['pd'] = not image.showing['pd']
		image.display()

if __name__ == "__main__":

    # user inteface
    root = Tk()
    panel = tkinter.Label(root)
    panel.pack()

    image = cellEval()

    root.bind('<KeyPress>', onKeyPress)
    root.mainloop() 