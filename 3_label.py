import os
import cv2
import json
import numpy as np
import pandas as pd
import utils.make_coco as make_coco


colors = {
	1: (0, 255, 0),			# bacteria
	2: (255, 0, 0),			# nucleus
	3: (0, 0, 255),			# common_cell
	4: (255, 255, 255),		# dividing_cell
}


def read_coco(path):
	with open(path) as file:
		d = json.load(file)
	return d


def read_anno(coco):
	maps = {}
	for i, annotation in enumerate(coco['annotations']):
		image_id = annotation['image_id']
		if not image_id in maps:
			maps[image_id] = [i]
		else:
			maps[image_id].append(i)
	return maps


def save_coco(path):
	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))
	with open(path, 'w') as outfile:
		json.dump(coco, outfile)


def extend_detailed(simple_coco, maps):

	# make new coco
	coco = {
		"info": make_coco.add_info(),
		"licenses": make_coco.add_licenses(),
		"categories": make_coco.add_categories_detailed(),
		"images": simple_coco["images"],
		"annotations": simple_coco["annotations"],
	}

	return coco


def update_category(_id):
	for ids in [nucleus_ids, cell_ids, bacteria_ids]:
		if _id in ids:
			p = ids.index(_id) + 1
			if p < len(ids):
				return ids[p]
			else:
				return ids[0]


def mouse_event(event, x, y, flags, param):

	# left mouse click event
	if event == cv2.EVENT_LBUTTONUP:

		# loop through every instances
		for i in range(len(image.label)):

			# check id. (We will change only cells.)
			cat_id = image.label[i]['category_id']
			if cat_id in bacteria_ids + nucleus_ids: continue
			if mode in ['bacteria', 'nucleus']: continue

			# update id of clicked
			ann_id = image.label[i]['id']
			contour = image.contours[ann_id]
			is_clicked = cv2.pointPolygonTest(contour,(x,y),True) > 0
			if is_clicked:
				new_id = update_category(cat_id)
				image.label[i]['category_id'] = new_id

		# display
		image.display()



class CocoCellsImage:
	def __init__(self, index):

		# load image
		image_path = os.path.join(root, coco['images'][index]['file_name'])
		image = cv2.imread(image_path)
		self.nuc, self.bac, self.cell = cv2.split(image)
		self.blank = np.zeros_like(self.nuc)
		self.blank3 = np.zeros_like(image)

		# get anno in coco format
		self.label = []
		for j in maps[coco['images'][index]['id']]:
			self.label.append(coco['annotations'][j])

		# get anno in poly
		self.contours = {}
		for j in range(len(self.label)):
			ann_id = self.label[j]['id']
			contour = np.array(self.label[j]['segmentation']).reshape(-1, 1, 2)
			self.contours[ann_id] = contour

		# change window name
		cv2.setWindowTitle(window_name, image_path)

		# display
		self.display()

	def canvas(self):

		image = self.blank.copy()
		if showing['bacteria']: image = cv2.bitwise_or(image, self.bac//2)
		if showing['nucleus']: image = cv2.bitwise_or(image, self.nuc//2)
		if showing['cell']: image = cv2.bitwise_or(image, self.cell//2)
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

		return image

	def draw(self):

		image = self.canvas()

		# draw overlay
		if showing['color']:
			overlay = self.blank3.copy()
			for i in range(len(self.label)):

				if (self.label[i]['category_id'] in bacteria_ids) and not mode == 'bacteria': continue
				if (self.label[i]['category_id'] in nucleus_ids) and not mode == 'nucleus': continue
				if (self.label[i]['category_id'] in cell_ids) and not mode == 'cell': continue

				cat_id = self.label[i]['category_id']
				ann_id = self.label[i]['id']
				contour = self.contours[ann_id]
				cv2.drawContours(overlay, [contour], -1, colors[cat_id], -1)
			image = cv2.addWeighted(image, .9, overlay, .3, 0)

		# draw bbox & poly
		for i in range(len(self.label)):

			if (self.label[i]['category_id'] in bacteria_ids) and not mode == 'bacteria': continue
			if (self.label[i]['category_id'] in nucleus_ids) and not mode == 'nucleus': continue
			if (self.label[i]['category_id'] in cell_ids) and not mode == 'cell': continue

			# draw bbox
			if showing['bbox']:
				x, y, w, h = self.label[i]['bbox']
				x1, y1, x2, y2 = x, y, x+w, y+h
				cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1)

			# draw contour
			if showing['poly']:
				ann_id = self.label[i]['id']
				contour = self.contours[ann_id]
				cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)

		return image

	def display(self):

		image = self.draw()		
		cv2.imshow(window_name, image)


if __name__ == "__main__":

	# set path
	root = os.path.dirname(os.path.realpath(__file__))
	coco1_path = os.path.join(root, "data/output_export/coco.json")
	coco2_path = os.path.join(root, "data/output_label/coco.json")

	coco1_path = os.path.join(root, "data/output_export/S1/Plate_03/Testing_set/coco.json")
	coco2_path = os.path.join(root, "data/output_export/S1/Plate_03/Testing_set/coco.json")

	# load coco data
	if not os.path.isfile(coco2_path):
		coco = read_coco(coco1_path)
		maps = read_anno(coco)
		coco = extend_detailed(coco ,maps)
	else:
		coco = read_coco(coco2_path)
		maps = read_anno(coco)
	total = len(coco['images'])

	# set window	
	window_name = 'default'
	cv2.namedWindow (window_name, flags=cv2.WINDOW_AUTOSIZE)

	# Initialize parameters
	showing = {
		'bacteria': True,
		'nucleus': True,
		'cell': True,
		'bbox': False,
		'poly': False,
		'color': True
	}
	view, index = 1, 0 # change image index
	modes = ['bacteria', 'nucleus', 'cell']
	mode = modes[view]
	bacteria_ids = [1]
	nucleus_ids = [2]
	cell_ids = [3, 4]
	image = CocoCellsImage(index)

	while True:

		# display
		image.display()

		# user feedback
		cv2.setMouseCallback(window_name, mouse_event)
		key = cv2.waitKeyEx(0)
		if key in [ord('q'), 27]:
			break
		elif key in [ord('a'), 2424832]:
			save_coco(coco2_path)
			index = max(index - 1, 0)
			image = CocoCellsImage(index)
		elif key in [ord('d'), 2555904]:
			save_coco(coco2_path)
			index = min(index + 1, total - 1)
			image = CocoCellsImage(index)
		elif key in [ord('w'), 2490368, 228]:
			view = max(view - 1, 0)
			mode = modes[view]
		elif key in [ord('s'), 2621440, 203]:
			view = min(view + 1, len(modes) - 1)
			mode = modes[view]
		elif key in [ord('1'), 49]:
			showing['bacteria'] = not showing['bacteria']
		elif key in [ord('2'), 50]:
			showing['nucleus'] = not showing['nucleus']
		elif key in [ord('3')]:
			showing['cell'] = not showing['cell']
		elif key in [ord('b')]:
			showing['bbox'] = not showing['bbox']
		elif key in [ord('p')]:
			showing['poly'] = not showing['poly']
		elif key in [ord('c')]:
			showing['color'] = not showing['color']
		elif key in [ord('z')]:
			save_coco(coco2_path)

