import os
import cv2
import json
import numpy as np
import pandas as pd
import utils.make_coco as make_coco

colors = {

	# bacteria
	1: (0, 255, 100),			# extra_bac
	2: (0, 255, 255),			# cyto_bac
	3: (0, 100, 255),			# nuc_bac

	# nucleus
	4: (255, 0, 255),			# boundary_nuc
	5: (255, 255, 0),			# divided_nuc
	6: (255, 0, 0),				# completed_nuc

	# cell
	7: (0, 0, 127),				# incompleted_cell
	8: (127, 0, 255),			# uninfected_cell
	9: (51, 255, 51),			# nuc_cell
	10: (255, 255, 0),		# cyto_cell
}

# label: categories
def add_categories_detailed():
	return [
		{
			"supercategory": 'bacteria',
			"id": 1,
			"name": 'extra_bac',
		},
		{
			"supercategory": 'bacteria',
			"id": 2,
			"name": 'cyto_bac',
		},
		{
			"supercategory": 'bacteria',
			"id": 3,
			"name": 'nuc_bac',
		},
		{
			"supercategory": 'nucleus',
			"id": 4,
			"name": 'completed_nuc',
		},
		{
			"supercategory": 'nucleus',
			"id": 5,
			"name": 'boundary_nuc',
		},
		{
			"supercategory": 'nucleus',
			"id": 6,
			"name": 'divided_nuc',
		},
		{
			"supercategory": 'cell',
			"id": 7,
			"name": 'incompleted_cell',
		},
		{
			"supercategory": 'cell',
			"id": 8,
			"name": 'uninfected_cell',
		},
		{
			"supercategory": 'cell',
			"id": 9,
			"name": 'cyto_cell',
		},
		{
			"supercategory": 'cell',
			"id": 10,
			"name": 'nuc_cell',
		},
	]


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
	pass

	#  (disabled)
	# if not os.path.exists(os.path.dirname(path)):
	# 	os.makedirs(os.path.dirname(path))
	# with open(path, 'w') as outfile:
	# 	json.dump(coco, outfile)


def extend_detailed(simple_coco, maps):

	# make new coco
	coco = {
		"info": make_coco.add_info(),
		"licenses": make_coco.add_licenses(),
		"categories": add_categories_detailed(),
		"images": simple_coco["images"],
		"annotations": simple_coco["annotations"],
	}

	# change ids to detailed coco
	transtions = {
		1: 1,		# bacteria -> extra_bac
		2: 4,		# nucleus  -> completed_nuc
		3: 8 		# cell     -> completed_cell
	}
	for i in range(len(coco['annotations'])):
		previous_id = coco['annotations'][i]['category_id']
		coco['annotations'][i]['category_id'] = transtions[previous_id]

	# loop through every images
	for i in range(len(coco['images'])):

		# generate masks for checking bacteria conditions
		height = coco['images'][i]['height']
		width = coco['images'][i]['width']
		mask_nucleus = np.zeros([height, width], np.uint8)
		mask_cell = np.zeros([height, width], np.uint8)
		list_bacteria = []

		# fill masks and list
		for j in maps[coco['images'][i]['id']]:
			ann = coco['annotations'][j]

			# fill nucleus
			if ann['category_id'] == 4:
				contour = np.array(ann['segmentation']).reshape(-1, 1, 2)
				cv2.drawContours(mask_nucleus, [contour], -1, 255, -1)

			# fill cell
			if ann['category_id'] == 8:
				contour = np.array(ann['segmentation']).reshape(-1, 1, 2)
				cv2.drawContours(mask_cell, [contour], -1, 255, -1)

			# fill bacteria
			if ann['category_id'] == 1:
				x, y, w, h = ann['bbox']
				x, y = x + w // 2, y + h // 2
				list_bacteria.append([x, y])

		# assign bacteria conditions
		for j in maps[coco['images'][i]['id']]:
			ann = coco['annotations'][j]

			# change bacteria 
			# [ extra_bac, cyto_bac, nuc_bac ]
			if ann['category_id'] == 1:
				x, y, w, h = ann['bbox']
				x = x + w // 2
				y = y + h // 2
				if mask_cell[y, x] == 255: coco['annotations'][j]['category_id'] = 2
				if mask_nucleus[y, x] == 255: coco['annotations'][j]['category_id'] = 3

			# change cell
			# [ uninfected_cell, cyto_cell, nuc_cell ]
			if ann['category_id'] == 4:

				# for bacteria counting
				cyto_bac = 0
				nuc_bac = 0
				total_bac = 0

				# nucleus param.
				n_ann = ann
				n_contour = np.array(n_ann['segmentation']).reshape(-1, 1, 2)
				x, y, w, h = n_ann['bbox']
				n_x = x + w // 2
				n_y = y + h // 2

				# cell param.
				c_contour = np.array([0, 0, 0, 0]).reshape(-1, 1, 2)

				# find cell pair
				for k in maps[coco['images'][i]['id']]:

					# filter bacteria and nucleus
					c_ann = coco['annotations'][k]
					if not c_ann['category_id'] == 8: continue

					# cell param.
					c_contour = np.array(c_ann['segmentation']).reshape(-1, 1, 2)
					if cv2.pointPolygonTest(c_contour,(n_x,n_y),True) > 0: break

				# find resident bacteria
				for x, y in list_bacteria:

					if cv2.pointPolygonTest(n_contour,(x,y),True) > 0: nuc_bac += 1
					if cv2.pointPolygonTest(c_contour,(x,y),True) > 0: total_bac += 1

				cyto_bac = total_bac - nuc_bac
				if total_bac == 0:
					coco['annotations'][k]['category_id'] = 8
				elif cyto_bac > nuc_bac:
					coco['annotations'][k]['category_id'] = 9
				else:
					coco['annotations'][k]['category_id'] = 10

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
	pass

	# # (disabled)
	# # left mouse click event
	# if event == cv2.EVENT_LBUTTONUP:

	# 	# loop through every instances
	# 	for i in range(len(image.label)):

	# 		# check id
	# 		cat_id = image.label[i]['category_id']
	# 		if (cat_id in bacteria_ids) and not mode == 'bacteria': continue
	# 		if (cat_id in nucleus_ids) and not mode == 'nucleus': continue
	# 		if (cat_id in cell_ids) and not mode == 'cell': continue

	# 		# update id of clicked
	# 		ann_id = image.label[i]['id']
	# 		contour = image.contours[ann_id]
	# 		is_clicked = cv2.pointPolygonTest(contour,(x,y),True) > 0
	# 		if is_clicked:
	# 			new_id = update_category(cat_id)
	# 			image.label[i]['category_id'] = new_id

	# 			# boundary or divided cases
	# 			if new_id in [5, 6]:

	# 				# loop through every instances (but looking for cells)
	# 				for i in range(len(image.label)):

	# 					# check id
	# 					cat_id = image.label[i]['category_id']
	# 					if not cat_id in cell_ids: continue

	# 					# update id of clicked 
	# 					ann_id = image.label[i]['id']
	# 					contour = image.contours[ann_id]
	# 					is_clicked = cv2.pointPolygonTest(contour,(x,y),True) > 0

	# 					# as incompleted cell
	# 					if is_clicked:
	# 						image.label[i]['category_id'] = 7

	# 	# display
	# 	image.display()


class CocoCellsImage:
	def __init__(self, index):

		# load image
		image_path = coco['images'][index]['file_name']
		print(image_path)
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

	# load coco data (disabled)
	# if not os.path.isfile(coco2_path):
	# 	coco = read_coco(coco1_path)
	# 	maps = read_anno(coco)
	# 	coco = extend_detailed(coco ,maps)
	# else:
	# 	coco = read_coco(coco2_path)
	# 	maps = read_anno(coco)

	# gonna make new coco
	coco = read_coco(coco1_path)
	maps = read_anno(coco)
	coco = extend_detailed(coco ,maps)

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
	bacteria_ids = [1, 2, 3]
	nucleus_ids = [4, 5, 6]
	cell_ids = [7, 8, 9, 10]
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

