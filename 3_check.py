import os
import cv2
import json
import numpy as np
import pandas as pd
import colorsys

def make_colors(variation):
    hues = np.linspace(0, 1, variation+1)[:-1]
    variation = [[int(255*c) for c in colorsys.hsv_to_rgb(h, 1, 1)] for h in hues]
    colors = [[0, 0, 0]]
    colors += variation
    colors = [c[::-1] for c in colors]
    return colors

def read_coco():
	with open(coco_path) as file:
		d = json.load(file)
	print('[ Finished Read COCO ]')
	return d

def read_categories(coco):

	variation = len(coco['categories'])
	colors = make_colors(variation)

	categories = {}
	for c in range(variation):
		categories[coco['categories'][c]['id']] = {
			'name': coco['categories'][c]['name'],
			'color': colors[c+1]
		}
	print('[ Finished Read Categories ]')
	return categories

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

def mouse_event(event, x, y, flags, param):
	
    # mouse motion
	if event == cv2.EVENT_MOUSEMOVE:

		x = int(x / image.scale)
		y = int(y / image.scale)

		new_hovers = []
		for i in range(len(image.label)):
			contour = np.array(image.label[i]['segmentation']).reshape(-1, 1, 2)
			if cv2.pointPolygonTest(contour, (x,y), True) > 0:
				new_hovers.append(i)

		# update draw
		if not image.hovers == new_hovers:
			image.hovers = new_hovers
			image.display()


class CocoCellsImage:
	def __init__(self, index):

		# load image
		image_path = coco['images'][index]['file_name']
		image = cv2.imread(image_path)
		self.nuc, self.bac, self.cell = cv2.split(image)

		# prepare blank for image processing
		self.height, self.width, _ = image.shape
		self.blank1 = np.zeros([self.height, self.width], np.uint8)
		self.blank3 = np.zeros([self.height, self.width, 3], np.uint8)

		# get coco label
		self.label = []
		if coco['images'][index]['id'] in maps:
			for j in maps[coco['images'][index]['id']]:
				self.label.append(coco['annotations'][j])

		# to display on mouse hover
		self.hovers = []

		# change window name
		cv2.setWindowTitle(window_name, image_path)
		self.scale = 0.9

		# display
		self.display()

	def canvas(self):

		ch1 = self.blank1.copy()
		ch2 = self.blank1.copy()
		ch3 = self.blank1.copy()

		if showing['bacteria']: ch2 = self.bac
		if showing['nucleus']: ch1 = self.nuc
		if showing['cell']: ch3 = self.cell

		return cv2.merge([ch1, ch2, ch3])

	def draw(self):

		if mode == 'color':
			image = self.canvas()

		if mode == 'gray':
			image = self.canvas()
			ch1, ch2, ch3 = cv2.split(image)
			image = cv2.bitwise_or(ch1, ch2)
			image = cv2.bitwise_or(ch3, image)
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

		if mode == 'overlay':
			image = self.canvas()
			ch1, ch2, ch3 = cv2.split(image)
			image = cv2.bitwise_or(ch1, ch2)
			image = cv2.bitwise_or(ch3, image)
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

			overlay = self.blank3.copy()
			for i in range(len(self.label)):

				contour = np.array(self.label[i]['segmentation']).reshape(-1, 1, 2)
				color = categories[self.label[i]['category_id']]['color']
				cv2.fillPoly(overlay, pts =[contour], color=color)

			image = cv2.addWeighted(image, 1, overlay, 0.3, 0)
			image[ch2 > 127] = (0, 255, 0)

		# draw poly and bbox
		for i in range(len(self.label)):

			if showing['poly']:
				contour = np.array(self.label[i]['segmentation']).reshape(-1, 1, 2)
				cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)

			if showing['bbox']:
				x, y, w, h = self.label[i]['bbox']
				x1, y1, x2, y2 = x, y, x+w, y+h
				cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1)

		# fill poly on hover
		for i in self.hovers:

			if hovering['fill']:
				contour = np.array(self.label[i]['segmentation']).reshape(-1, 1, 2)
				category_id = self.label[i]['category_id']
				area = self.label[i]['area']
				name = categories[category_id]['name']
				color = categories[category_id]['color']
				cv2.fillPoly(image, pts =[contour], color=color)
				print(len(self.hovers), name)

		return image

	def display(self):

		image = self.draw()

		image = cv2.resize(image, (0, 0), fx=self.scale, fy=self.scale)		
		cv2.imshow(window_name, image)


if __name__ == "__main__":

	# set path
	root = os.path.dirname(os.path.realpath(__file__))

	# load coco data
	coco_path = os.path.join(root, "data/output_coco/S1/Plate_03/Testing_set/003007-5-001001.json")
	# coco_path = os.path.join(root, "data/output_predicted1/S1/Plate_03/Training/001001-1-001001.json")
	coco_path = 'groundtruth.json'
	# coco_path = 'test.json'
	coco = read_coco()
	categories = read_categories(coco)
	maps = read_anno(coco)
	total = len(coco['images'])

	# set window	
	window_name = 'default'
	cv2.namedWindow (window_name, flags=cv2.WINDOW_AUTOSIZE)

	# Initialize parameters
	showing = {

		'bacteria': True,
		'nucleus': False,
		'cell': True,

		'bbox': False,
		'poly': False,

	}
	hovering = {

		'fill': True,
	
	}
	modes = ['color', 'gray', 'overlay']
	mode = 'overlay'
	view, index = 0, 0
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
			index = max(index - 1, 0)
			image = CocoCellsImage(index)
		elif key in [ord('d'), 2555904]:
			index = min(index + 1, total - 1)
			image = CocoCellsImage(index)
		elif key in [65470]: # F1
			showing['bacteria'] = not showing['bacteria']
		elif key in [65471]: # F2
			showing['nucleus'] = not showing['nucleus']
		elif key in [65472]: # F3
			showing['cell'] = not showing['cell']
		elif key in [ord('b')]:
			showing['bbox'] = not showing['bbox']
		elif key in [ord('p')]:
			showing['poly'] = not showing['poly']
		elif key in [ord('c')]:
			mode = 'color'
		elif key in [ord('g')]:
			mode = 'gray'
		elif key in [ord('o')]:
			mode = 'overlay'
