import os
import cv2
import json
import numpy as np
import pandas as pd

def read_coco():
	with open(coco_path) as file:
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

class CocoCellsImage:
	def __init__(self, index):

		# load image
		image_path = coco['images'][index]['file_name']
		image = cv2.imread(image_path)
		self.nuc, self.bac, self.cell = cv2.split(image)
		self.blank = np.zeros_like(self.nuc)

		# get coco label
		self.label = []
		for j in maps[coco['images'][index]['id']]:
			self.label.append(coco['annotations'][j])

		# change window name
		cv2.setWindowTitle(window_name, image_path)
		self.display()

	def canvas(self):

		ch1 = self.blank.copy()
		ch2 = self.blank.copy()
		ch3 = self.blank.copy()

		if showing['bacteria']: ch2 = self.bac
		if showing['nucleus']: ch1 = self.nuc
		if showing['cell']: ch3 = self.cell

		return cv2.merge([ch1, ch2, ch3])

	def draw(self):

		image = self.canvas()

		for i in range(len(self.label)):

			if (self.label[i]['category_id'] == 1) and not showing['bacteria']: continue
			if (self.label[i]['category_id'] == 2) and not showing['nucleus']: continue
			if (self.label[i]['category_id'] == 3) and not showing['cell']: continue

			# draw bbox
			if showing['bbox']:
				x, y, w, h = self.label[i]['bbox']
				x1, y1, x2, y2 = x, y, x+w, y+h
				cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1)

			# draw contour
			if showing['poly']:
				contour = np.array(self.label[i]['segmentation']).reshape(-1, 1, 2)
				cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)

		if not showing['color']:
			ch1, ch2, ch3 = cv2.split(image)
			image = cv2.bitwise_or(ch1, ch2)
			image = cv2.bitwise_or(ch3, image)

		return image

	def display(self):

		image = self.draw()
		
		cv2.imshow(window_name, image)


if __name__ == "__main__":

	# set path
	root = os.path.dirname(os.path.realpath(__file__))

	# load coco data
	coco_path = os.path.join(root, "data/output_export/coco.json")
	coco = read_coco()
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
		'poly': True,
		'color': True,
	}
	view, index = 0, 0
	image = CocoCellsImage(index)

	while True:

		# display
		image.display()

		# user feedback
		key = cv2.waitKeyEx(0)
		if key in [ord('q'), 27]:
			break
		elif key in [ord('a'), 2424832]:
			index = max(index - 1, 0)
			image = CocoCellsImage(index)
		elif key in [ord('d'), 2555904]:
			index = min(index + 1, total - 1)
			image = CocoCellsImage(index)
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
