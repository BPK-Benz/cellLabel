import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset


def preproc(image):

	image = image.astype(np.float32)
	image /= 255
	image = np.transpose(image, (2, 0, 1))

	return image


def data_augmentation(image, label):

	# horizontal flip
	if random.random() > .5:
		image = cv2.flip(image, 1)
		label = cv2.flip(label, 1)

	# vertical flip
	if random.random() > .5:
		image = cv2.flip(image, 0)
		label = cv2.flip(label, 0)

	# padding x
	if random.random() > .5:
		h, w = image.shape[:2]
		extend1 = int(w * random.uniform(0, .3))
		extend2 = int(w * random.uniform(0, .3))

		# image
		pad1 = np.zeros([h, extend1, 3], np.uint8)
		pad2 = np.zeros([h, extend2, 3], np.uint8)
		image = cv2.hconcat([pad1, image, pad2])

		# label
		pad1 = np.zeros([h, extend1], np.uint8)
		pad2 = np.zeros([h, extend2], np.uint8)
		label = cv2.hconcat([pad1, label, pad2])

	# padding y
	if random.random() > .5:
		h, w = image.shape[:2]
		extend1 = int(h * random.uniform(0, .3))
		extend2 = int(h * random.uniform(0, .3))

		# image
		pad1 = np.zeros([extend1, w, 3], np.uint8)
		pad2 = np.zeros([extend2, w, 3], np.uint8)
		image = cv2.vconcat([pad1, image, pad2])

		# label
		pad1 = np.zeros([extend1, w], np.uint8)
		pad2 = np.zeros([extend2, w], np.uint8)
		label = cv2.vconcat([pad1, label, pad2])

	# crop
	if random.random() > .3:
		h, w, _ = image.shape
		ratio_h = random.uniform(.4, 1)
		ratio_w = random.uniform(.4, 1)
		start_y = random.uniform(0, 1 - ratio_h)
		start_x = random.uniform(0, 1 - ratio_w)
		x1 = int(w * start_x)
		y1 = int(h * start_y)
		x2 = int(w * (start_x + ratio_w))
		y2 = int(h * (start_y + ratio_h))
		image = image[y1:y2, x1:x2]
		label = label[y1:y2, x1:x2]

	return image, label


class cell_dataset(Dataset):

	def __init__(self, split, augment=True, transform=preproc):

		self.h = 1024 // 2
		self.w = 1360 // 2
		self.transform = transform
		self.augment = augment

		# get image names
		with open(split) as f:
			self.samples = f.readlines()
		self.total = len(self.samples)

	def __len__(self):
		return self.total

	def __getitem__(self, index):

		# get data
		name = self.samples[index].replace('\n', '.png')
		image_path = os.path.join('data/output_export', name)
		label_path = os.path.join('data/output_segment', name)

		image = cv2.imread(image_path)
		label = cv2.imread(label_path, 0)

		# data augmentation
		if self.augment:
			image, label = data_augmentation(image, label)

		image = cv2.resize(image, (self.w, self.h))
		label = cv2.resize(label, (self.w, self.h))

		# pre process
		if self.transform:

			mask = np.zeros_like(image)
			mask[label == 1] = (0, 0, 255)
			mask[label == 2] = (255, 0, 0)
			mask[label == 3] = (255, 255, 255)
			mask = self.transform(mask)

			image = self.transform(image)
			label = mask.copy()

		return image, label

if __name__ == "__main__":

	# load dataset

	train_split_path = 'source/train.txt'
	train = cell_dataset(
		split = train_split_path, 
		augment = True, 
		transform = False
	)

	test_split_path = 'source/test.txt'
	test = cell_dataset(
		split = test_split_path, 
		augment = False,
		transform = False
	)

	# print total
	total = train.total + test.total

	print('Training Sample  {}'.format(train.total))
	print('Training Percent {:.2f}'.format(train.total / total * 100))
	print('-' * 80)

	print('Testing Sample   {}'.format(test.total))
	print('Testing Percent  {:.2f}'.format(test.total / total * 100))
	print('-' * 80)

	# # showing
	for image, label in train:
		mask = np.zeros_like(image)
		mask[label == 1] = (127, 127, 255)
		mask[label == 2] = (255, 127, 127)
		mask[label == 3] = (255, 255, 255)
		display = cv2.hconcat([image, mask])
		display = cv2.resize(display, (0, 0), fx=.5, fy=.5)
		cv2.imshow('display', display)
		key = cv2.waitKey(0)
		if key == 27:
			break