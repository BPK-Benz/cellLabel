import os
import cv2
import json
import numpy as np
import utils.make_coco as make_coco

def load_coco(coco_path):
	f = open(coco_path)
	return json.load(f)

def calc_avg_mean_std(img_names, size=None):
    mean_sum = np.array([0., 0., 0.])
    std_sum = np.array([0., 0., 0.])
    n_images = len(img_names)
    for img_name in img_names:
        img = cv2.imread(img_name)
        if size: img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean, std = cv2.meanStdDev(img)
        mean_sum += np.squeeze(mean)
        std_sum += np.squeeze(std)
    return (mean_sum / n_images, std_sum / n_images)

if __name__ == "__main__":

	# set path
	root = os.path.dirname(os.path.realpath(__file__))

	# load coco
	coco_path = os.path.join(root, "data/output_export/coco.json")
	# coco_path = os.path.join(root, "data/output_export/S1/Plate_07/Testing_set/coco.json")
	coco = load_coco(coco_path)
	img_names = [sample['file_name'] for sample in coco['images']]
	means, stds = calc_avg_mean_std(img_names)

	print('mean Blue ', means[0])
	print('mean Green', means[1])
	print('mean Red  ', means[2])
	print('std Blue ', stds[0])
	print('std Green', stds[1])
	print('std Red  ', stds[2])
