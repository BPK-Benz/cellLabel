import os
import cv2
import time
import json
import numpy as np
import pandas as pd
import utils.calculation as calc
import utils.make_coco as make_coco

# json file to dictionary
def load_label(path):

	data = {}
	with open(path) as file:
		d = json.load(file)
		for key in d:
			data[key] = {
				'contour': np.array(d[key]['contour']),
				'centroid': d[key]['centroid'],
				'label': d[key]['label'],
			}
	return data

# bacteria position to list
def load_position(bacteria, number):

	data = []
	b = bacteria.loc[bacteria['ImageNumber'] == number]
	for index, row in b.iterrows():
		x = int(row['AreaShape_Center_X'])
		y = int(row['AreaShape_Center_Y'])
		data.append([x, y])
	return data

def create_coco():
	return {
		"info": make_coco.add_info(),
		"licenses": make_coco.add_licenses(),
		"categories": make_coco.add_categories(),
		"images": [],
		"annotations": [],
	}

def load_coco(coco_path):
	f = open(coco_path)
	return json.load(f)

def save_coco(coco_path, coco):
	with open(coco_path, 'w') as outfile:
		json.dump(coco, outfile)

def merge_coco():

	# init super_coco
	super_coco = create_coco()

	# set path
	root = os.path.dirname(os.path.realpath(__file__))

	# loop
	replicates = os.listdir(os.path.join(root, 'data/output_localize'))
	replicates = sorted(replicates)
	for r in replicates:
		plates = os.listdir(os.path.join(root, 'data/output_localize', r))
		plates = sorted(plates)
		for p in plates:
			sets = os.listdir(os.path.join(root, 'data/output_localize', r, p))
			sets = sorted(sets)
			for s in sets:
				coco_path = os.path.join(root, 'data/output_export', r, p, s, 'coco.json')				
				coco = load_coco(coco_path) if os.path.exists(coco_path) else create_coco()
				super_coco['images'] += coco['images']
				super_coco['annotations'] += coco['annotations']

	super_coco_path = os.path.join(root, 'data/output_export', 'coco.json')
	save_coco(super_coco_path, super_coco)

def count_total():

	# init total
	total = 0

	# set path
	root = os.path.dirname(os.path.realpath(__file__))

	# loop
	replicates = os.listdir(os.path.join(root, 'data/output_localize'))
	replicates = sorted(replicates)
	for r in replicates:
		plates = os.listdir(os.path.join(root, 'data/output_localize', r))
		plates = sorted(plates)
		for p in plates:
			sets = os.listdir(os.path.join(root, 'data/output_localize', r, p))
			sets = sorted(sets)
			for s in sets:
				wells = [well for well in os.listdir(os.path.join(root, 'data/output_localize', r, p, s)) if well.endswith('003.json')]
				total += len(wells)
	return total

def get_features(image, mask):

	# calculate contour
	contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	# calculate area and choose the biggest as represent
	area, contour = 0, None
	for con in contours:
		a = cv2.contourArea(con)
		if a > area:
			area = a
			contour = con

	# empty key case
	if contour is None: return 
	if len(contour) < 5: return 
	if area < 100: return 

	# calculate centroid
	cx, cy = calc.centroid(contour)

	# calculate compactness
	compactness = calc.compactness(contour, area)

	# calculate minAxis, maxAxis
	minAxis, maxAxis = calc.axis(contour)

	# calculate equivalent diameter
	diameter = calc.diameter(area)

	# calculate minFeret, maxFeret
	minFeret, maxFeret = calc.feret(contour, area)

	# # calculate eccenticity
	eccenticity = calc.eccenticity(contour)

	# calculate perimeter
	perimeter = calc.perimeter(contour)

	# calculate extent from standed bounding box
	extent = calc.extent(contour, area)

	# calculate extent from rotated bounding box
	extent_axis = calc.extent2(area, minAxis, maxAxis)

	# calculate solidity
	solidity = calc.solidity(contour, area)

	# calculate integrated intensity
	intensity = calc.intensity(image, mask)

	# calculate bounding rectangle
	rectangle = cv2.boundingRect(contour)

	return {
		'X': cx,
		'Y': cy,
		'area': area,
		'compactness': compactness, 
		'minAxis': minAxis, 
		'maxAxis': maxAxis, 
		'diameter': diameter, 
		'minFeret': minFeret, 
		'maxFeret': maxFeret, 
		'eccenticity': eccenticity, 
		'perimeter': perimeter, 
		'extent': extent, 
		'extent_axis': extent_axis, 
		'solidity': solidity, 
		'intensity': intensity,
		'contour': [contour.flatten().tolist()],
		'rectangle': rectangle
	}


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1, alpha_ratio=1, beta_ratio=1):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Calculate grayscale histogram
	hist = cv2.calcHist([gray],[0],None,[256],[0,256])
	hist_size = len(hist)

	# Calculate cumulative distribution from the histogram
	accumulator = []
	accumulator.append(float(hist[0]))
	for index in range(1, hist_size):
		accumulator.append(accumulator[index -1] + float(hist[index]))

	# Locate points to clip
	maximum = accumulator[-1]
	clip_hist_percent *= (maximum/100.0)
	clip_hist_percent /= 2.0

	# Locate left cut
	minimum_gray = 0
	while accumulator[minimum_gray] < clip_hist_percent:
		minimum_gray += 1

	# Locate right cut
	maximum_gray = hist_size -1
	while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
		maximum_gray -= 1

	# Calculate alpha and beta values
	if maximum_gray - minimum_gray > 0:
		alpha = 255 / (maximum_gray - minimum_gray)
	else:
		alpha = 255
	beta = -minimum_gray * alpha

	auto_result = cv2.convertScaleAbs(image, alpha=alpha/alpha_ratio, beta=beta/beta_ratio)
	return (auto_result, alpha, beta)


if __name__ == "__main__":

	# set path
	root = os.path.dirname(os.path.realpath(__file__))

	columns = []

	# collect file information as csv
	columns += ['Replicate', 'Plate', 'Well']

	# collect nucleus information
	columns += ['Nucleus_X', 'Nucleus_Y', 'Nucleus_Area', 'Nucleus_Perimeter']
	columns += ['Nucleus_Compactness', 'Nucleus_Diameter', 'Nucleus_Eccentricity']
	columns += ['Nucleus_Extent', 'Nucleus_Extent_Axis', 'Nucleus_Solidity', 'Nucleus_Intensity']
	columns += ['Nucleus_MinAxis', 'Nucleus_MaxAxis', 'Nucleus_MinFeret', 'Nucleus_MaxFeret']

	# collect cell information
	columns += ['Cell_X', 'Cell_Y', 'Cell_Area', 'Cell_Perimeter']
	columns += ['Cell_Compactness', 'Cell_Diameter', 'Cell_Eccentricity']
	columns += ['Cell_Extent', 'Cell_Extent_Axis', 'Cell_Solidity', 'Cell_Intensity']
	columns += ['Cell_MinAxis', 'Cell_MaxAxis', 'Cell_MinFeret', 'Cell_MaxFeret']

	# collect bacteria information
	columns += ['Nuc_Bac', 'Cyto_Bac', 'Nuc_Cell', 'Cyto_Cell']


	# make label data in coco json format
	count_image = 1
	count_annotation = 1
	total = count_total()

	# run through directories
	replicates = os.listdir(os.path.join(root, 'data/output_localize'))
	replicates = sorted(replicates)
	# replicates = ['S1']
	for r in replicates:
		plates = os.listdir(os.path.join(root, 'data/output_localize', r))
		plates = sorted(plates)
		# plates = plates[:3]
		# plates = ['Plate_03']
		for p in plates:
			sets = os.listdir(os.path.join(root, 'data/output_localize', r, p))
			sets = sorted(sets)
			# sets = ['Testing_set']
			for s in sets:

				# get bacteria infomation
				properties = pd.DataFrame([], columns = columns)
				coco = create_coco()

				# at each image
				wells = [well for well in os.listdir(os.path.join(root, 'data/output_localize', r, p, s)) if well.endswith('003.json')]
				wells = sorted(wells)
				# wells = ['012013-2-001001003.json']
				# wells = wells[:3]
				for well in wells:

					# timer
					start = time.time()

					# point to path
					name = well[:-8]
					original1_path = os.path.join(root, 'data/Reference_gene', r, p, s, name + '001.tif')
					original2_path = os.path.join(root, 'data/Reference_gene', r, p, s, name + '002.tif')
					original3_path = os.path.join(root, 'data/Reference_gene', r, p, s, name + '003.tif')
					label1_path = os.path.join(root, 'data/output_localize', r, p, s, name + '001.json')
					label2_path = os.path.join(root, 'data/output_localize', r, p, s, name + '002.json')
					label3_path = os.path.join(root, 'data/output_localize', r, p, s, name + '003.json')

					# load images (nucleus + cytoplasm)
					image1 = cv2.imread(original1_path)
					image2 = cv2.imread(original2_path)
					image3 = cv2.imread(original3_path)
					# image1 = automatic_brightness_and_contrast(image1, alpha_ratio=10, beta_ratio=15)[0]
					# image2 = automatic_brightness_and_contrast(image2)[0]
					# image3 = automatic_brightness_and_contrast(image3)[0]

					# save enhanced image
					output_path = os.path.join(root, 'data/output_export', r, p, s)
					output_file = output_path + '/' + name + '.png'
					if not os.path.exists(output_path): os.makedirs(output_path)
					ch1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
					ch1[ch1<100] = 0
					ch2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
					ch3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
					merged = cv2.merge([ch2, ch1, ch3])
					cv2.imwrite(output_file, merged)

					# get resolution
					height, width, _ = image2.shape

					# load data (bac + nucleus + cytoplasm)
					data1 = load_label(label1_path)
					data2 = load_label(label2_path)
					data3 = load_label(label3_path)

					# list of bacteria location
					bacteria_list = []

					# add bacteria to coco
					for key in data1:

						contour = data1[key]['contour']
						centroid = data1[key]['centroid']
						label = data1[key]['label']

						if not label: continue

						x1, y1 = contour[0][0]
						x2, y2 = contour[2][0]
						x1, x2 = [sorted([0, int(x), width -1])[1] for x in [x1, x2]]
						y1, y2 = [sorted([0, int(y), height-1])[1] for y in [y1, y2]]
						w, h = x2 - x1, y2 - y1
						area = w * h

						if area < 22: continue

						# make list for check nuc-bac, cyto-bac cases
						bacteria_list += [centroid]

						# append nucleus data to coco json
						# coco_object = {
						# 	"area": int(area),
						# 	"iscrowd": 0,
						# 	"image_id": count_image,
						# 	"bbox": [int(i) for i in [x1, y1, w, h]],
						# 	"segmentation": [int(i) for i in [x1, y1, x1, y2, x2, y2, x2, y1]],
						# 	"category_id": 1,
						# 	"id": count_annotation,
						# }
						coco_object = {
							"area": area,
							"iscrowd": 0,
							"image_id": count_image,
							"bbox": [x1, y1, w, h],
							"segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
							"category_id": 1,
							"id": count_annotation,
						}
						coco['annotations'].append(coco_object)
						count_annotation += 1

					# make nucleus mask
					channel2 = np.zeros([height, width], np.uint8)
					for key in data2:

						# fill image
						contour = data2[key]['contour']
						cv2.fillPoly(channel2, pts =[contour], color=255)

					# fuse all nucleus
					channel2 = cv2.morphologyEx(channel2, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))

					# make cytoplasm mask
					channel3 = np.zeros([height, width], np.int32)
					for key in data3:

						# store single cell region for preprocessing
						cytoplasm = np.zeros([height, width], np.uint8)

						# fill image
						for index in data3:
							if key == str(data3[index]['label']):
								contour = data3[index]['contour']
								cv2.fillPoly(cytoplasm, pts =[contour], color=255)

						# remove fusing artifacts
						cytoplasm = cv2.morphologyEx(cytoplasm, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))

						# overlay region on markers image
						channel3[cytoplasm==255] = int(key)

					# get properties of each
					for key in data3:

						# store single cell region for calculation
						nucleus = np.zeros([height, width], np.uint8)
						cytoplasm = np.zeros([height, width], np.uint8)

						# get region info
						cytoplasm[channel3==int(key)] = 255
						nucleus = cv2.bitwise_and(cytoplasm, channel2)

						# apply sequence of functions
						features2 = get_features(image2, nucleus.copy())
						features3 = get_features(image3, cytoplasm.copy())
						if features2 is None: continue
						if features3 is None: continue

						# nucbac, cytobac
						allbac = 0
						nucbac = 0
						cytobac = 0
						for x, y in bacteria_list:
							if cytoplasm[y, x] == 255:
								allbac += 1
								if nucleus[y, x] == 255:
									nucbac += 1
						cytobac = allbac - nucbac

						# nuccell, cytocell
						nuccell = int(nucbac >= cytobac)
						cytocell = int(cytobac > nucbac)

						# append data to dictionary
						cell = {

							# file
							'Replicate': r, 
							'Plate': p, 
							'Well': name, 

							# nucleus
							'Nucleus_X': features2['X'],
							'Nucleus_Y': features2['Y'],
							'Nucleus_Area': features2['area'],
							'Nucleus_Perimeter': features2['perimeter'],
							'Nucleus_Compactness': features2['compactness'],
							'Nucleus_Diameter': features2['diameter'],
							'Nucleus_Eccentricity': features2['eccenticity'],
							'Nucleus_Extent': features2['extent'],
							'Nucleus_Extent_Axis': features2['extent_axis'],
							'Nucleus_Solidity': features2['solidity'],
							'Nucleus_Intensity': features2['intensity'],
							'Nucleus_MinAxis': features2['minAxis'],
							'Nucleus_MaxAxis': features2['maxAxis'],
							'Nucleus_MinFeret': features2['minFeret'],
							'Nucleus_MaxFeret': features2['maxFeret'],

							# cytoplasm
							'Cell_X': features3['X'],
							'Cell_Y': features3['Y'],
							'Cell_Area': features3['area'],
							'Cell_Perimeter': features3['perimeter'],
							'Cell_Compactness': features3['compactness'],
							'Cell_Diameter': features3['diameter'],
							'Cell_Eccentricity': features3['eccenticity'],
							'Cell_Extent': features3['extent'],
							'Cell_Extent_Axis': features3['extent_axis'],
							'Cell_Solidity': features3['solidity'],
							'Cell_Intensity': features3['intensity'],
							'Cell_MinAxis': features3['minAxis'],
							'Cell_MaxAxis': features3['maxAxis'],
							'Cell_MinFeret': features3['minFeret'],
							'Cell_MaxFeret': features3['maxFeret'],

							# bacteria
							'Nuc_Bac': nucbac,
							'Nuc_Cell': nuccell,
							'Cyto_Bac': cytobac,
							'Cyto_Cell': cytocell,
						}
						properties = properties.append(cell, ignore_index=True)

						# append nucleus data to coco json
						coco_object = {
							"area": features2['area'],
							"iscrowd": 0,
							"image_id": count_image,
							"bbox": features2['rectangle'],
							"segmentation": features2['contour'],
							"category_id": 2,
							"id": count_annotation,
						}
						coco['annotations'].append(coco_object)
						count_annotation += 1

						# append cell data to coco json
						coco_object = {
							"area": features3['area'],
							"iscrowd": 0,
							"image_id": count_image,
							"bbox": features3['rectangle'],
							"segmentation": features3['contour'],
							"category_id": 3,
							"id": count_annotation,
						}
						coco['annotations'].append(coco_object)
						count_annotation += 1

					# append bacteria image to coco json
					coco_image = {
						"license": 1,
						"file_name": output_file.replace(root + '/', ''),
						"height": height,
						"width": width,
						"id": count_image,
					}
					coco['images'].append(coco_image)
					count_image += 1

					# print progress
					elapse = time.time() - start
					print('[ {:4d} of {:4d} | {:.2f} ]  {}  {}  {}  {}'.format(count_image-1, total, elapse, r, p, s, well))

				# save properties to csv
				properties_path = os.path.join(root, 'data/output_export', r, p, s, 'properties.csv')
				properties.to_csv(properties_path, index= False)

				# save coco to json
				coco_path = os.path.join(root, 'data/output_export', r, p, s, 'coco.json')
				save_coco(coco_path, coco)
	merge_coco()