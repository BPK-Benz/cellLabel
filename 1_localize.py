import os
import cv2
import json
import numpy as np
import pandas as pd
import colorsys

# make color dictioanary
def make_colors():
	n_variation = 20
	n_multitime = 50
	hues = np.linspace(0, 1, n_variation+1)[:-1]
	variation = [[int(255*c) for c in colorsys.hsv_to_rgb(h, 1, 1)] for h in hues]
	colors = [[0, 0, 0]]
	for _ in range(n_multitime):
		colors += variation
	return colors
cell_colors = make_colors()
bac_colors = [[0, 0, 0], [0, 255, 0]]

def display(src):

	img = src.copy()

	if mode in ['nucleus', 'cytoplasm']:	
		color = cell_colors[selected]
		cv2.circle(img, mouse, 7, color, -1)

	elif mode in ['bacteria']:
		color = (0,0,255) if remove_bac else (0,255,0)
		x, y = mouse
		cv2.rectangle(img, (x-5, y-5), (x+5, y+5), color, -1)

	img = cv2.resize(img, (0,0), fx=scale, fy=scale)
	cv2.imshow(window_name, img)

def display_original():
	if mode == 'bacteria':
		img = image.org1.copy()
	if mode == 'nucleus':
		img = image.org2.copy()
	if mode == 'cytoplasm':
		img = image.org3.copy()
	color = (255, 255, 0)
	cv2.circle(img, mouse, 7, color, -1)
	img = cv2.resize(img, (0,0), fx=scale, fy=scale)
	cv2.imshow('Original Image', img)

def pick_color():
	global selected

	# change selected
	selected = 0
	for key in image.cytoplasm:
		contour = image.cytoplasm[key]['contour']
		is_hovered = cv2.pointPolygonTest(contour,mouse,True)
		if is_hovered > 0:
			selected = image.cytoplasm[key]['label']

def mouse_event(event, x, y, flags, param):
	global mouse

	if mode in ['nucleus', 'cytoplasm']:

		if event == cv2.EVENT_LBUTTONUP:

			x = int(x/scale)
			y = int(y/scale)

			# select and change cytoplasm
			# currently cannot get this cytoplasm back, I hope...
			for key in image.cytoplasm:
				if image.cytoplasm[key]['label'] == 0: continue
				contour = image.cytoplasm[key]['contour']
				is_clicked = cv2.pointPolygonTest(contour,(x,y),True)
				if is_clicked > 0:
					image.cytoplasm[key]['label'] = selected

			# remove nucleus if selected = background
			# currently cannot get this nucleus back, I hope...
			if not selected:
				for key in image.nucleus:
					if image.nucleus[key]['label'] == 0: continue
					contour = image.nucleus[key]['contour']
					is_clicked = cv2.pointPolygonTest(contour,(x,y),True)
					if is_clicked > 0:
						image.nucleus[key]['label'] = 0

			display(image.draw())
			display_original()
		
		# update cursor position
		if event == cv2.EVENT_MOUSEMOVE:
			x = int(x/scale)
			y = int(y/scale)
			mouse = (x,y)
			display(image.canvas)
			display_original()

	elif mode in ['bacteria']:

		if event == cv2.EVENT_LBUTTONUP:

			x = int(x/scale)
			y = int(y/scale)

			# if remove bacteria
			if remove_bac:
				for key in image.bacteria:
					contour = image.bacteria[key]['contour']
					is_clicked = cv2.pointPolygonTest(contour,(x,y),True) > -9
					if is_clicked:
						image.bacteria[key]['label'] = 0
			else:
				image.count1 += 1
				con = np.array([
								[[x-5, y-5]],
								[[x-5, y+5]],
								[[x+5, y+5]],
								[[x+5, y-5]],
							])
				image.bacteria[image.count1] = {
					'contour': con,
					'centroid': [x, y],
					'label': 1
				}

			display(image.draw())
			display_original()
		
		# update cursor position
		if event == cv2.EVENT_MOUSEMOVE:
			x = int(x/scale)
			y = int(y/scale)
			mouse = (x,y)
			display(image.canvas)
			display_original()

def display_draw(src):

	# copy from source
	img = src.copy()

	# change color depend on what kind of region
	if mode == 'nucleus':
		color = (0, 0, 255)
	elif mode == 'cytoplasm':
		color = (255, 0, 255)

	# draw new border
	for i in range(len(new_contour)):
		x, y = new_contour[i][0]
		if i > 0:
			x0, y0 = new_contour[i-1][0]
			cv2.line(img, (x, y), (x0, y0), color, 1)
		if i == len(new_contour)-1:
			xm, ym = mouse
			cv2.line(img, (x, y), (xm, ym), color, 1)
		cv2.circle(img, (x, y), 1, color, -1)

	# display
	img = cv2.resize(img, (0,0), fx=scale, fy=scale)
	cv2.imshow(window_name, img)

def mouse_event_draw(event, x, y, flags, param):
	global mouse, new_contour, isPressed

	# select and change label
	if event == cv2.EVENT_LBUTTONDOWN:
		isPressed = True

	# select and change label
	if event == cv2.EVENT_LBUTTONUP:
		isPressed = False

	# update cursor position
	if event == cv2.EVENT_MOUSEMOVE:

		x = int(x/scale)
		y = int(y/scale)
		mouse = (x,y)

		# append the drawing line
		if isPressed:
			if len(new_contour):
				current_point = np.array([x, y]) * scale
				previous_point = np.array(new_contour[-1][0]) * scale
				if np.linalg.norm(current_point-previous_point) > 10:
					new_contour.append([[x, y]])
			else:
				new_contour.append([[x, y]])

		display_draw(image.canvas2 if mode == 'nucleus' else image.canvas3)
		display_original()

	# # update cursor position
	# if event == cv2.EVENT_MOUSEMOVE:

	# 	if isPressed:
	# 		x = int(x/scale)
	# 		y = int(y/scale)
	# 		new_contour.append([[x, y]])
	# 		display_draw(image.canvas2 if mode == 'nucleus' else image.canvas3)
	# 		display_original()

	# 	else:
	# 		x = int(x/scale)
	# 		y = int(y/scale)
	# 		mouse = (x,y)
	# 		display_draw(image.canvas2 if mode == 'nucleus' else image.canvas3)
	# 		display_original()

def draw_nucleus():
	global new_contour

	new_contour = []
	while True:

		display_draw(image.canvas2)
		cv2.setMouseCallback(window_name, mouse_event_draw)
		key = cv2.waitKeyEx(0)
		if key in [ord('q'), 27, 230]:
			break
		if key in [ord('v'), 205]:
			if len(new_contour) > 2:
				con = np.array(new_contour)
				area = cv2.contourArea(con)
				if area < 400: break
				if area > 80000: break
				image.count2 += 1
				M = cv2.moments(con)
				cx = int(M["m10"] / M["m00"])
				cy = int(M["m01"] / M["m00"])
				image.nucleus[image.count2] = {
					'contour': con,
					'centroid': [cx, cy],
					'label': 1
				}
			break
		if key in [ord('b'), 212]:
			if len(new_contour): new_contour.pop()

def draw_cytoplasm():
	global new_contour

	new_contour = []
	while True:

		display_draw(image.canvas3)
		cv2.setMouseCallback(window_name, mouse_event_draw)
		key = cv2.waitKeyEx(0)
		if key in [ord('q'), 27, 230]:
			break
		if key in [ord('v'), 205]:
			if len(new_contour) > 2:
				con = np.array(new_contour)
				area = cv2.contourArea(con)
				if area < 400: break
				if area > 80000: break
				image.count3 += 1
				M = cv2.moments(con)
				cx = int(M["m10"] / M["m00"])
				cy = int(M["m01"] / M["m00"])
				image.cytoplasm[image.count3] = {
					'contour': con,
					'centroid': [cx, cy],
					'label': image.count3
				}
			break
		if key in [ord('b'), 212]:
			if len(new_contour): new_contour.pop()

class CellsImage:
	def __init__(self, index):

		# image index
		self.index = index

		# set path
		name = names[self.index]
		original1_path = os.path.join(original_path, name + '001.tif')
		original2_path = os.path.join(original_path, name + '002.tif')
		original3_path = os.path.join(original_path, name + '003.tif')
		cellprofiler1_path = os.path.join(cellprofiler_path, name + '001_FITC.png')
		cellprofiler2_path = os.path.join(cellprofiler_path, name + '002_hoechst.png')
		cellprofiler3_path = os.path.join(cellprofiler_path, name + '003_Evan.png')
		self.label1_path = os.path.join(save_path, name + '001.json')
		self.label2_path = os.path.join(save_path, name + '002.json')
		self.label3_path = os.path.join(save_path, name + '003.json')
		cv2.setWindowTitle(window_name, self.label3_path)

		# load image
		self.org1 = cv2.imread(original1_path)
		self.org2 = cv2.imread(original2_path)
		self.org3 = cv2.imread(original3_path)
		# self.org1 = self.automatic_brightness_and_contrast(self.org1, alpha_ratio=10, beta_ratio=15)[0]
		# self.org2 = self.automatic_brightness_and_contrast(self.org2)[0]
		# self.org3 = self.automatic_brightness_and_contrast(self.org3)[0]
		self.cpf1 = cv2.imread(cellprofiler1_path)
		self.cpf2 = cv2.imread(cellprofiler2_path)
		self.cpf3 = cv2.imread(cellprofiler3_path)

		# get resolution and create blank template
		self.height, self.width = np.shape(self.org3)[:2]
		blank = np.zeros([self.height, self.width, 3], np.uint8)
		
		# create nucleus border
		self.edge2 = blank.copy()
		self.edge2[np.where((self.cpf2==[0,255,255]).all(axis=2))] = [255,255,255]

		# create cytoplasm border
		self.edge3 = blank.copy()
		self.edge3[np.where((self.cpf3==[0,0,255]).all(axis=2))] = [255,255,255]

		# load or make bacteria data
		if os.path.exists(self.label1_path):
			self.load_bacteria()
		else:
			self.make_bacteria()

		# load or make nucleus data
		if os.path.exists(self.label2_path):
			self.load_nucleus()
		else:
			self.make_nucleus()

		# load or make cytoplasm data
		if os.path.exists(self.label3_path):
			self.load_cytoplasm()
		else:
			self.make_cytoplasm()

		# GUI parameters 
		self.canvas = blank.copy()
		self.canvas2 = self.org2
		self.canvas3 = cv2.bitwise_or(self.org3, self.edge2)

		# update canvas
		self.draw()

	# Automatic brightness and contrast optimization with optional histogram clipping
	def automatic_brightness_and_contrast(self, image, clip_hist_percent=1, alpha_ratio=1, beta_ratio=1):

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Calculate grayscale histogram
		hist = cv2.calcHist([gray],[0],None,[256],[0,256])
		hist_size = len(hist)

		# Calculate cumulative distribution from the histogram
		accumulator = []
		accumulator.append(float(hist[0]))
		for _index in range(1, hist_size):
			accumulator.append(accumulator[_index -1] + float(hist[_index]))

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

	def get_centers(self, profile):
		centers = []
		for _, row in profile.iterrows():
			if 'Location_Center_X' in row:
				x = int(row['Location_Center_X'])
				y = int(row['Location_Center_Y'])
			elif 'AreaShape_Center_X' in row:
				x = int(row['AreaShape_Center_X'])
				y = int(row['AreaShape_Center_Y'])
			centers.append([x, y])
		return centers

	def make_bacteria(self):

		# contain nucleus infomation
		self.bacteria = dict()

		# load center
		profile = bacteria_profiles.loc[bacteria_profiles['ImageNumber'] == self.index+1]
		centers = self.get_centers(profile)

		# identify each bac
		count = 0
		padding = 7
		for x, y in centers:
			count += 1
			con = np.array([
							[[x-padding, y-padding]],
							[[x-padding, y+padding]],
							[[x+padding, y+padding]],
							[[x+padding, y-padding]],
						])
			self.bacteria[count] = {
				'contour': con,
				'centroid': [x, y],
				'label': 1
			}

		self.count1 = count

	def make_nucleus(self):

		# contain nucleus infomation
		self.nucleus = dict()

		# find contours
		edge = cv2.cvtColor(self.edge2, cv2.COLOR_BGR2GRAY)
		edge = cv2.rectangle(edge, (0, 0), (self.width-1, self.height-1), 255, 1) 
		edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
		ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

		# load center
		profile = nucleus_profiles.loc[nucleus_profiles['ImageNumber'] == self.index+1]
		centers = self.get_centers(profile)

		# identify each cell
		count = 0
		for con, hie in zip(contours, hierarchy[0]):

			# filter out the noise
			area = cv2.contourArea(con)
			if area < 200: continue
			if area > 80000: continue
			# if hie[2] != -1: continue

			# calculate centroid
			M = cv2.moments(con)
			cx = int(M["m10"] / M["m00"])
			cy = int(M["m01"] / M["m00"])

			# calculate min diff
			for x, y in centers.copy():
				if abs(cx-x) + abs(cy-y) < 25:

					# make label
					count += 1
					self.nucleus[count] = {
						'contour': con,
						'centroid': [cx, cy],
						'label': 1
					}

					# reduce meaning calculation
					centers.remove([x, y])

		self.count2 = count

	def make_cytoplasm(self):

		# contain cytoplasm infomation
		self.cytoplasm = dict()

		# find contours
		edge = cv2.cvtColor(self.edge3, cv2.COLOR_BGR2GRAY)
		edge = cv2.rectangle(edge, (0, 0), (self.width-1, self.height-1), 255, 1) 
		edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
		ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

		# load center
		profile = cytoplasm_profiles.loc[cytoplasm_profiles['ImageNumber'] == self.index+1]
		centers = self.get_centers(profile)

		# identify each cell
		count = 0
		for con, hie in zip(contours, hierarchy[0]):

			# filter out the noise
			area = cv2.contourArea(con)
			if area < 400: continue
			if area > 80000: continue
			# if hie[2] != -1: continue

			# calculate centroid
			M = cv2.moments(con)
			cx = int(M["m10"] / M["m00"])
			cy = int(M["m01"] / M["m00"])

			# calculate min diff
			for x, y in centers.copy():
				if abs(cx-x) + abs(cy-y) < 25:

					# make label
					count += 1
					self.cytoplasm[count] = {
						'contour': con,
						'centroid': [cx, cy],
						'label': count
					}

					# reduce meaning calculation
					centers.remove([x, y])

		self.count3 = count

	# load bacteria from file
	def load_bacteria(self):
		with open(self.label1_path) as file:
			d = json.load(file)
			self.bacteria = {}
			for key in d:
				self.bacteria[key] = {
					'contour': np.array(d[key]['contour']),
					'centroid': d[key]['centroid'],
					'label': d[key]['label'],
				}
			self.count1 = len(self.bacteria)

	# load nucleus from file
	def load_nucleus(self):
		with open(self.label2_path) as file:
			d = json.load(file)
			self.nucleus = {}
			for key in d:
				self.nucleus[key] = {
					'contour': np.array(d[key]['contour']),
					'centroid': d[key]['centroid'],
					'label': d[key]['label'],
				}
			self.count2 = len(self.nucleus)

	# load cytoplasm from file
	def load_cytoplasm(self):
		with open(self.label3_path) as file:
			d = json.load(file)
			self.cytoplasm = {}
			for key in d:
				self.cytoplasm[key] = {
					'contour': np.array(d[key]['contour']),
					'centroid': d[key]['centroid'],
					'label': d[key]['label'],
				}
			self.count3 = len(self.cytoplasm)

	# save label to file
	def save_bacteria(self):
		with open(self.label1_path, 'w') as file:
			d = {}
			for key in self.bacteria:
				d[key] = {
					'contour': self.bacteria[key]['contour'].tolist(),
					'centroid': self.bacteria[key]['centroid'],
					'label': self.bacteria[key]['label'],
				}
			json.dump(d, file)
		print('save bacteria:', self.label1_path)

	# save label to file
	def save_nucleus(self):
		with open(self.label2_path, 'w') as file:
			d = {}
			for key in self.nucleus:
				d[key] = {
					'contour': self.nucleus[key]['contour'].tolist(),
					'centroid': self.nucleus[key]['centroid'],
					'label': self.nucleus[key]['label'],
				}
			json.dump(d, file)
		print('save nucleus:', self.label2_path)

	# save label to file
	def save_cytoplasm(self):
		with open(self.label3_path, 'w') as file:
			d = {}
			for key in self.cytoplasm:
				d[key] = {
					'contour': self.cytoplasm[key]['contour'].tolist(),
					'centroid': self.cytoplasm[key]['centroid'],
					'label': self.cytoplasm[key]['label'],
				}
			json.dump(d, file)
		print('save cytoplasm:', self.label3_path)

	def save_all(self):
		self.save_bacteria()
		self.save_nucleus()
		self.save_cytoplasm()
		print('saved')

	# draw canvas
	def draw(self):

		# draw nucleus border
		nucleus_border = np.zeros([self.height, self.width, 3], np.uint8)
		for key in self.nucleus:
			if self.nucleus[key]['label'] == 0: continue
			contour = self.nucleus[key]['contour']
			cv2.drawContours(nucleus_border, [contour], -1, (255, 255, 255), 1)
		self.canvas3 = cv2.bitwise_or(self.org3, nucleus_border)	# should I update cyto canvas here?

		# Bacteria + Cell Profiler
		if view == 0:

			self.canvas = self.cpf1.copy()

		# Bacteria + Region + Bacteria Index
		elif view == 1:

			# draw cell region
			overlay = np.zeros([self.height, self.width, 3], np.uint8)
			for key in self.bacteria:
				contour = self.bacteria[key]['contour']
				label = self.bacteria[key]['label']
				color = bac_colors[label]
				cv2.fillPoly(overlay, pts =[contour], color=color)
			self.canvas = cv2.addWeighted(self.org1, 1, overlay, 0.3, 0)

		# Nucleus + Cell Profiler
		elif view == 2:

			self.canvas = self.cpf2.copy()

		# Nucleus + Region + Cell Index
		elif view == 3:

			# draw cell region
			overlay = np.zeros([self.height, self.width, 3], np.uint8)
			for key in self.cytoplasm:
				contour = self.cytoplasm[key]['contour']
				label = self.cytoplasm[key]['label']
				color = cell_colors[label]
				cv2.fillPoly(overlay, pts =[contour], color=color)
			self.canvas = cv2.addWeighted(self.org2, 1, overlay, 0.3, 0)

			# draw cell index
			overlay = np.zeros([self.height, self.width, 3], np.uint8)
			for key in self.cytoplasm:
				cx, cy = self.cytoplasm[key]['centroid']
				label = self.cytoplasm[key]['label']
				if label == 0: continue
				cv2.putText(self.canvas, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA) 

			# draw nucleus border
			self.canvas = cv2.bitwise_or(self.canvas, nucleus_border)

		# Cytoplasm + Cell Profiler
		if view == 4:

			self.canvas = self.cpf3.copy()

		# Cytoplasm + Region + Cell Index
		elif view == 5:

			# draw cell region
			overlay = np.zeros([self.height, self.width, 3], np.uint8)
			for key in self.cytoplasm:
				contour = self.cytoplasm[key]['contour']
				label = self.cytoplasm[key]['label']
				color = cell_colors[label]
				cv2.fillPoly(overlay, pts =[contour], color=color)
			self.canvas = cv2.addWeighted(self.org3, 1, overlay, 0.3, 0)

			# draw cell index
			overlay = np.zeros([self.height, self.width, 3], np.uint8)
			for key in self.cytoplasm:
				cx, cy = self.cytoplasm[key]['centroid']
				label = self.cytoplasm[key]['label']
				if label == 0: continue
				cv2.putText(self.canvas, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA) 

		# return operated image
		return self.canvas

def auto():
	for i in range(total):
		print(i, 'in', total)
		image = CellsImage(i)
		image.make_bacteria()
		image.make_nucleus()
		image.make_cytoplasm()
		image.save_all()
	exit()

if __name__ == '__main__':

	# set path
	root = os.path.dirname(os.path.realpath(__file__))
	direct = 'S1/Plate_12/Testing_set'
	print(direct)
	original_path = os.path.join(root, "data/Reference_gene", direct)
	cellprofiler_path = os.path.join(root, "data/Output_Reference_gene", direct)
	save_path = os.path.join(root, "data/output_localize", direct)

	# create save path if not exists
	if not os.path.exists(save_path): os.makedirs(save_path)

	# get all detected cells from cellprofiler
	bacteria_profiles = pd.read_csv(os.path.join(cellprofiler_path, 'Reference_Bac.csv'))
	nucleus_profiles = pd.read_csv(os.path.join(cellprofiler_path, 'Reference_Nucleus.csv'))
	cytoplasm_profiles = pd.read_csv(os.path.join(cellprofiler_path, 'Reference_Cell.csv'))

	# set window
	scale = 0.9 # change screen size
	mouse = (0, 0)
	window_name = 'default'
	cv2.namedWindow (window_name, flags=cv2.WINDOW_AUTOSIZE)

	# Initialize parameters
	files = sorted(os.listdir(original_path))
	total = len(files) // 3
	names = [files[3*i][:-7] for i in range(total)]
	view, index = 1, 0 # change image index
	modes = ['bacteria', 'bacteria', 'nucleus', 'nucleus', 'cytoplasm', 'cytoplasm']
	mode = modes[view]
	image = CellsImage(index)
	selected = 0 		# select cell region
	remove_bac = False  # remove or add bac
	isPressed = False

	# auto generate localization
	auto()

	# loop over different field
	while True:

		# dsiplay
		display(image.canvas)
		display_original()

		# user feedback
		cv2.setMouseCallback(window_name, mouse_event)
		key = cv2.waitKeyEx(0)
		if key in [ord('q'), 27, 230]:
			break
		elif key in [ord(' '), 32]:
			if mode in ['nucleus', 'cytoplasm']:
				pick_color()
			if mode in ['bacteria']:
				remove_bac = not remove_bac
		elif key in [ord('a'), 2424832, 191]:
			image.save_all()
			index = max(index - 1, 0)
			image = CellsImage(index)
		elif key in [ord('d'), 2555904, 161]:
			image.save_all()
			index = min(index + 1, total - 1)
			image = CellsImage(index)
		elif key in [ord('w'), 2490368, 228]:
			view = max(view - 1, 0)
			mode = modes[view]
			display(image.draw())
		elif key in [ord('s'), 2621440, 203]:
			view = min(view + 1, len(modes) - 1)
			mode = modes[view]
			display(image.draw())
		elif key in [ord('z'), 188]:
			image.save_all()
		elif key in [ord('x'), 187]:
			pass
		elif key in [ord('c'), 225]:
			image.make_bacteria()
			image.make_nucleus()
			image.make_cytoplasm()
			display(image.draw())
		elif key in [ord('v'), 205]:
			if mode == 'nucleus':
				draw_nucleus()
			elif mode == 'cytoplasm':
				draw_cytoplasm()
			display(image.draw())
