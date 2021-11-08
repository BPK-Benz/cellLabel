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

import utils.calculation as calc
import utils.make_coco as make_coco


# mode = mark: color dictioanary
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

# mode = label: class dictionary
label_colors = {
    'not divide': cell_colors[1],
    'divide': cell_colors[11],
    'not border': cell_colors[6],
    'border': cell_colors[16],
}

def check_contour_boundary(contour, w, h): # check criteria for boundary cell

    touch = 0
    touch_threshold = 20 # distance touching boarder
    margin = 10 # height from boarder

    for i in range(len(contour)):
        x1, y1 = contour[i-1][0]
        x2, y2 = contour[i][0]
        # left
        if (x1 < margin) and (x2 < margin):
            touch += abs(y1 - y2)
        # right
        if (x1 > w - margin) and (x2 > w - margin):
            touch += abs(y1 - y2)
        # top
        if (y1 < margin) and (y2 < margin):
            touch += abs(x1 - x2)
        # bottom
        if (y1 > h - margin) and (y2 > h - margin):
            touch += abs(x1 - x2)

    return touch > touch_threshold

def create_coco():
    return {
        "info": make_coco.add_info(),
        "licenses": make_coco.add_licenses(),
        "categories": make_coco.add_categories(),
        "images": [],
        "annotations": [],
    }

def get_features(image, mask): # generating features for classical machine learning

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
    if area < 1000: return 

    # calculate centroid
    cx, cy = calc.centroid(contour) # calc is a library in utils folder to calcualte any features

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


class cellPlate: # managing folders for source and target 
    def __init__(self, navigation):

        # get absolute
        self.root_path = os.path.dirname(os.path.realpath(__file__))
        self.navigation = navigation
    
        # input paths
        self.original_path = os.path.join("r", self.root_path, "data/Enhance_image", self.navigation)
        self.cellprofiler_path = os.path.join("r", self.root_path, "data/Output_CP", self.navigation)

        # output paths, hint: put "r" to usable all os for reading pathway
        self.localize_path = os.path.join("r", self.root_path, "data/output_localize_ip", self.navigation)
        self.image_path = os.path.join("r", self.root_path, "data/output_image_ip", self.navigation)
        self.segment1_path = os.path.join("r", self.root_path, "data/output_segment1_ip", self.navigation)
        self.segment3_path = os.path.join("r", self.root_path, "data/output_segment3_ip", self.navigation)
        self.coco_path = os.path.join("r", self.root_path, "data/output_coco_ip", self.navigation)
        self.properties_path = os.path.join("r", self.root_path, "data/output_properties_ip", self.navigation)

        # create save path if not exists
        if not os.path.exists(self.localize_path): os.makedirs(self.localize_path)
        if not os.path.exists(self.image_path): os.makedirs(self.image_path)
        if not os.path.exists(self.segment1_path): os.makedirs(self.segment1_path)
        if not os.path.exists(self.segment3_path): os.makedirs(self.segment3_path)
        if not os.path.exists(self.coco_path): os.makedirs(self.coco_path)
        if not os.path.exists(self.properties_path): os.makedirs(self.properties_path)

        # get all detected cells from cellprofiler to guide image localization
        self.bacteria_profiles = pd.read_csv(os.path.join(self.cellprofiler_path, 'Reference_Bac.csv'))
        self.nucleus_profiles = pd.read_csv(os.path.join(self.cellprofiler_path, 'Reference_Nucleus.csv'))
        self.cytoplasm_profiles = pd.read_csv(os.path.join(self.cellprofiler_path, 'Reference_Cell.csv'))

        # select image
        files = sorted(os.listdir(self.original_path))
        self.names = [file[:-7] for file in files if file.endswith('001.tif')]
        self.total = len(self.names)
        # self.index = <title index> - 1
        self.index = 0

        # modes
        self.modes = ['mark', 'draw_nucleus', 'draw_cytoplasm', 'label_divide', 'label_border']
        self.mode = 'mark'

        # mode = mark: select source
        self.views = ['original', 'cellprofiler', 'overlay']
        self.view = 'overlay'

        # mode = mark: select channel
        self.channels = ['bacteria', 'nucleus', 'cell']
        self.channel = 'nucleus'

        # mode = mark: pick
        self.selected = 0        # select cell region
        self.remove_bac = False  # remove or add bac

        # mode = draw: new contour
        self.new_contour = []

        # mode = label
        self.init_border = True

        # window
        self.scale = 0.9
        self.mouse = 0, 0

        self.is_export = True  # to save time export should be False and changing it when localization and labeling are completed
        self.is_auto = True   # If is_export is True, is_auto shoul be True as well to automatic changing images (every change images are saved)

        # automatic 
        if self.is_auto:
            self.auto_generate()

        # load first image
        self.load_image()

    def auto_generate(self):

        for i in range(self.total):
            t0 = time.time()

            self.index = i
            self.load_image()
            self.save_all()

            elapse = time.time() - t0
            print('[ saved {} of {} | {} in {:.2f} seconds ]'.format(self.index+1, self.total, self.names[self.index], elapse))

    def load_image(self):

        # change window title
        name = self.names[self.index]
        # name = '008016-6-001001'
        title = "{} | {} of {}".format(name, self.index+1, self.total)
        root.winfo_toplevel().title(title)

        # set path
        ## For object localization
        original1_path = os.path.join(self.original_path, name + '001.tif')
        original2_path = os.path.join(self.original_path, name + '002.tif')
        original3_path = os.path.join(self.original_path, name + '003.tif')
        cellprofiler1_path = os.path.join(self.cellprofiler_path, name + '001_FITC.png')
        cellprofiler2_path = os.path.join(self.cellprofiler_path, name + '002_hoechst.png')
        cellprofiler3_path = os.path.join(self.cellprofiler_path, name + '003_Evan.png')
        self.local1_path = os.path.join(self.localize_path, name + '001.csv')
        self.local2_path = os.path.join(self.localize_path, name + '002.json')
        self.local3_path = os.path.join(self.localize_path, name + '003.json')

        # load image
        ## 1. Download raw image and automatically enhance image by line 3-6
        ## 2. Download directly enhance image that perform out of this script
        self.org1 = cv2.imread(original1_path)
        self.org2 = cv2.imread(original2_path)
        self.org3 = cv2.imread(original3_path)
        # self.org1 = self.automatic_brightness_and_contrast(self.org1, alpha_ratio=10, beta_ratio=15)[0]
        # self.org2 = self.automatic_brightness_and_contrast(self.org2)[0]
        # self.org3 = self.automatic_brightness_and_contrast(self.org3)[0]

        ## Download cellprofiler output
        self.cpf1 = cv2.imread(cellprofiler1_path)
        self.cpf2 = cv2.imread(cellprofiler2_path)
        self.cpf3 = cv2.imread(cellprofiler3_path)

        # filter cell profile
        self.profile2 = self.filter_profile(self.nucleus_profiles, name)
        self.profile3 = self.filter_profile(self.cytoplasm_profiles, name)

        # get resolution and create blank template
        self.height, self.width = np.shape(self.org3)[:2]
        self.blank1 = np.zeros([self.height, self.width], np.uint8)
        self.blank3 = np.zeros([self.height, self.width, 3], np.uint8)
        
        # create nucleus border
        self.edge2 = self.blank3.copy()
        self.edge2[np.where((self.cpf2==[0,255,255]).all(axis=2))] = [255,255,255]

        # create cytoplasm border
        self.edge3 = self.blank3.copy()
        self.edge3[np.where((self.cpf3==[0,0,255]).all(axis=2))] = [255,255,255]

        # load or make bacteria data
        if os.path.exists(self.local1_path):
            self.load_bacteria()
        else:
            self.make_bacteria2()

        # load or make nucleus data
        if os.path.exists(self.local2_path):
            self.load_nucleus()
        else:
            self.make_nucleus()  # For only the first time

        # load or make cytoplasm data
        if os.path.exists(self.local3_path):
            self.load_cytoplasm()
        else:
            self.make_cytoplasm()  # For only the first time

        # GUI parameters 
        self.canvas = self.blank3.copy()
        self.canvas2 = self.org2
        self.canvas3 = cv2.bitwise_or(self.org3, self.edge2)  # or operation between cell and nuclues

        # update canvas
        if self.mode in ['mark']:
            self.display_mark()
        elif self.mode in ['draw_nucleus', 'draw_cell']:
            self.display_draw()
        elif self.mode in ['label_divide', 'label_border']:
            self.display_label()

    def pick_color(self):

        # change selected
        self.selected = 0
        for key in self.cytoplasm:
            contour = self.cytoplasm[key]['contour']
            is_hovered = cv2.pointPolygonTest(contour,self.mouse,True)
            if is_hovered > 0:
                self.selected = self.cytoplasm[key]['index']

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

    def filter_profile(self, profiles, name): # How to read raw image from columbus: row, column and field
        metadata_row = int(name[0:3])
        metadata_col = int(name[3:6])
        metadata_field = int(name.split('-')[1])
        profile = profiles.loc[
                ( profiles['Metadata_Row'] == metadata_row ) &
                ( profiles['Metadata_Column'] == metadata_col ) & 
                ( profiles['Metadata_Field'] == metadata_field ) 
        ]
        return profile

    def get_centers(self, profile): # to guide center of each cell or nucleus
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

        # contain bacteria infomation
        self.bacteria = dict()

        # load center
        profile = self.bacteria_profiles.loc[self.bacteria_profiles['ImageNumber'] == self.index+1]
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
                'index': 1,
                'label': 1,
                'contour': con,
                'centroid': [x, y],
            }

        self.count1 = count

    def make_bacteria2(self): # to guide bacteria detection from image processing: adaptive thresholding, function of cv2

        # contain bacteria infomation
        self.bacteria = []

        img = self.org1.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(gray_img ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,2) # ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C

        # get contour
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # create an empty image for contours

        # draw the contours on the empty image
        c = 0 # count number of bacteria that passed inclusion criteria
        cm = [] # keep location of bacteria that passed inclusion criteria
        out = 0 # count number of bacteria that be a member of exclusion criteria

        isolated_count = 0 # Green: number is 1
        cluster_count = 0 # Red: number is 2
        semi_count = 0 # Orange: number is 1
        minArea = 20
        maxArea = 300
        clusterArea = 60
        clusterRatio = 0.95
        redunDist = 10

        for cnt in contours:
            area = cv2.contourArea(cnt) # Measure area
            
            if area < minArea or area > maxArea:
                continue

            M=cv2.moments(cnt) # Measure centroid
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            convex_hull = cv2.convexHull(cnt) # Measure convex hull
            convex_hull_area = cv2.contourArea(convex_hull) # Measure area of convex hull
            ratio = area / convex_hull_area # find the criteria of cluster or isolated
        
            if (cX, cY) not in cm:
                if c != 0:
                    dist =int(distance.euclidean( (cX,cY), cm[c-1] ))
                    
                    if dist > redunDist:
                        if ratio < clusterRatio:
                            if area > clusterArea:
                                self.bacteria.append([cX, cY, 'cluster'])
                            else:
                                self.bacteria.append([cX, cY, 'semi'])
                        else:
                            self.bacteria.append([cX, cY, 'isolated'])
                else:
                    if ratio < clusterRatio and area > clusterArea:
                        self.bacteria.append([cX, cY, 'cluster'])
                    else:
                        self.bacteria.append([cX, cY, 'isolated'])
                
                    dist = 0

                cm = cm + [(cX,cY)]
                c += 1

        self.count1 = len(self.bacteria)

    def make_nucleus(self): # to find the location and contour of nucleus from result of cellprofiler

        # contain nucleus infomation
        self.nucleus = dict()

        # find contours
        edge = cv2.cvtColor(self.edge2, cv2.COLOR_BGR2GRAY)
        edge = cv2.rectangle(edge, (0, 0), (self.width-1, self.height-1), 255, 1) 
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # calculate areas and centroids
        areas = [cv2.contourArea(contour) for contour in contours]
        centroids = []
        for con, hie in zip(contours, hierarchy[0]):
            M = cv2.moments(con)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append([cx, cy])

        # skip 
        skips = []
        skips += [i for i in range(len(areas)) if areas[i] < 200]
        skips += [i for i in range(len(areas)) if areas[i] > 80000]

        # load center from cell profiler
        centers = self.get_centers(self.profile2)

        # identify each nucleus
        count = 1
        for x, y in centers:
            min_area = 80001
            for i in range(len(contours)):
                if i in skips: continue
                cx, cy = centroids[i]
                area = areas[i]
                contour = contours[i]
                if abs(cx-x) + abs(cy-y) < 25 and area < min_area:
                    min_area = area
                    skips.append(i)
                    self.nucleus[count] = {
                        'index': 1,
                        'divide': 0,
                        'border': 0,
                        'contour': contour,
                        'centroid': [cx, cy],
                    }

            if not min_area == 80001:
                count += 1

        self.count2 = count - 1

    def label_boundary(self):

        for key in self.cytoplasm:
            index = self.cytoplasm[key]['index']
            border = self.cytoplasm[key]['border']
            contour = self.cytoplasm[key]['contour']

            # calculate boundary class
            if check_contour_boundary(contour, self.width, self.height):

                # label as boundary
                for key2 in self.cytoplasm:
                    index2 = self.cytoplasm[key2]['index']
                    if index == index2:
                        self.cytoplasm[key2]['border'] = 1

    def make_cytoplasm(self): # to find the location and contour of cytoplasm from result of cellprofiler

        # contain cytoplasm infomation
        self.cytoplasm = dict()

        # find contours
        edge = cv2.cvtColor(self.edge3, cv2.COLOR_BGR2GRAY)
        edge = cv2.rectangle(edge, (0, 0), (self.width-1, self.height-1), 255, 1) 
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # calculate areas and centroids
        areas = [cv2.contourArea(contour) for contour in contours]
        centroids = []
        for con, hie in zip(contours, hierarchy[0]):
            M = cv2.moments(con)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append([cx, cy])

        # skip 
        skips = []
        skips += [i for i in range(len(areas)) if areas[i] < 200]
        skips += [i for i in range(len(areas)) if areas[i] > 80000]

        # load center from cell profiler
        centers = self.get_centers(self.profile3)

        # identify each cell
        count = 1
        for x, y in centers:
            min_area = 80001
            for i in range(len(contours)):
                if i in skips: continue
                cx, cy = centroids[i]
                area = areas[i]
                contour = contours[i]
                if abs(cx-x) + abs(cy-y) < 25 and area < min_area:
                    min_area = area
                    skips.append(i)
                    self.cytoplasm[count] = {
                        'index': count,
                        'divide': 0,
                        'border': 0,
                        'contour': contour,
                        'centroid': [cx, cy],
                    }

            if not min_area == 80001:
                count += 1

        self.count3 = count - 1

        if self.init_border:
            self.label_boundary()

        # # check positions
        # image = self.edge3.copy()

        # for x, y in centers.copy():

        #     image = cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # for con, hie in zip(contours, hierarchy[0]):

        #     area = cv2.contourArea(con)
        #     if area < 400: continue
        #     if area > 80000: continue

        #     M = cv2.moments(con)
        #     cx = int(M["m10"] / M["m00"])
        #     cy = int(M["m01"] / M["m00"])
            
        #     image = cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

        # print('len(contours)', len(contours))
        # cv2.imshow('image', image)
        # key = cv2.waitKey(0)
        # if key in [ord('q')]: exit()

    # load bacteria from file that already finished labelling
    def load_bacteria(self):
        df = pd.read_csv(self.local1_path)
        self.bacteria = []
        for index, row in df.iterrows():
            self.bacteria.append([row['x'], row['y'], row['group']]) 

    # load nucleus from file that already finished labelling
    def load_nucleus(self):
        with open(self.local2_path) as file:
            d = json.load(file)
            self.nucleus = {}
            for key in d:
                if not 'index' in d[key]:
                    d[key]['index'] = d[key]['label']
                if not 'divide' in d[key]:
                    d[key]['divide'] = 0
                if not 'border' in d[key]:
                    d[key]['border'] = 0
                self.nucleus[key] = {
                    'index': d[key]['index'],
                    'divide': d[key]['divide'],
                    'border': d[key]['border'],
                    'contour': np.array(d[key]['contour']),
                    'centroid': d[key]['centroid'],
                }
            self.count2 = len(self.nucleus)

    # load cytoplasm from file that already finished labelling
    def load_cytoplasm(self):

        ever_calc_border = False

        with open(self.local3_path) as file:
            d = json.load(file)
            self.cytoplasm = {}
            for key in d:
                if not 'index' in d[key]:
                    d[key]['index'] = d[key]['label']
                if not 'divide' in d[key]:
                    d[key]['divide'] = 0
                if not 'border' in d[key]:
                    d[key]['border'] = 0
                if d[key]['border'] > 0:
                    ever_calc_border = True
                self.cytoplasm[key] = {
                    'index': d[key]['index'],
                    'divide': d[key]['divide'],
                    'border': d[key]['border'],
                    'contour': np.array(d[key]['contour']),
                    'centroid': d[key]['centroid'],
                }
            self.count3 = len(self.cytoplasm)

        if not ever_calc_border and self.init_border:
            self.label_boundary()

    # save label to csv file for bacteria after moving to next image
    def save_bacteria(self):
        df = pd.DataFrame(columns=['x', 'y', 'group'])
        for x, y, group in self.bacteria:
            d = {
                'x': x, 'y': y,
                'group': group,
            }
            df = df.append(d, ignore_index=True)
        df.to_csv(self.local1_path, index=False)

    # save label to json file for nucleus after moving to next image
    def save_nucleus(self):
        with open(self.local2_path, 'w') as file:
            d = {}
            for key in self.nucleus:
                d[key] = {
                    'index': self.nucleus[key]['index'],
                    'divide': self.nucleus[key]['divide'],
                    'border': self.nucleus[key]['border'],
                    'contour': self.nucleus[key]['contour'].tolist(),
                    'centroid': self.nucleus[key]['centroid'],
                }
            json.dump(d, file)

    # save label to json file for cytoplasm after moving to next image
    def save_cytoplasm(self):
        with open(self.local3_path, 'w') as file:
            d = {}
            for key in self.cytoplasm:
                d[key] = {
                    'index': self.cytoplasm[key]['index'],
                    'divide': self.cytoplasm[key]['divide'],
                    'border': self.cytoplasm[key]['border'],
                    'contour': self.cytoplasm[key]['contour'].tolist(),
                    'centroid': self.cytoplasm[key]['centroid'],
                }
            json.dump(d, file)

    def export_image(self): # to merge three channels together so we got an image

        # make image
        b = cv2.cvtColor(self.org2, cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(self.org1, cv2.COLOR_BGR2GRAY)
        r = cv2.cvtColor(self.org3, cv2.COLOR_BGR2GRAY)
        out_image = cv2.merge([b, g, r])

        # save image
        out_name = self.names[self.index] + '.png'
        out_path = os.path.join(self.image_path, out_name)
        cv2.imwrite(out_path, out_image)

    def export_segment(self): # to save labelled segment

        # segment classes
        # 1 = cell
        # 2 = nucleus
        # 3 = cell membrane

        mask_nucleus = self.blank1.copy()
        mask_cell = self.blank1.copy()
        mask_membrane = self.blank1.copy()

        # make nucleus mask
        for key in self.nucleus:

            # fill nucleus
            contour = self.nucleus[key]['contour']
            cv2.fillPoly(mask_nucleus, pts =[contour], color=255)

        # make cell and membrane mask
        for key in self.cytoplasm:

            index = self.cytoplasm[key]['index']
            if index == 0: continue

            # fill cell
            contour = self.cytoplasm[key]['contour']
            cv2.fillPoly(mask_cell, pts =[contour], color=255)

            # fill cell membrane
            mask = self.blank1.copy()
            for key2 in self.cytoplasm:
                index2 = self.cytoplasm[key2]['index']
                if index == index2:
                    contour = self.cytoplasm[key2]['contour']
                    cv2.fillPoly(mask, pts =[contour], color=255)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((8,8),np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_membrane, contours, -1, (255, 255, 255), 3)

        mask_nucleus =cv2.bitwise_and(mask_cell, mask_nucleus)

        # make segment1
        segment1 = self.blank1.copy()
        segment1[mask_cell == 255] = 1
        segment1[mask_nucleus == 255] = 2
        segment1[mask_membrane == 255] = 3

        # save segment1
        segment1_name = self.names[self.index] + '.png'
        segment1_path = os.path.join(self.segment1_path, segment1_name)
        cv2.imwrite(segment1_path, segment1)

        # make segment3
        segment3 = self.blank3.copy()
        segment3[mask_cell == 255] = (0, 0, 255)
        segment3[mask_nucleus == 255] = (255, 0, 0)
        segment3[mask_membrane == 255] = (255, 255, 255)

        # save segment3
        segment3_name = self.names[self.index] + '.png'
        segment3_path = os.path.join(self.segment3_path, segment3_name)
        cv2.imwrite(segment3_path, segment3)

    def export_coco(self):

        mask_nucleus = self.blank1.copy()
        mask_cell = self.blank1.copy().astype(np.int32)

        # make nucleus mask
        for key in self.nucleus:

            index = self.nucleus[key]['index']
            if index == 0: continue

            # fill nucleus
            contour = self.nucleus[key]['contour']
            cv2.fillPoly(mask_nucleus, pts =[contour], color=255)

        # make cytoplasm mask
        for key in self.cytoplasm:

            index = self.cytoplasm[key]['index']
            if index == 0: continue

            # store single cell region for preprocessing
            mask = self.blank1.copy()
            for key2 in self.cytoplasm:
                index2 = self.cytoplasm[key2]['index']
                if index == index2:
                    contour = self.cytoplasm[key2]['contour']
                    cv2.fillPoly(mask, pts =[contour], color=255)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((8,8),np.uint8))
            mask_cell[mask == 255] = index

            # cv2.imshow('mask', mask)
            # k = cv2.waitKey(0)
            # if k in [ord('q')]: break

        # mask_nucleus[mask_cell == 0] = 0

        # properties
        columns = [
            'Name', 'Replicate', 'Plate', 'Set',
            'Nucleus_X', 'Nucleus_Y', 'Nucleus_Area', 'Nucleus_Perimeter',
            'Nucleus_Compactness', 'Nucleus_Diameter', 'Nucleus_Eccentricity',
            'Nucleus_Extent', 'Nucleus_Extent_Axis', 'Nucleus_Solidity', 'Nucleus_Intensity',
            'Nucleus_MinAxis', 'Nucleus_MaxAxis', 'Nucleus_MinFeret', 'Nucleus_MaxFeret',
            'Cell_X', 'Cell_Y', 'Cell_Area', 'Cell_Perimeter',
            'Cell_Compactness', 'Cell_Diameter', 'Cell_Eccentricity',
            'Cell_MinAxis', 'Cell_MaxAxis', 'Cell_MinFeret', 'Cell_MaxFeret',
            'Nuc_Bac', 'Cyto_Bac', 'Nuc_Cell', 'Cyto_Cell'
        ]
        properties = pd.DataFrame([], columns = columns)
        if len(self.navigation.split('/')) == 3:
            r, p, s = self.navigation.split('/')
        else:
            r, p, s = '-', '-', self.navigation

        # coco
        count_image = 1
        count_annotation = 1
        coco = create_coco()
        skip = [0] # skip background and duplicate

        # write each
        for key in self.cytoplasm:

            index = self.cytoplasm[key]['index']
            if index in skip: continue
            skip.append(index)

            # store single cell region for calculation
            mask_n = self.blank1.copy()
            mask_c = self.blank1.copy()

            # get region info
            mask_c[mask_cell == index] = 255
            mask_n = cv2.bitwise_and(mask_c, mask_nucleus)

            # apply sequence of functions
            features2 = get_features(self.org2, mask_n.copy())
            features3 = get_features(self.org3, mask_c.copy())
            if features2 is None: continue
            if features3 is None: continue

            # Determine number of nucbac and cytobac based on image processing and manual annotation
            allbac = 0
            nucbac = 0
            cytobac = 0

            for x, y, group in self.bacteria:

                # get number of bacteria
                if group in ['isolated', 'semi']: # semi bacteria is proven by biologist and is counted one and doblec check agian by manual annotation
                    nbac = 1
                elif group in ['cluster']: # two bacteria are close
                    nbac = 2
                else:
                    nbac = 0

                # count
                if mask_c[y, x] == 255:
                    allbac += nbac
                    if mask_n[y, x] == 255:
                        nucbac += nbac
            cytobac = allbac - nucbac

            # Define what is nuccell and cytocell
            nuccell = int(nucbac >= cytobac) # cells that number of nucbac more than cytobac
            cytocell = int(cytobac > nucbac) # cyto cells that are the contrast of nuccell

            infect = None
            if allbac == 0:
                infect = 'non-infected'
            elif nucbac >= cytobac:
                infect = 'nuccell'
            else:
                infect = 'cytocell'

            # append data to dictionary
            cell = {

                # file
                'Name': self.names[self.index], 
                'Replicate': r, 
                'Plate': p, 
                'Set': s, 

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

                # label
                "divide": int(self.cytoplasm[key]['divide']),
                "border": int(self.cytoplasm[key]['border']),
                "infect": infect,
            }
            properties = properties.append(cell, ignore_index=True)

            # append nucleus data to coco json
            coco_object = {
                "channel": 'nucleus',
                "divide": int(self.cytoplasm[key]['divide']),
                "border": int(self.cytoplasm[key]['border']),
                "infect": infect,
                "iscrowd": 0,
                "area": features2['area'],
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
                "channel": 'cell',
                "divide": int(self.cytoplasm[key]['divide']),
                "border": int(self.cytoplasm[key]['border']),
                "infect": infect,
                "iscrowd": 0,
                "area": features3['area'],
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
            "file_name": self.image_path.replace(self.root_path + '/', '') + '/' + self.names[self.index] + '.png',
            "height": self.height,
            "width": self.width,
            "id": count_image,
        }
        coco['images'].append(coco_image)
        count_image += 1

        # save properties to csv file
        properties_name = self.names[self.index] + '.csv'
        properties_path = os.path.join(self.properties_path, properties_name)
        properties.to_csv(properties_path, index= False)

        # save coco to json file
        coco_name = self.names[self.index] + '.json'
        coco_path = os.path.join(self.coco_path, coco_name)
        with open(coco_path, 'w') as outfile:
            json.dump(coco, outfile)

    def save_all(self):
        self.save_bacteria()
        self.save_nucleus()
        self.save_cytoplasm()
        if self.is_export: 
            self.export_image()
            self.export_segment()
            self.export_coco()

    # create canvas -> display
    def display_mark(self):

        self.canvas = self.blank3.copy()

        if self.view == 'original':

            if self.channel == 'bacteria':
                self.canvas = self.org1

            elif self.channel == 'nucleus':
                self.canvas = self.org2

            elif self.channel == 'cell':
                self.canvas = self.org3

        elif self.view == 'cellprofiler':

            if self.channel == 'bacteria':
                self.canvas = self.cpf1

            elif self.channel == 'nucleus':
                self.canvas = self.cpf2

            elif self.channel == 'cell':
                self.canvas = self.cpf3

        elif self.view == 'overlay':

            if self.channel == 'bacteria':
                self.canvas = self.org1.copy()
                for cX, cY, group in self.bacteria:
                    color = {
                        'cluster': [0, 0, 255],
                        'semi': [0, 100, 200],
                        'isolated': [55, 200, 255],
                    }[group]
                    cv2.circle(self.canvas, (cX, cY), 8, color, 2)

                semi_count = len([b for b in self.bacteria if b[2]=='semi'])
                cluster_count = len([b for b in self.bacteria if b[2]=='cluster'])
                isolated_count = len([b for b in self.bacteria if b[2]=='isolated'])
                total_count = semi_count + cluster_count + isolated_count
                text = 'Number_semi_clusters:'+str(semi_count)
                text0 = 'Number_clusters:'+str(cluster_count)
                text1 = 'Number_isolated:'+str(isolated_count)
                text2 = "Number of Circular Blobs: " + str(total_count)

                # Create message
                ypos = 300
                
                cv2.putText(self.canvas, text, (100,ypos-30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 200), 2)
                cv2.putText(self.canvas, text0, (100,ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(self.canvas, text1, (100,ypos+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (55, 200, 0), 2)
                cv2.putText(self.canvas, text2, (100,ypos+60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 100, 0), 2)

            elif self.channel == 'nucleus':

                # draw nucleus border
                nucleus_border = self.blank3.copy()
                for key in self.nucleus:
                    if self.nucleus[key]['index'] == 0: continue
                    contour = self.nucleus[key]['contour']
                    cv2.drawContours(nucleus_border, [contour], -1, (255, 255, 255), 1)
                self.canvas3 = cv2.bitwise_or(self.org3, nucleus_border)    # should I update cyto canvas here?

                # draw cell region
                overlay = np.zeros([self.height, self.width, 3], np.uint8)
                for key in self.cytoplasm:
                    contour = self.cytoplasm[key]['contour']
                    index = self.cytoplasm[key]['index']
                    color = cell_colors[index]
                    cv2.fillPoly(overlay, pts =[contour], color=color)
                self.canvas = cv2.addWeighted(self.org2, 1, overlay, 0.3, 0)

                # draw cell index
                overlay = np.zeros([self.height, self.width, 3], np.uint8)
                for key in self.cytoplasm:
                    cx, cy = self.cytoplasm[key]['centroid']
                    index = self.cytoplasm[key]['index']
                    if index == 0: continue
                    cv2.putText(self.canvas, str(index), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA) 

                # draw nucleus border
                self.canvas = cv2.bitwise_or(self.canvas, nucleus_border)

            elif self.channel == 'cell':

                # draw cell region
                overlay = np.zeros([self.height, self.width, 3], np.uint8)
                for key in self.cytoplasm:
                    contour = self.cytoplasm[key]['contour']
                    index = self.cytoplasm[key]['index']
                    color = cell_colors[index]
                    cv2.fillPoly(overlay, pts =[contour], color=color)
                self.canvas = cv2.addWeighted(self.org3, 1, overlay, 0.3, 0)

                # draw cell index
                overlay = np.zeros([self.height, self.width, 3], np.uint8)
                for key in self.cytoplasm:
                    cx, cy = self.cytoplasm[key]['centroid']
                    index = self.cytoplasm[key]['index']
                    if index == 0: continue
                    cv2.putText(self.canvas, str(index), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA) 

        # return operated image
        self.update_mouse(self.canvas)

    # update cursor position
    def update_mouse(self, src):

        img = src.copy()

        if self.channel in ['nucleus', 'cell']:    
            color = cell_colors[self.selected]
            cv2.circle(img, self.mouse, 7, color, -1)

        elif self.channel in ['bacteria']:
            color = (0,0,255) if self.remove_bac else (0,255,0)
            x, y = self.mouse
            cv2.rectangle(img, (x-5, y-5), (x+5, y+5), color, -1)

        self.update_tk(img)

    def add_contour(self):

        if len(self.new_contour) > 2:
            contour = np.array(self.new_contour)
            area = cv2.contourArea(contour)
            if area < 400: return
            if area > 80000: return
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if self.mode == 'draw_nucleus':
                image.count2 += 1
                image.nucleus[image.count2] = {
                    'index': 1,
                    'divide': 0,
                    'border': 0,
                    'contour': contour,
                    'centroid': [cx, cy],
                }
            elif self.mode == 'draw_cell':
                image.count3 += 1
                image.cytoplasm[image.count3] = {
                    'index': image.count3,
                    'divide': 0,
                    'border': 0,
                    'contour': contour,
                    'centroid': [cx, cy],
                }

    def display_draw(self):

        if self.channel == 'nucleus':
            canvas = self.canvas2.copy()
            color = (0, 0, 255)
        elif self.channel == 'cell':
            canvas = self.canvas3.copy()
            color = (255, 0, 255)

        for i, [[x, y]] in enumerate(self.new_contour):
            if i > 0:
                x0, y0 = self.new_contour[i-1][0]
                cv2.line(canvas, (x, y), (x0, y0), color, 1)
            if i == len(self.new_contour)-1:
                xm, ym = self.mouse
                cv2.line(canvas, (x, y), (xm, ym), color, 1)
            cv2.circle(canvas, (x, y), 1, color, -1)

        self.update_tk(canvas)

    def display_label(self):

        # draw cell region
        overlay = np.zeros([self.height, self.width, 3], np.uint8)
        for key in self.cytoplasm:
            contour = self.cytoplasm[key]['contour']
            index = self.cytoplasm[key]['index']
            if index == 0: continue
            if self.mode == 'label_divide':
                label = 'divide' if self.cytoplasm[key]['divide'] else 'not divide'
            if self.mode == 'label_border':
                label = 'border' if self.cytoplasm[key]['border'] else 'not border'
            color = label_colors[label]
            cv2.fillPoly(overlay, pts =[contour], color=color)
        self.canvas = cv2.addWeighted(self.org2, 1, overlay, 0.3, 0)

        # update
        self.update_tk(self.canvas)

    def update_tk(self, imagecv):

        # to tkinter image
        imagecv = cv2.resize(imagecv, (0,0), fx=self.scale, fy=self.scale)
        imagecv = cv2.cvtColor(imagecv, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imagecv)
        imagetk = ImageTk.PhotoImage(image=im)
        panel.configure(image=imagetk)
        panel.image = imagetk


def onMouseMotion(event):

    # mouse motion
    x = int(event.x / image.scale)
    y = int(event.y / image.scale)
    image.mouse = (x, y)

    if image.mode in ['mark']:

        image.update_mouse(image.canvas)

    elif image.mode in ['draw_nucleus', 'draw_cell']:

        # mouse motion + left mouse down
        if event.state in [256, 264, 272]:

            # append the drawing line
            if len(image.new_contour):
                current_point = np.array([x, y]) * image.scale
                previous_point = np.array(image.new_contour[-1][0]) * image.scale
                if np.linalg.norm(current_point-previous_point) > 10:
                    image.new_contour.append([[x, y]])
            else:
                image.new_contour.append([[x, y]])

        image.display_draw()

def onMouseButton(event):

    # mouse position
    x = int(event.x / image.scale)
    y = int(event.y / image.scale)

    if image.mode in ['mark']:

        if image.channel in ['bacteria']:

            # Left click to add bacteria
            if event.num == 1:
                image.bacteria.append([x, y, 'isolated'])

            # Right click to remove bacteria
            elif event.num == 3:
                for i, (bx, by, group) in enumerate(image.bacteria):
                    dist =int(distance.euclidean((bx, by), (x, y)))
                    if dist < 8:
                        del image.bacteria[i]
                        break

        elif image.channel in ['nucleus', 'cell']:

            # Left mouse button press
            if event.num == 1:

                # select and change cytoplasm
                # currently cannot get this cytoplasm back, I hope...
                for key in image.cytoplasm:
                    if image.cytoplasm[key]['index'] == 0: continue
                    contour = image.cytoplasm[key]['contour']
                    is_clicked = cv2.pointPolygonTest(contour,(x,y),True)
                    if is_clicked > 0:
                        image.cytoplasm[key]['index'] = image.selected

                # remove nucleus if selected = background
                # currently cannot get this nucleus back, I hope...
                if not image.selected:
                    for key in image.nucleus:
                        if image.nucleus[key]['index'] == 0: continue
                        contour = image.nucleus[key]['contour']
                        is_clicked = cv2.pointPolygonTest(contour,(x,y),True)
                        if is_clicked > 0:
                            image.nucleus[key]['index'] = 0

            # Right mouse button press
            elif event.num == 3:

                # remove cell
                for key in image.cytoplasm:
                    if image.cytoplasm[key]['index'] == 0: continue
                    contour = image.cytoplasm[key]['contour']
                    is_clicked = cv2.pointPolygonTest(contour,(x,y),True)
                    if is_clicked > 0:
                        image.cytoplasm[key]['index'] = 0

                # remove nucleus
                for key in image.nucleus:
                    if image.nucleus[key]['index'] == 0: continue
                    contour = image.nucleus[key]['contour']
                    is_clicked = cv2.pointPolygonTest(contour,(x,y),True)
                    if is_clicked > 0:
                        image.nucleus[key]['index'] = 0

        image.display_mark()

    elif image.mode in ['draw_nucleus', 'draw_cell']:

        # Right mouse button press
        if event.num == 3:
            if len(image.new_contour): 
                image.new_contour.pop()
                image.display_draw()

    elif image.mode in ['label_divide', 'label_border']:

        # Left mouse button press
        if event.num == 1:

            target_label = image.mode.replace('label_', '')
            target_index = None
            target_value = 0

            # select cytoplasm
            # currently cannot get this cytoplasm back
            for key in image.cytoplasm:
                if image.cytoplasm[key]['index'] == 0: continue
                contour = image.cytoplasm[key]['contour']
                is_clicked = cv2.pointPolygonTest(contour,(x,y),True)
                if is_clicked > 0:
                    target_index = image.cytoplasm[key]['index']
                    target_value = image.cytoplasm[key][target_label]
                    target_value = 1 - target_value

                    # change cytoplasm label
                    for key in image.cytoplasm:
                        if image.cytoplasm[key]['index'] == target_index: 
                            image.cytoplasm[key][target_label] = target_value
                    break

        image.display_label()

def onKeyPress(event):

    # mode == mark
    if image.mode in ['mark']:

        # Exit
        if event.keysym == 'Escape':
            root.destroy()
            exit()

        # Change mouse's marking mode
        elif event.char == ' ':
            if image.channel in ['nucleus', 'cell']:
                image.pick_color()
            if image.channel in ['bacteria']:
                image.remove_bac = not image.remove_bac
            image.display_mark()

        # Change channel
        elif event.keysym == 'F1':
            image.channel = 'bacteria'
            image.display_mark()
        elif event.keysym == 'F2':
            image.channel = 'nucleus'
            image.display_mark()
        elif event.keysym == 'F3':
            image.channel = 'cell'
            image.display_mark()

        # Change view
        elif event.char in ['w', 'W', 'ไ', '"'] or event.keysym == 'Up':
            i = (image.views.index(image.view) - 1) % len(image.views)
            image.view = image.views[i]
            image.display_mark()
        elif event.char in ['s', 'S', 'ห', 'ฆ'] or event.keysym == 'Down':
            i = (image.views.index(image.view) + 1) % len(image.views)
            image.view = image.views[i]
            image.display_mark()

        # Change image
        elif event.char in ['a', 'A', 'ฟ', 'ฤ'] or event.keysym == 'Left':
            image.save_all()
            image.index = sorted([0, image.index-1, image.total-1])[1]
            image.load_image()
        elif event.char in ['d', 'D', 'ก', 'ฏ'] or event.keysym == 'Right':
            image.save_all()
            image.index = sorted([0, image.index+1, image.total-1])[1]
            image.load_image()

        # Save
        elif event.char in ['z', 'Z', 'ผ', '(']:
            image.save_all()

        # Clear all data to default
        elif event.char in ['c', 'C', 'แ', 'ฉ']:
            image.make_bacteria2()
            image.make_nucleus()
            image.make_cytoplasm()
            image.display_mark()

        # Change to draw mode
        elif event.char in ['2', '@', '/', '๑', 'v', 'V', 'อ', 'ฮ']: # think without numpad
            if image.channel == 'nucleus':
                image.mode = 'draw_nucleus'
                image.new_contour = []
            elif image.channel == 'cell':
                image.mode = 'draw_cell'
                image.new_contour = []
            image.display_draw()

        # Change to label mode (divide)
        elif event.char in ['3', '#', '-', '๒']:
            image.mode = 'label_divide'
            image.display_label()

        # Change to label mode (border)
        elif event.char in ['4', '$', 'ภ', '๓']:
            image.mode = 'label_border'
            image.display_label()

    # mode == draw nucleus, cell
    elif image.mode in ['draw_nucleus', 'draw_cell']:

        # Exit
        if event.keysym == 'Escape':
            root.destroy()
            exit()

        # Return to mark mode
        elif event.char in ['q', 'Q', 'ๆ', '๐']:
            image.mode = 'mark'
            image.display_mark()

        # Save and Return to mark mode
        elif event.char in ['1', '!', 'ๅ', '+']:
            image.add_contour()
            image.mode = 'mark'
            image.display_mark()

        # Save and Return to mark mode
        elif event.char in ['2', '@', '/', '๑', 'v', 'V', 'อ', 'ฮ']: # think without numpad
            image.add_contour()
            image.mode = 'mark'
            image.display_mark()

        # Change to label mode (divide)
        elif event.char in ['3', '#', '-', '๒']:
            image.mode = 'label_divide'
            image.display_label()

        # Change to label mode (border)
        elif event.char in ['4', '$', 'ภ', '๓']:
            image.mode = 'label_border'
            image.display_label()

    # mode == label divide
    elif image.mode in ['label_divide', 'label_border']:

        # Exit
        if event.keysym == 'Escape':
            root.destroy()
            exit()

        # Change image
        elif event.char in ['a', 'A', 'ฟ', 'ฤ'] or event.keysym == 'Left':
            image.save_all()
            image.index = sorted([0, image.index-1, image.total-1])[1]
            image.load_image()
        elif event.char in ['d', 'D', 'ก', 'ฏ'] or event.keysym == 'Right':
            image.save_all()
            image.index = sorted([0, image.index+1, image.total-1])[1]
            image.load_image()

        # Label boundary
        elif event.char in ['b', 'B', 'ิ', 'ฺ']:
            image.label_boundary()
            image.display_label()

        # Return to mark mode
        elif event.char in ['q', 'Q', 'ๆ', '๐']:
            image.mode = 'mark'
            image.display_mark()

        # Change to mark mode
        elif event.char in ['1', '!', 'ๅ', '+']:
            image.mode = 'mark'
            image.display_mark()

        # Change to draw mode
        elif event.char in ['2', '@', '/', '๑', 'v', 'V', 'อ', 'ฮ']: # think without numpad
            image.mode = 'draw_cell'
            image.new_contour = []
            image.display_draw()

        # Change to label mode (divide)
        elif event.char in ['3', '#', '-', '๒']:
            image.mode = 'label_divide'
            image.display_label()

        # Change to label mode (border)
        elif event.char in ['4', '$', 'ภ', '๓']:
            image.mode = 'label_border'
            image.display_label()


if __name__ == "__main__":

    # user inteface
    root = Tk()
    panel = tkinter.Label(root)
    panel.pack()

    navigation = 'S1/Plate_04/Training_set'
    navigation = 'S1/Plate_03/Testing_set'
    image = cellPlate(navigation)

    root.bind('<Motion>', onMouseMotion)
    root.bind('<Button>', onMouseButton)
    root.bind('<KeyPress>', onKeyPress)
    root.mainloop() 