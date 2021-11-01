import os
import cv2
import time
import json
import numpy as np
import pandas as pd
import tkinter
from tkinter import *
from PIL import Image, ImageTk
from scipy.spatial import distance

import utils.make_coco as make_coco


def create_coco():
    return {
        "info": make_coco.add_info(),
        "licenses": make_coco.add_licenses(),
        "categories": make_coco.add_categories(),
        "images": [],
        "annotations": [],
    }


# check criteria for boundary cell
def check_contour_boundary(contour, w, h, touch_threshold):

    touch = 0
    # touch_threshold is distance touching boarder
    margin = 10  # height from boarder

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


class cellPlate:
    def __init__(self, navigation):

        # get absolute
        self.root_path = os.path.dirname(os.path.realpath(__file__))
        self.navigation = navigation

        # input paths
        self.original_path = os.path.join("r", self.root_path, "data/Enhance_image", self.navigation)
        self.cellprofiler_path = os.path.join("r", self.root_path, "data/Output_CP", self.navigation)

        # output paths, image processing
        self.image_path = os.path.join("r", self.root_path, "data/output_image", self.navigation)
        self.predicted1_path = os.path.join("r", self.root_path, "data/output_predicted1", self.navigation)

        # create save path if not exists
        if not os.path.exists(self.image_path): os.makedirs(self.image_path)
        if not os.path.exists(self.predicted1_path): os.makedirs(self.predicted1_path)

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

        # mode = mark: select channel
        self.channels = ['bacteria', 'nucleus', 'cell']
        self.channel = 'nucleus'

        self.showing = {
            'predicted_bacteria': False,
            'predicted_nucleus': False,
            'predicted_cell': False,
            'predicted_border': False,
            'predicted_divide': True,
        }

        # window
        self.scale = 0.9
        self.mouse = 0, 0

        # automatic 
        self.is_auto = True   # If is_export is True, is_auto shoul be True as well to automatic changing images (every change images are saved)
        if self.is_auto: self.auto_generate()

        # load first image
        self.load_image()

    def auto_generate(self):

        for i in range(self.total):
            t0 = time.time()

            self.index = i
            self.load_image()
            self.export_image()
            # self.export_predicted()

            elapse = time.time() - t0
            print('[ saved {} of {} | {} in {:.2f} seconds ]'.format(self.index+1, self.total, self.names[self.index], elapse))

    def load_image(self):

        # change window title
        name = self.names[self.index]
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

        self.make_bacteria2()
        self.predict_nucleus()
        self.predict_cell()

        self.display()

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

    def predict_nucleus(self): # to find the location and contour of nucleus from result of cellprofiler

        # contain nucleus infomation
        self.nucleus = dict()

        # find contours
        edge = cv2.cvtColor(self.edge2, cv2.COLOR_BGR2GRAY)
        edge = cv2.rectangle(edge, (0, 0), (self.width-1, self.height-1), 255, 1) 
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # load center from cell profiler
        centers = self.get_centers(self.profile2)

        # identify each cell
        count = 0
        for contour, hie in zip(contours, hierarchy[0]):

            # 1.1.1 Area
            area = cv2.contourArea(contour)

            # 1.1.2.calculate centroid
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 1.1.3. Intensity
            mask = cv2.fillPoly(self.blank1.copy(), pts =[contour], color=255)
            checkIntensity = self.org2[mask==255]
            maxIntensity = np.max(checkIntensity)
            if len(checkIntensity[checkIntensity < 200]) == 0: continue
            IntRatio = len(checkIntensity[checkIntensity >= 200])/len(checkIntensity[checkIntensity < 200])
            numberPixel = 30
            averageNumberPixel = np.average(np.array(checkIntensity)[np.array(checkIntensity).argsort()[-numberPixel:]])

            # 1.1.4. Shape
            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            ratio_length = np.max([w, h])/np.min([w, h])

            # Reduce randundancy of contours
            for n, (x, y) in enumerate(centers):
                if abs(cx-x) + abs(cy-y) < 1   : # change 25 to 5

                    divide = int(
                        (area < 4000 and area > 1200) and 
                        (ratio_length > 1.2 and np.min([w, h]) < 45 and maxIntensity == 255) and 
                        (averageNumberPixel > 30 and IntRatio > 0.3)
                    )

                    # make label
                    count += 1
                    self.nucleus[count] = {
                        'index': count,
                        'divide': divide,
                        'border': 0,
                        'contour': contour,
                        'centroid': [cx, cy],
                    }

                    # prevent duplicate 
                    centers.remove([x, y])

        self.count2 = count

    def predict_cell(self): # to find the location and contour of nucleus from result of cellprofiler

        # contain cell infomation
        self.cell = dict()

        # find contours
        edge = cv2.cvtColor(self.edge3, cv2.COLOR_BGR2GRAY)
        edge = cv2.rectangle(edge, (0, 0), (self.width-1, self.height-1), 255, 1) 
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        # load center
        centers = self.get_centers(self.profile3)

        # identify each cell
        count = 0
        for contour, hie in zip(contours, hierarchy[0]):

            border = 0

            # filter out the noise
            area = cv2.contourArea(contour)
            if area < 1000: continue
            if area > 20000: continue

            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            ratio_length = np.max([w, h])/np.min([w, h])

            for n, (x, y) in enumerate(centers):
                if abs(cx-x) + abs(cy-y) < 25:

                    border = int(check_contour_boundary(contour, self.width, self.height, touch_threshold=10))

                    # make label
                    count += 1
                    self.cell[count] = {
                        'index': count,
                        'divide': 0,
                        'border': border,
                        'contour': contour,
                        'centroid': [cx, cy],
                    }

                    # prevent duplicate 
                    centers.remove([x, y])

        self.count3 = count

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

    def export_predicted(self):

        # coco
        count_image = 1
        count_annotation = 1
        coco = create_coco()

        # write each
        for key3 in self.cell:

            # match pair cell nucleus
            contour3 = self.cell[key3]['contour']
            is_member = False
            for key2 in self.nucleus:
                controid2 = self.nucleus[key2]['centroid']
                divide = self.nucleus[key2]['divide']
                is_member = cv2.pointPolygonTest(contour3,controid2,True) > 0
                if is_member: break
            if not is_member: continue

            # label by condition
            border = self.cell[key3]['border']
            divide = self.nucleus[key2]['divide']

            # properties 2
            contour2 = self.nucleus[key2]['contour']
            area2 = cv2.contourArea(contour2)
            rectangle2 = cv2.boundingRect(contour2)
            flat_contour2 = [contour2.flatten().tolist()]

            contour3 = self.cell[key3]['contour']
            area3 = cv2.contourArea(contour3)
            rectangle3 = cv2.boundingRect(contour3)
            flat_contour3 = [contour3.flatten().tolist()]

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

                in_nucleus = cv2.pointPolygonTest(contour2, (x, y), True)
                in_cell = cv2.pointPolygonTest(contour3, (x, y), True)

                if in_cell > 0:
                    allbac += nbac
                    if in_nucleus > 0:
                        nucbac += nbac
            cytobac = allbac - nucbac

            infect = None
            if allbac == 0:
                infect = 'non-infected'
            elif nucbac >= cytobac:
                infect = 'nuccell'
            else:
                infect = 'cytocell'

            # append nucleus data to coco json
            coco_object = {
                "channel": 'nucleus',
                "divide": divide,
                "border": border,
                "infect": infect,
                "iscrowd": 0,
                "area": area2,
                "image_id": count_image,
                "bbox": rectangle2,
                "segmentation": flat_contour2,
                "category_id": 2,
                "id": count_annotation,
            }
            coco['annotations'].append(coco_object)
            count_annotation += 1

            # append cell data to coco json
            coco_object = {
                "channel": 'cell',
                "divide": divide,
                "border": border,
                "infect": infect,
                "iscrowd": 0,
                "area": area3,
                "image_id": count_image,
                "bbox": rectangle3,
                "segmentation": flat_contour3,
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

        # save coco to json file
        coco_name = self.names[self.index] + '.json'
        coco_path = os.path.join(self.predicted1_path, coco_name)
        with open(coco_path, 'w') as outfile:
            json.dump(coco, outfile)

    def display(self):

        # select channel
        if self.channel == 'bacteria':
            canvas = self.org1.copy()
        elif self.channel == 'nucleus':
            canvas = self.org2.copy()
        elif self.channel == 'cell':
            canvas = self.org3.copy()

        if self.showing['predicted_cell']:
            for key in self.cell:
                contour = self.cell[key]['contour']
                cv2.drawContours(canvas, [contour], -1, (200, 100, 255), 3)

        if self.showing['predicted_nucleus']:
            for key in self.nucleus:
                contour = self.nucleus[key]['contour']
                cv2.drawContours(canvas, [contour], -1, (255, 200, 100), 3)

        if self.showing['predicted_bacteria']:
            for x, y, group in self.bacteria:
                color = {
                    'cluster': [0, 0, 255],
                    'semi': [0, 100, 200],
                    'isolated': [55, 200, 255],
                }[group]
                cv2.circle(canvas, (x, y), 8, color, 2)

        if self.showing['predicted_border']:
            for key in self.cell:
                if not self.cell[key]['border']: continue
                x, y = self.cell[key]['centroid']
                cv2.circle(canvas, (x, y), 8, (255, 0, 0), -1)

        if self.showing['predicted_divide']:
            for key in self.nucleus:
                if not self.nucleus[key]['divide']: continue
                x, y = self.nucleus[key]['centroid']
                cv2.circle(canvas, (x, y), 8, (0, 255, 0), -1)

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

    # Change channel
    elif event.keysym == 'F1':
        image.channel = 'bacteria'
        image.display()
    elif event.keysym == 'F2':
        image.channel = 'nucleus'
        image.display()
    elif event.keysym == 'F3':
        image.channel = 'cell'
        image.display()

    # Change image
    elif event.char in ['a', 'A', 'ฟ', 'ฤ'] or event.keysym == 'Left':
        image.export_predicted()
        image.index = sorted([0, image.index-1, image.total-1])[1]
        image.load_image()
    elif event.char in ['d', 'D', 'ก', 'ฏ'] or event.keysym == 'Right':
        image.export_predicted()
        image.index = sorted([0, image.index+1, image.total-1])[1]
        image.load_image()

    # Show predicted
    elif event.char in ['1', '!', 'ๅ', '+']:
        image.showing['predicted_bacteria'] = not image.showing['predicted_bacteria']
        image.display()
    elif event.char in ['2', '@', '/', '๑']:
        image.showing['predicted_cell'] = not image.showing['predicted_cell']
        image.display()
    elif event.char in ['3', '#', '-', '๒']:
        image.showing['predicted_nucleus'] = not image.showing['predicted_nucleus']
        image.display()
    elif event.char in ['4', '$', 'ภ', '๓']:
        image.showing['predicted_border'] = not image.showing['predicted_border']
        image.display()
    elif event.char in ['5', '%', 'ถ', '๔']:
        image.showing['predicted_divide'] = not image.showing['predicted_divide']
        image.display()

if __name__ == "__main__":

    # user inteface
    root = Tk()
    panel = tkinter.Label(root)
    panel.pack()

    navigation = 'S1/Plate_12/Training_set'
    navigation = 'S1/Plate_03/Testing_set'
    image = cellPlate(navigation)

    root.bind('<KeyPress>', onKeyPress)
    root.mainloop() 