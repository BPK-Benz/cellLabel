import os
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import distance


# How to read raw image from columbus: row, column and field
def filter_profile(profiles, name):
    metadata_row = int(name[0:3])
    metadata_col = int(name[3:6])
    metadata_field = int(name.split('-')[1])
    profile = profiles.loc[
            (profiles['Metadata_Row'] == metadata_row) &
            (profiles['Metadata_Column'] == metadata_col) &
            (profiles['Metadata_Field'] == metadata_field)
    ]
    return profile


def get_centers(profile):  # to guide center of each cell or nucleus
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

# To find contours from cv2
def findContours(edge_object):
    edge = cv2.cvtColor(edge_object, cv2.COLOR_BGR2GRAY)
    edge = cv2.rectangle(edge, (0, 0), (width-1, height-1), 255, 1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ret, thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # print(len(contours))
    return contours, hierarchy


for head in ['003007-']:  # ,'005007-','005015-','012012-','012013-'
    for _ in range(1, 11):

        # Define image and cellprofiler output path
        base_image = 'data/Enhance_image/S1/Plate_03/Testing_set/'
        name_imageBac = head+str(_)+'-001001001.tif'
        name_imageNuc = head+str(_)+'-001001002.tif'

        base_cellprofiler = 'data/Output_CP/S1/Plate_03/Testing_set/'
        name_cpBac = head+str(_)+'-00100100001_FITC.png'
        name_cpNucleus = head+str(_)+'-001001002_hoechst.png'
        name_cpCell = head+str(_)+'-001001003_Evan.png'

        original1_path = os.path.join(base_image, name_imageBac)
        original2_path = os.path.join(base_image, name_imageNuc)
        cellprofiler1_path = os.path.join(base_cellprofiler, name_cpBac)
        cellprofiler2_path = os.path.join(base_cellprofiler, name_cpNucleus)
        cellprofiler3_path = os.path.join(base_cellprofiler, name_cpCell)

        org1 = cv2.imread(original1_path)
        org2 = cv2.imread(original2_path)
        cpf2 = cv2.imread(cellprofiler2_path)
        cpf3 = cv2.imread(cellprofiler3_path)

        bac_profiles = pd.read_csv(os.path.join(
            base_cellprofiler, 'Reference_Bac.csv'))
        profile1 = filter_profile(bac_profiles, name=name_cpBac)

        nucleus_profiles = pd.read_csv(os.path.join(
            base_cellprofiler, 'Reference_Nucleus.csv'))
        profile2 = filter_profile(nucleus_profiles, name=name_cpNucleus)

        cell_profiles = pd.read_csv(os.path.join(
            base_cellprofiler, 'Reference_Cell.csv'))
        profile3 = filter_profile(cell_profiles, name=name_cpCell)

        # get resolution and create blank template
        height, width = org2.shape[:2]
        blank1 = np.zeros([height, width], np.uint8)  # prepare for intensity mask
        blank3 = np.zeros([height, width, 3], np.uint8)

        
        # Starting with image processing 
        # 1. Nucleus because all overlays are on nucleus image
        # 1.1 Define divided nucleus and border nucleus
        # create nucleus border by finding nucleus border from output of CP
        edge2 = blank3.copy()
        edge2[np.where((cpf2==[0,255,255]).all(axis=2))] = [255,255,255]
        # load center from cellprofiler
        centers2 = get_centers(profile2)
        # find contours
        contours2, hierarchy2 = findContours(edge_object=edge2)


        for con2, hie2 in zip(contours2, hierarchy2[0]): # keep only outer border
            # 1.1.1 Area
            area2 = cv2.contourArea(con2)
            # 1.1.2.calculate centroid
            M2 = cv2.moments(con2)
            cx2 = int(M2["m10"] / M2["m00"])
            cy2 = int(M2["m01"] / M2["m00"])

            # 1.1.3. Intensity
            mask2 = cv2.fillPoly(blank1.copy(), pts =[con2], color=255)
            checkIntensity = org2[mask2==255]
            maxIntensity = np.max(checkIntensity)
            IntRatio = len(checkIntensity[checkIntensity >= 200])/len(checkIntensity[checkIntensity < 200])
            numberPixel = 30
            aevrageNumberPixel = np.average(np.array(checkIntensity)[np.array(checkIntensity).argsort()[-numberPixel:]])

            # 1.1.4. Shape
            rect2 = cv2.minAreaRect(con2)
            (x2, y2), (w2, h2), angle2 = rect2
            box2 = cv2.boxPoints(rect2)
            box2 = np.int0(box2)
            ratio_length2 = np.max([w2,h2])/np.min([w2,h2])
            

            # Reduce randundancy of contours
            for n, (x2, y2) in enumerate(centers2):
                if abs(cx2-x2) + abs(cy2-y2) < 1   : # change 25 to 5
                    
                    # border nucleus
                    if check_contour_boundary(con2, width, height, touch_threshold=10): 
                        cv2.polylines(cpf2, [box2], True, (255, 180, 0), 5)

                    if (area2 < 4000 and area2 > 1200) and (ratio_length2 > 1.2 and np.min([w2,h2]) < 45 and maxIntensity == 255) and (aevrageNumberPixel > 30 and IntRatio > 0.3) :
                        cv2.circle(cpf2, (cx2, cy2), 10, (255, 0, 255), -1)
                        # cv2.putText(cpf2, str(round(area,3)), (cX2-30, cy2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # cv2.putText(cpf2, str(round(np.min([w,h]),3)), (cX2, cy2+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
                        # cv2.putText(cpf2, str(maxIntensity), (cX2, cy2+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # 2. Cell
        # 2.1 Define only border cells
        # create border from output of cellprofiler
        edge3 = blank3.copy()
        edge3[np.where((cpf3==[0,0,255]).all(axis=2))] = [255,255,255]

        # load center from cellprofiler
        centers3 = get_centers(profile3)

        # find contours
        contours3, hierarchy3 = findContours(edge_object=edge3)

        for con3, hie3 in zip(contours3, hierarchy3[0]):
            # filter out the noise
            area3 = cv2.contourArea(con3)
            if area3 < 1000: continue
            if area3 > 20000: continue
            M3 = cv2.moments(con3)
            cx3 = int(M3["m10"] / M3["m00"])
            cy3 = int(M3["m01"] / M3["m00"])
            rect3 = cv2.minAreaRect(con3)
            (x3, y3), (w3, h3), angle3 = rect3
            box3 = cv2.boxPoints(rect3)
            box3 = np.int0(box3)
            ratio_length3 = np.max([w3,h3])/np.min([w3,h3])

            for n, (x3, y3) in enumerate(centers3):
                if abs(cx3-x3) + abs(cy3-y3) < 25:
                    if check_contour_boundary(con3, width, height, touch_threshold=10):
                        cv2.polylines(cpf2, [box3], True, (50, 255, 100), 5) # Overlay on nucleus image
                        cv2.polylines(cpf3, [box3], True, (50, 255, 100), 5) # Overlay on cell image
        
        # 3. Bacteria is the last step because its overlay is on nucleus image
        # Directly image processing
        BacCenters = get_centers(profile1)
        padding = 7
        for Bx, By in BacCenters:
            # count += 1
            con = np.array([
                            [[Bx-padding, By-padding]],
                            [[Bx-padding, By+padding]],
                            [[Bx+padding, By+padding]],
                            [[Bx+padding, By-padding]], ])


        # Start image processing
        img = org1.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
        # Adaptive thresholding help us fix intensity variation of image
        threshold1 = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
        contours1, hierarchy1 = cv2.findContours(threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        bacteria = []
        c = 0  # count number of bacteria that passed inclusion criteria
        cm = []  # keep location of bacteria that passed inclusion criteria
        out = 0  # count number of bacteria that be a member of exclusion criteria

        isolated_count = 0  # Green: number is 1
        cluster_count = 0  # Red: number is 2
        semi_count = 0  # Orange: number is 1
        minArea = 20
        maxArea = 300
        clusterArea = 60
        clusterRatio = 0.95
        redunDist = 10

        for cnt1 in contours1:
            area1 = cv2.contourArea(cnt1)  # Measure area

            if area1 < minArea or area1 > maxArea: continue

            M1 = cv2.moments(cnt1)  # Measure centroid
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])

            convex_hull1 = cv2.convexHull(cnt1)  # Measure convex hull
            convex_hull_area1 = cv2.contourArea(convex_hull1)  # Measure area of convex hull
            ratio1 = area1 / convex_hull_area1  # find the criteria of cluster or isolated

            if (cX1, cY1) not in cm: # Reduce redundancy
                if c != 0:
                    dist = int(distance.euclidean((cX1, cY1), cm[c-1]))

                    if dist > redunDist:
                        if ratio1 < clusterRatio:
                            if area1 > clusterArea:
                                cv2.circle(img, (cX1, cY1), 8, (0, 0, 255), 2)
                                cv2.circle(cpf2, (cX1, cY1), 8, (0, 0, 255), 2) # overlay bacteria on nucleus image 
                                bacteria.append([cX1, cY1, 'cluster'])
                            else:
                                cv2.circle(img, (cX1, cY1), 8, (0, 100, 255), 2)
                                cv2.circle(cpf2, (cX1, cY1), 8, (0, 100, 255), 2)
                                bacteria.append([cX1, cY1, 'semi'])
                        else:
                            cv2.circle(img, (cX1, cY1), 8, (55, 150, 255), 2)
                            cv2.circle(cpf2, (cX1, cY1), 8, (55, 150, 255), 2)
                            bacteria.append([cX1, cY1, 'isolated'])
                else:
                    if ratio1 < clusterRatio and area1 > clusterArea:
                        cv2.circle(img, (cX1, cY1), 8, (0, 0, 255), 2)
                        cv2.circle(cpf2, (cX1, cY1), 8, (0, 0, 255), 2)
                        bacteria.append([cX1, cY1, 'cluster'])
                    else:
                        cv2.circle(img, (cX1, cY1), 8, (55, 150, 255), 2)
                        cv2.circle(cpf2, (cX1, cY1), 8, (55, 150, 255), 2)
                        bacteria.append([cX1, cY1, 'isolated'])
                
                    dist = 0

                cm = cm + [(cX1,cY1)]
                c += 1


        # To show the output image of three channels and overall results
        band = np.zeros([height,10,3],dtype=np.uint8)
        band[:] = 255
        vis1 = cv2.hconcat([org2, band, cpf2])
        vis2 = cv2.hconcat([img, band, cpf3])
       
        band = np.zeros([10,width*2+10,3],dtype=np.uint8)
        band[:] = 255

        vis = cv2.vconcat([vis1, band, vis2])

        cv2.imshow(name_imageBac[:-7], cv2.resize(vis, None, fx=0.4, fy=0.4))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
