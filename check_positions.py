import cv2
import pandas as pd


def get_centers(profile):
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

ref2 = 'data/Error/Output_CP/Reference_Nucleus.csv'
profile2 = pd.read_csv(ref2)
ref3 = 'data/Error/Output_CP/Reference_Cell.csv'
channel2 = 'data/Error/Output_CP/001004-1-001001002_hoechst.png'
channel3 = 'data/Error/Output_CP/001004-1-001001003_Evan.png'

profile = profile2.loc[profile2['ImageNumber'] == 1]
centers = get_centers(profile)

image = cv2.imread(channel2)
cv2.imshow('image', image)
key = cv2.waitKey(0)
if key in [ord('q')]: exit()
