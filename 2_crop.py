import os
import cv2
import json
import utils.make_coco as make_coco


def load_coco(coco_path):
    with open(coco_path) as file:
        coco = json.load(file)
    return coco

def condition(annotation=None):
    if not annotation:
        return [
            {
                "supercategory": 'cell_fused',
                "id": 1,
                "name": 'border',
            },
            {
                "supercategory": 'cell_fused',
                "id": 2,
                "name": 'divide',
            },
            {
                "supercategory": 'cell_fused',
                "id": 3,
                "name": 'normal',
            },
        ]
    else:

        channel = annotation['channel']
        divide = annotation['divide']
        border = annotation['border']
        infect = annotation['infect']

        if channel == 'cell':
            if border:
                return 1
            elif divide:
                return 2
            else:
                return 3
        else:
            return 0


if __name__ == "__main__":

    data = {
        'train': {
            'src':[
                'data/output_coco/S1/Plate_03/Testing_set',
            ],
            'dst': 'groundtruth.json',
        },
        'test': {
            'src':[
                'data/output_predicted1/S1/Plate_03/Testing_set',
            ],
            'dst': 'image_processing.json',
        },
    }    
    
    dst_path = 'data/output_crop'

    names = {}
    categories = condition()
    for category in categories:
        names[category['id']] = category['name']
        folder = os.path.join(dst_path, category['name'])
        if not os.path.exists(folder):
            os.makedirs(folder)

    count_folder = 1
    count_image = 1
    count_annotation = 1

    for key in data:

        folders = data[key]['src']
        for folder in folders:

            print('[ {} : {} of {} | {} ]'.format(key, count_folder, len(folders), folder))
            count_folder += 1

            files = sorted(os.listdir(folder))
            for file in files:

                coco_path = os.path.join(folder, file)
                coco = load_coco(coco_path)
                images = coco['images']
                annotations = coco['annotations']

                image_path = images[0]['file_name']
                image = cv2.imread(image_path)

                for ann in annotations:

                    if count_annotation > 1000: exit()

                    label = condition(ann)
                    if label in names: 
                        category_name = names[label] 
                    else:
                        continue
                    x, y, w, h = ann['bbox']
                    cropped = image[y:y+h, x:x+w]
                    file_name = str(count_annotation).zfill(8) + '.png'
                    file_path = os.path.join(dst_path, category_name, file_name)
                    cv2.imwrite(file_path, cropped)
                    count_annotation += 1
                count_image += 1