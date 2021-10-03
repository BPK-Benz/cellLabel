import os
import json
import utils.make_coco as make_coco


def load_coco(coco_path):
    with open(coco_path) as file:
        coco = json.load(file)
    return coco

def save_coco(coco_path, coco):
    with open(coco_path, 'w') as outfile:
        json.dump(coco, outfile)

def create_coco():
    return {
        "info": make_coco.add_info(),
        "licenses": make_coco.add_licenses(),
        "categories": [],
        "images": [],
        "annotations": [],
    }


def condition(annotation=None):
    if not annotation:
        return [
            {
                "supercategory": 'cell',
                "id": 1,
                "name": 'non-border',
            },
            {
                "supercategory": 'cell',
                "id": 2,
                "name": 'border',
            },
        ]
    else:

        channel = annotation['channel']
        divide = annotation['divide']
        border = annotation['border']
        infect = annotation['infect']

        if channel == 'nucleus':
            return 0

        if not border:
            return 1
        else:
            return 2

if __name__ == "__main__":

    conditions = {
        'cell': False,
        'nucleus': True, # x 2
        'divide': False, # x 2
        'border': False, # x 2
        'infect': False  # x 3
    }

    data = {
        'train': {
            'src':[
                'data/output_coco/S1/Plate_07/Testing_set',
                'data/output_coco/S1/Plate_08/Testing_set',
            ],
            'dst': 'train.json',
        },
        'test': {
            'src':[
                'data/output_coco/S1/Plate_03/Testing_set',
            ],
            'dst': 'test.json',
        }
    }

    for key in data:

        all_coco = create_coco()
        all_coco['categories'] = condition()

        count_folder = 1
        count_image = 1
        count_annotation = 1

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
                new_image_ids = {}

                for i in range(len(images)):
                    new_image_ids[images[i]['id']] = count_image                    
                    images[i]['id'] = count_image
                    count_image += 1

                new_annotations = []
                for a in range(len(annotations)):
                    label = condition(annotations[a])
                    if label == 0: continue
                    coco_object = {
                        'category_id': label,
                        'image_id': new_image_ids[annotations[a]['image_id']],
                        'id': count_annotation,
                        'area': annotations[a]['area'],
                        'bbox': annotations[a]['bbox'],
                        'segmentation': annotations[a]['segmentation'],
                    }
                    new_annotations.append(coco_object)
                    count_annotation += 1

                all_coco['images'] += images
                all_coco['annotations'] += new_annotations

        all_coco_path = data[key]['dst']
        save_coco(all_coco_path, all_coco)



