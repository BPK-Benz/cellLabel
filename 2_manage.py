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
                "supercategory": 'cell_fused',
                "id": 1,
                "name": 'divide_cell',
            },
            {
                "supercategory": 'cell_fused',
                "id": 2,
                "name": 'not_divided_cell',
            },
        ]
    else:

        channel = annotation['channel']
        divide = annotation['divide']
        border = annotation['border']
        infect = annotation['infect']

        if channel == 'cell':
            if divide:
                return 1
            else:
                return 2
        else:
            return 0


if __name__ == "__main__":

    data = {
        'groundtruth': {
            'src':[
                'data/output_coco/S1/Plate_03/Testing_set',
            ],
            'dst': 'groundtruth.json',
        },
        # 'image_processing': {
        #     'src':[
        #         'data/output_predicted1/S1/Plate_03/Testing_set',
        #     ],
        #     'dst': 'image_processing.json',
        # },
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
                    if not label: continue
                    coco_object = {
                        'channel': annotations[a]['channel'],
                        'divide': annotations[a]['divide'],
                        'border': annotations[a]['border'],
                        'infect': annotations[a]['infect'],
                        'category_id': label,
                        'image_id': new_image_ids[annotations[a]['image_id']],
                        'id': count_annotation,
                        'iscrowd': 0,
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



