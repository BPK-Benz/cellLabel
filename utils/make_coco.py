import os
import cv2

root = './'
images_path = os.path.join(root, "images")

# label: info
def add_info():
	return {
    	"description": "The cell segmentation tool for P'Benz.",
    	"url": "",
    	"version": "1.0",
    	"year": 2021,
    	"contributor": "AIMLab",
    	"date_created": "2021/03/29",
	}

# label: licenses
def add_licenses():
	return [
		{
			"url": "",
			"id": 1,
			"name": "AIMLab, Mahidol U.",
		}
	]

# label: categories = divide
def add_categories():
	return [
		{
			"supercategory": 'bacteria',
			"id": 1,
			"name": 'bacteria',
		},
		{
			"supercategory": 'nucleus',
			"id": 2,
			"name": 'nucleus',
		},
		{
			"supercategory": 'cell',
			"id": 3,
			"name": 'cell',
		}
	]