import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import random
import shutil
start = 0
counter_object_id = 0
counter_image_id = 0
# saved_ad = '/home/ehsan_taherkhani/SEGMENTED_LABELED_DATA/filter_RGBRight'
# saved_ad_anotated = '/home/ehsan/PycharmProjects/SwinT_detectron2-main/SEGMENTED_LABELED_DATA/anotated_RGBRight'
address = '/home/ehsan_taherkhani/train/trainSwinC.json'
file = open(address, 'r')
data = json.load(file)

coco_data = {
    "info": {},
    "licenses": [],
    "categories": [],
    "images": [],
    "annotations": []

}
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
coco_data["categories"].append({"id": 1, "name": "package"})

# This is done based on the images from ground truth lables

folder_path = '/home/ehsan_taherkhani/train/images'
new_address_images = '/home/ehsan_taherkhani/converted_single_ob'
curent_image = -10
for item in data['annotations']:
    Flag = False
    for img in data['images']:
        if item['image_id'] == img['id']:
            if curent_image != img['id']:
                curent_image = img['id']
                create_name = str(counter_image_id) + '.png'
                create_name_address = os.path.join(new_address_images, create_name)
                # if not os.path.exists(create_name_address):
                img_address = os.path.join(folder_path, img['file_name'])
                image_cp = cv2.imread(img_address)

                # Save the image in PNG format
                cv2.imwrite(create_name_address, image_cp)

                #shutil.copy(img_address, create_name_address)
                print(f'image {counter_image_id} is copied')

                coco_data["images"].append({'file_name': create_name, 'id': int(counter_image_id), 'width': int(640),
                           'height': int(480)})

                coco_data["annotations"].append({
                    "id": int(counter_object_id),
                    "image_id": int(counter_image_id),
                    "category_id": int(1),
                    "segmentation": item['segmentation'],
                    "area": int(item['area']),
                    "bbox": item['bbox'],
                    "iscrowd": int(0)
                })
                counter_image_id += 1
            else:
                coco_data["annotations"].append({
                    "id": int(counter_object_id),
                    "image_id": int(counter_image_id),
                    "category_id": int(1),
                    "segmentation": item['segmentation'],
                    "area": int(item['area']),
                    "bbox": item['bbox'],
                    "iscrowd": int(0)
                })
            Flag = True
            break

    if Flag == False:
        print('error may happen!')
    counter_object_id += 1


with open('ground_ruth_train.json', 'w') as destination_file:
    json.dump(coco_data, destination_file)
print('images from ground truth lables is finished !')

# start += len(data['images'])
# # This is done based on the images from RGBright
#
# address = '/home/ehsan/PycharmProjects/SwinT_detectron2-main/SEGMENTED_LABELED_DATA/RGBright/sam_annotated_data_rGBRight.json'
#
# file = open(address, 'r')
#
# data = json.load(file)
#
# folder_path = '/home/ehsan/PycharmProjects/SwinT_detectron2-main/SEGMENTED_LABELED_DATA/RGBright/unlabled_RGBright/'
#
# for item in data['annotations']:
#     Flag = False
#     for img in data['images']:
#         if item['image_id'] == img['id']:
#             counter_img_id = start + img['id']
#             coco_data["images"].append({"id": counter_img_id, "file_name": img['file_name']})
#             Flag = True
#             break
#     if Flag == False:
#         print('error may happen!')
#     counter_object_id += 1
#     coco_data["annotations"].append({
#         "id": counter_object_id,
#         "image_id": counter_img_id,
#         "category_id": 0,
#         "segmentation": item['segmentation'],
#         "area": int(item['area']),
#         "bbox": item['bbox'],
#         "iscrowd": 0
#                     })
#     print(f'image {counter_img_id} is done')
# print('images RGBright is finished !')
# # This is done based on the images from RGBRight
# start += len(data['images'])
#
# address = '/home/ehsan/PycharmProjects/SwinT_detectron2-main/SEGMENTED_LABELED_DATA/RGBRight/sam_annotated_data_RGBRight.json'
# file = open(address, 'r')
#
# data = json.load(file)
#
# folder_path = '/home/ehsan/PycharmProjects/SwinT_detectron2-main/SEGMENTED_LABELED_DATA/RGBright/unlabled_RGBright/'
#
# for item in data['annotations']:
#         Flag = False
#         for img in data['images']:
#             if item['image_id'] == img['id']:
#                 counter_img_id = start + img['id']
#                 coco_data["images"].append({"id": counter_img_id, "file_name": img['file_name']})
#                 Flag = True
#                 break
#         if Flag == False:
#             print('error may happen!')
#         counter_object_id += 1
#         coco_data["annotations"].append({
#             "id": counter_object_id,
#             "image_id": counter_img_id,
#             "category_id": 0,
#             "segmentation": item['segmentation'],
#             "area": int(item['area']),
#             "bbox": item['bbox'],
#             "iscrowd": 0
#         })
#         print(f'image {counter_img_id} is done')
#
# print('images RGBRight is finished !')
# with open('combine_dataset.json', 'w') as destination_file:
#     json.dump(coco_data, destination_file)