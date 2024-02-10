import os.path
import shutil
import cv2
import numpy as np
import json

J_1 = '/home/ehsan/PycharmProjects/data_augmentation/SYNTHETIC_DATA_3/json_file_first/myjson40.json'
I_1 = '/home/ehsan/PycharmProjects/data_augmentation/SYNTHETIC_DATA_3/synthetic_images_first'

J_2 = '/home/ehsan/PycharmProjects/data_augmentation/SYNTHETIC_DATA_3/json_file_second/myjson24.json'
I_2 = '/home/ehsan/PycharmProjects/data_augmentation/SYNTHETIC_DATA_3/synthetic_images_second'

J_3 = '/home/ehsan/PycharmProjects/data_augmentation/SYNTHETIC_DATA_3/json_file_sixth/myjson24.json'
I_3 = '/home/ehsan/PycharmProjects/data_augmentation/SYNTHETIC_DATA_3/synthetic_images_sixth'

J_4 = '/media/ehsan/Bereket/SYNTHETIC_DATA_4/set/json_file/myjson18.json'
I_4 = '/media/ehsan/Bereket/SYNTHETIC_DATA_4/set/synthetic_images'

J_5 = '/media/ehsan/Bereket/SYNTHETIC_DATA_2/json_file/myjson19.json'
I_5 = '/media/ehsan/Bereket/SYNTHETIC_DATA_2/synthetic_images'

J_6 = '/media/ehsan/Bereket/SYNTHETIC_DATA/json_file_first/myjson29.json'
I_6 = '/media/ehsan/Bereket/SYNTHETIC_DATA/synthetic_images_first'

J_7 = '/media/ehsan/Bereket/SYNTHETIC_DATA/json_file_second/myjson41.json'
I_7 = '/media/ehsan/Bereket/SYNTHETIC_DATA/synthetic_images_second'

J_8 = '/media/ehsan/Bereket/SYNTHETIC_DATA/json_file_third/myjson37.json'
I_8 = '/media/ehsan/Bereket/SYNTHETIC_DATA/synthetic_images_thid'

J_9 = '/media/ehsan/Bereket/SYNTHETIC_DATA/json_file_fourth/myjson52.json'
I_9 = '/media/ehsan/Bereket/SYNTHETIC_DATA/synthetic_images_fourth'

J_10 = '/media/ehsan/Bereket/SYNTHETIC_DATA/json_file_fifth/myjson18.json'
I_10 = '/media/ehsan/Bereket/SYNTHETIC_DATA/synthetic_images_fifth'

J_11 = '/media/ehsan/Bereket/SYNTHETIC_DATA_4/json_file/myjson48.json'
I_11 = '/media/ehsan/Bereket/SYNTHETIC_DATA_4/synthetic_images'

J_12 = '/home/ehsan/PycharmProjects/datasetSplit/train/trainSwinC.json'
I_12 = '/home/ehsan/PycharmProjects/datasetSplit/train/images'


list_json_file = [J_1, J_2, J_3, J_4, J_5, J_6, J_7, J_8, J_9, J_10, J_11, J_12]

list_image_file = [I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8, I_9, I_10, I_11, I_12]


json_address = '/media/ehsan/Bereket/Combined_Dataset'
image_training_address = '/media/ehsan/Bereket/Combined_Dataset/images'

coco_data = {
    "info": {},
    "licenses": [],
    "categories": [],
    "images": [],
    "annotations": []
}

coco_data["categories"].append({"id": 1, "name": "package"})
object_id_global = 0
image_id_global = -1
current_img_id = -10
for jason_file_index in range(len(list_json_file)):
    file = open(list_json_file[jason_file_index], 'r')
    jason_data = json.load(file)
    image_address_root = list_image_file[jason_file_index]
    for an in jason_data["annotations"]:
        image_name = str(an["image_id"])+'.png'
        image_address = os.path.join(image_address_root, image_name)
        if not os.path.exists(image_address):
            print('File was not found!')
        else:
            if current_img_id != an["image_id"]:
                current_img_id = an["image_id"]
                image_id_global += 1
                new_image_name = str(image_id_global) + '.png'
                new_image_address = os.path.join(image_training_address, new_image_name)
                if not os.path.exists(new_image_address):
                    shutil.copyfile(image_address, new_image_address)
                    image_name = str(image_id_global) + '.png'
                    imd_dic = {'file_name': image_name, 'id': int(image_id_global), 'width': int(480),
                               'height': int(640)}
                    coco_data['images'].append(imd_dic)

            coco_data["annotations"].append({
                "id": int(object_id_global),
                "image_id": int(image_id_global),
                "category_id": int(1),
                "segmentation": an['segmentation'],
                "area": int(an['area']),
                "bbox": an['bbox'],
                "iscrowd": int(0)
            })
            object_id_global += 1
    print('one json file is completed')
        # jason_file_name = 'combine_dataset_train.json'
        # saved_address = os.path.join(json_address, jason_file_name)
        # with open(saved_address, 'w') as destination_file:
        #     json.dump(coco_data, destination_file)
        # print('stop')

jason_file_name = 'combine_dataset_train.json'
saved_address = os.path.join(json_address, jason_file_name)
with open(saved_address, 'w') as destination_file:
    json.dump(coco_data, destination_file)

# plt.imshow(back_img)
# plt.show()s




