import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import chain

import data_augmentation as DAG
import json
def resize():
    foolder = '/home/ehsan/PycharmProjects/data_augmentation/background_edit'
    folder_save = '/home/ehsan/PycharmProjects/data_augmentation/background_edit1/'
    images = os.listdir(foolder)
    for img in images:
        img_ = os.path.join(foolder, img)
        original = cv2.imread(img_)
        resize_image = cv2.resize(original, (640, 480))
        img_save = folder_save + img
        cv2.imwrite(img_save, resize_image)
def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res_, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    res = []
    for contour_ in res_:
        area = cv2.contourArea(contour_)
        if area > 1000:
            res.append(contour_)

    if hierarchy is None:  # empty mask
        return [], False

    # TODO speed this search up
    # new_polys = res.copy()
    new_polys = res [:]
    indices_to_keep = list(range(len(res)))
    for i, r1 in enumerate(res):
        for j, r2 in enumerate(res):
            if i == j:
                continue
            if np.in1d(r2.ravel(), r1.ravel()).all():
                if len(r2) > len(r1):
                    if i in indices_to_keep:
                        indices_to_keep.remove(i)
                elif len(r1) > len(r2):
                    if j in indices_to_keep:
                        indices_to_keep.remove(j)

    res = [val for i, val in enumerate(new_polys) if i in indices_to_keep]

    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x for x in res if len(x) >= 6]
    return res, has_holes

def data_engin(image_path, packages, background_path, aug_background, aug_package, image_id, object_id ,coco_data):
    backgrounds = os.listdir(background_path)
    start_range_package = 0; end_range_package = len(packages) - 1
    number_packages_in_scene = 100
    json_address = 'json_file/'
    for back_gr in backgrounds:
        background = os.path.join(background_path, back_gr)
        # back_img = cv2.resize(back_img, (640))
        #back_img_save = cv2.resize(back_img_save, (480, 640))
        #package_name = packages[random_number]
        for package_name in packages:
            back_img = cv2.imread(background)
            back_img_save = cv2.imread(background)
            segmentation_map = np.zeros((480, 640, 3))
            for i in range(1, number_packages_in_scene):
                random_number = random.randint(start_range_package, end_range_package)
                package_path = os.path.join(image_path, package_name)
                picked_img = os.listdir(package_path)
                start_range_img = 0; end_range_img = len(picked_img) - 1

                random_number_img = random.randint(start_range_img, end_range_img)
                img_name = picked_img[random_number_img]
                img_path = os.path.join(package_path, img_name)
                image_input = cv2.imread(img_path)
                # image_input = cv2.resize(image_input, (640, 480))
                # image_cp = image_
                back_img = np.where(image_input == 0, back_img, image_input)
                back_img_save = np.where(image_input == 0, back_img_save, image_input)
                segmentation_map = np.where(image_input > 0, i, segmentation_map)
                segmentation_ploygon = []
                image_id += 1
                for an in range(1, i + 1):
                    mask = np.where(segmentation_map == an, 1, 0).astype(np.uint8)
                    mask_ = mask[:, :, 0]
                    check_area = sum(sum(mask_))
                    if check_area > 1000:
                    # kernel = np.ones((10, 10), np.uint8)
                    # mask_ = cv2.morphologyEx(mask_, cv2.MORPH_CLOSE, kernel)
                    # area = sum(sum(mask_))
                    #     plt.imshow(mask_)
                    #     plt.show()
                        res, has_hole = mask_to_polygons(mask_)
                        for obj in range(len(res)):
                            random_number = random.randint(0, 5)
                            polygon_sample = np.array(res[obj]).reshape((-1, 2)).astype(np.int32)
                            cv2.drawContours(back_img, [polygon_sample], 0, colors[random_number], 2)
                            X = [];  Y =[]
                            object_id += 1
                            for point in polygon_sample:
                                X.append(point[0]); Y.append(point[1])
                                bbox = [int(min(X)), int(min(Y)), int(abs(max(X)-min(X))), int(abs(max(Y)-min(Y)))]
                            area = abs(max(X)-min(X)) * abs(max(Y)-min(Y))
                            if area > 1000:
                                coco_data["annotations"].append({
                                    "id": int(object_id),
                                    "image_id": int(image_id),
                                    "category_id": int(1),
                                    "segmentation": [res[obj].tolist()],
                                    "area": int(area),
                                    "bbox": bbox,
                                    "iscrowd": int(0)
                                })

                image_name = str(image_id) + '.png'
                imd_dic = {'file_name': image_name, 'id': int(image_id), 'width': int(480), 'height': int(640)}
                coco_data['images'].append(imd_dic)
                if image_id % 1000 == 0:
                    json_name = json_address+'myjson'+str(image_id//1000)+'.json'
                    with open(json_name, 'w') as destination_file:
                         json.dump(coco_data, destination_file)

                name_ = 'synthetic_images/' + str(image_id) + '.png'
                cv2.imwrite(name_, back_img_save)
                annotated_name = 'annotated_images/' + str(image_id) + '.png'
                cv2.imwrite(annotated_name, back_img)
                # plt.imshow(back_img)
                # plt.show()
    return image_id, object_id
 # ! a very simple approach for finding the countor no preprocedding is needed
 #                contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 #                # # Convert contours to polygons
 #                max_area = 0
 #                contour = [0]
 #                for contour_ in contours:
 #                    area = cv2.contourArea(contour_)
 #                    # list_area.append(area)
 #                    if area > max_area:
 #                        max_area = area
 #                        contour = np.squeeze(contour_).tolist()
 #
 #                flattened_list = []
 #                if isinstance(contour[0], list):
 #                    X=[]; Y =[]
 #                    for point in contour:
 #                        X.append(point[0]); Y.append(point[1])
 #                    bbox = [min(X), min(Y), abs(max(X)-min(X)), abs(max(Y)-min(Y))]
 #                    flattened_list = list(chain(*contour))
 #                    # print(flattened_list)
 #                    segmentation_ploygon.append(flattened_list)
                # Display the polygon segmentation
                # print(segmentation_ploygon)
                # print('stop right here')
                # rgb_image_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                # rgb_image_mask[:, :, 0] = mask
                # rgb_image_mask[:, :, 1] = mask
                # rgb_image_mask[:, :, 2] = mask

                # if len(flattened_list) > 0:
                #     coco_data["annotations"].append({
                #         "id": global_counter,
                #         "image_id": global_counter,
                #         "category_id": 1,
                #         "segmentation": [flattened_list],
                #         "area": max_area,
                #         "bbox": bbox,
                #         "iscrowd": 0
                #     })
                #     if global_counter % 100==0:
                #         with open(json_address, 'w') as destination_file:
                #             json.dump(coco_data, destination_file)
                #     global_counter += 1
                #     name_ = 'synthetic_images/' + str(global_counter) + '.png'
                #     anotated_name = 'annotated_images/' + str(global_counter) + '.png'
                #     cv2.imwrite(name_, back_img)
                #     imd_dic = {'name': str(global_counter) + '.png', 'width': 480, 'height': 640}
                #     coco_data['images'].append(imd_dic)
            # for segmentation_ in segmentation_ploygon:
            #     random_number = random.randint(0, 5)
            #     polygon_sample = np.array(segmentation_).reshape((-1, 2)).astype(np.int32)
            #     cv2.drawContours(back_img, [polygon_sample], 0, colors[random_number], 2)
            # x, y, width, height = item['bbox']
            # cv2.rectangle(image_holder, (x, y), (x + width, y + height), colors[random_number], 2)
            # plt.imshow(back_img)
            # plt.show()

            # shape = image_.shape
            # for chanel in range(3):
            #     for row in range(shape[0]):
            #         for col in range(shape[1]):
            #             if image_[row, col, chanel] > 0:
            #                 back_img[row, col, chanel] = image_[row, col, chanel]
            #             # else:
            #             #     back_img[row, col, chanel] = image_[row, col, chanel]

            # back_img += image_
        # plt.imshow(back_img)
        # plt.show()
        #     cv2.imwrite(anotated_name, back_img)

if __name__ == '__main__':
    image_path = './cropped_dataset_1000/'
    packages = ['Box', 'Envelope', 'Flats', 'Polybag']
    background_path = '/home/ehsan/PycharmProjects/data_augmentation/background_edit1'
    resize()
    aug_background = ''; aug_package = ''
    global_counter = 0; image_id = -1; object_id = -1
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    coco_data = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    coco_data["categories"].append({"id": 1, "name": "package"})
    while True:
         image_id, object_id = data_engin(image_path, packages, background_path, aug_background, aug_package, image_id, object_id ,coco_data)
         print(image_id, object_id)




