
import numpy as np
from PIL import Image
import urllib.request


import torch
import cv2
import requests
import json




# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator,sam_model_registry


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
def save_anno(image, anns, points_per_side, location):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    mask_all = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        
        m = [m for x in range(3)]
        m = np.swapaxes(m,0,2)
        m = np.swapaxes(m,0,1)
        img = img*m
        # ax.imshow(np.dstack((img, m*0.35)))
        mask_all.append(img)
        # print(img.shape)
    mask_all = np.array(mask_all).sum(axis=0)
    mask_all *= 255
    #np.save(f"output/{location}_{points_per_side}", mask_all)
    save_img = image/2+mask_all
    save_img = Image.fromarray(save_img.astype(np.uint8))
    save_img.save(f"output/{location}_{points_per_side}.png")




def Poly_Anything(image,  points_per_side=16):

    generic_mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)
    segmented_frame_masks = generic_mask_generator.generate(image)

    return segmented_frame_masks

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def box_to_center(box):
    xc = box[0] + box[2]/2
    yc = box[1] + box[1]/2
    return [xc, yc]


def find_the_best_mask(point, anns):
    neighs = []
    for ann in anns:
        coords = convert_box_xywh_to_xyxy(ann['bbox'])
        in_x = (point[0] >= coords[0]) and (point[0] <= coords[2])
        in_y = (point[1] <= coords[0]) and (point[1] >= coords[3])
        if in_x and in_y:
            neighs.append(ann)

    dict_ious = {}
    for n in neighs:

        dict_ious[n['predicted_iou']]= n['bbox']
    return dict_ious[max(dict_ious)]



def ann_to_polygon_ui(box, poly):
    coords = convert_box_xywh_to_xyxy(box)
    if poly:
      poly[0]["polygons"].append({"coordinates": [{ "x": coords[0], "y": coords[1] },
                                        { "x": coords[0], "y": coords[3] },
                                        { "x": coords[1], "y": coords[1] },
                                        { "x": coords[1], "y": coords[3] }],
                                         "color": "255,255,0",
                                        "opacity": "0.5"})
    else:
        poly = {"polygons":[{"coordinates": [{ "x": coords[0], "y": coords[1] },
                                          { "x": coords[0], "y": coords[3] },
                                          { "x": coords[1], "y": coords[1] },
                                          { "x": coords[1], "y": coords[3] }],
                                           "color": "255,255,0",
                                          "opacity": "0.5"}]}
    return poly
    



if __name__ == '__main__' :

    image_path = 'data/Test001.jpg'
    location = "test"

    #SAM
    sam_checkpoint = 'models/sam_vit_h_4b8939.pth'
    #sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)


    image = cv2.imread(image_path)
    w,h = image.shape[0], image.shape[1]

    points_per_side = 16 # min(w,h)//90
    anns = Poly_Anything(image, points_per_side)

    save_anno(image, anns, points_per_side,location)
    np.save(f"output/{location}_{points_per_side}", anns)

    point = [1200,800]
    ann = find_the_best_mask(point, anns)
    print(ann)
    #save_anno(image, ann, points_per_side,'point_to_poly')
    if ann:
        poly = ann_to_polygon_ui(ann, [])
        print(poly)


