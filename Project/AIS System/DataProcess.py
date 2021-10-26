import os
import cv2
import numpy as np
import xml.dom.minidom
import json
import Config as config

svd = np.linalg.svd


def gen_mask_aif(path_file, path_mask_dst):
    """This is for generating the bbox area mask of the manual selected AIF area.

    To open labelImg:
    open Anaconda Powershell Prompt.
    cd to labelImg folder.
    python labelImg.py

    Args:
        - path_file (str): path includes the xml file.
        - path_mask_dst (str): ...

    Returns: None
    """

    global path_xml_dst

    # ---- get path of the xml file ----
    list_files = os.listdir(path_file)
    for f in list_files:
        if f.endswith(".xml"):
            path_xml_dst = r"{}/{}".format(path_file, f)
            break

    DOMTree = xml.dom.minidom.parse(path_xml_dst)
    annotation = DOMTree.documentElement

    # ---- set AIF area -> 255 ----
    mask_aif = np.zeros([256, 256])
    objects = annotation.getElementsByTagName("object")
    for object in objects:
        bndboxs = object.getElementsByTagName("bndbox")
        for bndbox in bndboxs:
            xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
            mask_aif[ymin:ymax, xmin:xmax] = 255

    # ---- saving AIF mask ----
    name_mask_aif = "{}_case_{}_{}_aif.png".format(config.global_patient, config.global_case, config.global_slc)

    if os.path.exists(path_mask_dst) == False:
        os.mkdir(path_mask_dst)
    cv2.imwrite(r"/{}/{}".format(path_mask_dst, name_mask_aif), mask_aif)


def gen_mask_brain(path_file, path_mask_dst):
    """This is for generating the polygonal area mask of the manual selected brain area.

    To open labelme:
    open Anaconda Powershell Prompt.
    # labelme

    Args:
        - path_file (str): path includes the json file.
        - path_mask_dst (str): ...

    Returns: None
    """

    global path_json_dst

    # ---- get path of the json file ----
    list_files = os.listdir(path_file)
    for f in list_files:
        if f.endswith(".json"):
            path_json_dst = r"{}/{}".format(path_file, f)
            break

    # parse json file.
    mask_brain = np.zeros([256, 256])
    f = open(path_json_dst, encoding='utf-8')
    setting = json.load(f)
    shapes = setting['shapes']
    contours = []
    for i in range(len(shapes)):
        contour = []
        points = shapes[i]["points"]
        contour.append(points)
        contour_np = np.array(contour)
        c = contour_np.astype(int)
        contours.append(c)

    for contour in contours:
        cv2.drawContours(mask_brain, contour, 0, (255, 255, 255), cv2.FILLED)

    # saving AIF mask.
    name_mask_brain = "{}_case_{}_{}_brain.png".format(config.global_patient, config.global_case, config.global_slc)
    if os.path.exists(path_mask_dst) == False:
        os.mkdir(path_mask_dst)
    cv2.imwrite(r"/{}/{}".format(path_mask_dst, name_mask_brain), mask_brain)
