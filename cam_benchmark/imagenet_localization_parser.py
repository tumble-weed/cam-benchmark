# import settings
import os
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from typing import Dict,Any
import collections
# voc_root_dir = os.path.join(
#         os.path.dirname(settings.RESULTS_DIR),
#         'VOCdevkit/VOC2007/')
voc_root_dir = None
def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict
def get_voc_label(
    root_dir = voc_root_dir,
    x = '000003',
    full_filename = None):
    
    # annotation_dir = os.path.join(root_dir,'Annotations')
    if full_filename is None:
        annotation_dir = root_dir
        
        target = os.path.join(
                    annotation_dir, x + ".xml")
        target = parse_voc_xml(ET_parse(target).getroot())
        return target
    else:
        target = parse_voc_xml(ET_parse(full_filename).getroot())
        return target

def get_bndbox(
    root_dir = voc_root_dir,
    x = '000003'):
    label = get_voc_label(
    root_dir = root_dir,
    x = x)
    size = label['annotation']['size']
    size = (int(size['height']),int(size['width']))
    bboxes = {}
    for object in label['annotation']['object']:
        # print(object['name'])
        bbox_dict = object['bndbox']
        y0,y1,x0,x1 = bbox_dict['ymin'],bbox_dict['ymax'],bbox_dict['xmin'],bbox_dict['xmax']
        y0,y1,x0,x1 = int(y0),int(y1),int(x0),int(x1)
        assert y1 > y0
        assert x1 > x0
        bboxes[object['name']] = [y0,y1,x0,x1]
    return bboxes,size
def rescale_bbox(y0y1x0x1,original_size,new_size):
    y0,y1,x0,x1 = y0y1x0x1[0],y0y1x0x1[1],y0y1x0x1[2],y0y1x0x1[3]
    y_scale = new_size[0]/original_size[0]
    x_scale = new_size[1]/original_size[1]
    new_y0y1 = y0*y_scale,y1*y_scale
    new_x0x1 = x0*x_scale,x1*x_scale
    new_y0y1x0x1 = (*new_y0y1,*new_x0x1)
    return new_y0y1x0x1

def get_gt_bbox(
    image_root,
    class_name,
    size,
    root_dir= '/root/evaluate-saliency-4/VOCdevkit/VOC2007/'
    ):
    original_gt_bbox,original_size = get_bndbox(
        root_dir,
        x = image_root)
    # gt_bbox
    gt_bbox = rescale_bbox(
        original_gt_bbox[class_name],
        original_size,
        size
        )
    return gt_bbox

if __name__ == '__main__':
    bboxes,size = get_bndbox(
        root_dir = voc_root_dir,
        x = '000003')
    print(bboxes)
    print(size)



