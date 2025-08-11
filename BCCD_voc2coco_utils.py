import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
import re

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x): return x

# labels.txt 파일로부터 라벨명과 ID 매핑 정보 생성
# Get label to ID mapping from labels.txt
def get_label2id(labels_path: str) -> Dict[str, int]:
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_str = [x.strip() for x in f.read().split() if x.strip()]
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))

# 어노테이션(XML) 파일 경로 목록 생성
# Get annotation paths from various sources
def get_annpaths(ann_dir_path: str, ann_ids_path: str, ext: str = 'xml', annpaths_list_path: str = None) -> List[str]:
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r', encoding='utf-8') as f:
            ann_paths = [line.strip() for line in f if line.strip()]
        return ann_paths

    with open(ann_ids_path, 'r', encoding='utf-8') as f:
        ann_ids = [line.strip() for line in f if line.strip()]
    ext_with_dot = ('.' + ext) if ext and not ext.startswith('.') else (ext or '')
    return [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]

# XML 어노테이션의 루트에서 이미지 정보 추출
# Get image information from XML annotation root
def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    filename = os.path.basename(path) if path else annotation_root.findtext('filename')

    img_name = os.path.basename(filename)
    img_id_raw = os.path.splitext(img_name)[0]  # 확장자 제거한 파일명

    if extract_num_from_imgid:
        m = re.findall(r'\d+', img_id_raw)
        if m:
            img_id = int(m[0])
        else:
            img_id = abs(hash(img_id_raw)) % (10**9)
    else:
        img_id = img_id_raw

    size = annotation_root.find('size')
    width = int(float(size.findtext('width')))
    height = int(float(size.findtext('height')))

    return {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }

# XML의 object 태그에서 COCO 형식의 어노테이션 정보 추출
# Extract COCO annotation from XML object
def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id ! (labels.txt 확인)"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    return {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []
    }

# 여러 XML 어노테이션을 COCO JSON 형식으로 변환
# Convert XML annotations to COCO JSON format
def convert_xmls_to_cocojson(annotation_paths: List[str], label2id: Dict[str, int], output_jsonpath: str, extract_num_from_imgid: bool = True):
    output_json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    bnd_id = 1
    for a_path in tqdm(annotation_paths):
        if not os.path.exists(a_path):
            print(f"[WARN] XML not found: {a_path}")
            continue

        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(ann_root, extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id += 1

    for label, label_id in label2id.items():
        output_json_dict['categories'].append({
            'supercategory': 'none',
            'id': label_id,
            'name': label
        })

    os.makedirs(os.path.dirname(output_jsonpath) or ".", exist_ok=True)
    with open(output_jsonpath, 'w', encoding='utf-8') as f:
        json.dump(output_json_dict, f, ensure_ascii=False)
    print(f"[OK] COCO saved -> {output_jsonpath}")