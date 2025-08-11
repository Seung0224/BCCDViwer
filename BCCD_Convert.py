import os

from BCCD_voc2coco_utils import (
    get_label2id,
    get_annpaths,
    convert_xmls_to_cocojson,
)

# 항상 슬래시(/)만 쓰도록 BASE를 / 로 지정
BASE = "D:/BCCDViwer/BCCD_Dataset/BCCD"

# join -> / 로 강제 변환하는 헬퍼
def j(*parts) -> str:
    return os.path.join(*parts).replace("\\", "/")

LABELS_TXT = j(BASE, "labels.txt")
ANN_DIR    = j(BASE, "Annotations")
IDS_DIR    = j(BASE, "ImageSets", "Main")

EXTRACT_NUM_FROM_IMGID = False
label2id = get_label2id(LABELS_TXT)

splits = ["train", "val", "test", "trainval"]
for split in splits:
    ann_ids_path = j(IDS_DIR, f"{split}.txt")
    out_jsonpath = j(BASE, f"{split}.json")

    ann_paths = get_annpaths(
        ann_dir_path=ANN_DIR,
        ann_ids_path=ann_ids_path,
        ext="xml",
        annpaths_list_path=None
    )

    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=out_jsonpath,
        extract_num_from_imgid=EXTRACT_NUM_FROM_IMGID
    )
    print(f"[OK] {split}: {out_jsonpath}")
