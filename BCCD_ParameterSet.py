from mmdet.apis import set_random_seed
from mmcv import Config
import os
import yaml
from BCCD_Main import BASE_DIR

config_file = BASE_DIR + r"\mmdetection\configs\faster_rcnn\faster_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file)
print(cfg.pretty_text)

# dataset
cfg.dataset_type = 'BCCDDataset'
cfg.data_root = BASE_DIR + r"\BCCD_Dataset\BCCD"

cfg.data.train.type = 'BCCDDataset'
cfg.data.train.data_root = BASE_DIR + r"\BCCD_Dataset\BCCD"
cfg.data.train.ann_file = 'train.json'
cfg.data.train.img_prefix = 'JPEGImages'

cfg.data.val.type = 'BCCDDataset'
cfg.data.val.data_root = BASE_DIR + r"\BCCD_Dataset\BCCD"
cfg.data.val.ann_file = 'val.json'
cfg.data.val.img_prefix = 'JPEGImages'

cfg.data.test.type = 'BCCDDataset'
cfg.data.test.data_root = BASE_DIR + r"\BCCD_Dataset\BCCD"
cfg.data.test.ann_file = 'test.json'
cfg.data.test.img_prefix = 'JPEGImages'

cfg.model.roi_head.bbox_head.num_classes = 3
cfg.load_from = BASE_DIR + r"\mmdetection\checkpoints\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

cfg.work_dir = BASE_DIR + r"\mmdetection\runs\tutorial_exps"
os.makedirs(cfg.work_dir, exist_ok=True)

cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

cfg.lr_config.policy = 'step'
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

print(cfg.pretty_text)

# 1) 경로 정리(모든 "\" → "/")
def _fix_paths(node):
    if isinstance(node, dict):
        return {k: _fix_paths(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        t = type(node)
        return t(_fix_paths(x) for x in node)
    if isinstance(node, str) and "\\" in node:
        return node.replace("\\", "/")
    return node

fixed = _fix_paths(cfg._cfg_dict.to_dict())

yaml_file = BASE_DIR + r"\mmdetection\runs\my_custom_cfg.yaml"
with open(yaml_file, "w", encoding="utf-8") as f:
    yaml.dump(fixed, f, allow_unicode=True, sort_keys=False)

print("Config saved to:", yaml_file)