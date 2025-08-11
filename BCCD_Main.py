import glob
import cv2
import torch
import mmcv
import os.path as osp
import numpy as np
from matplotlib.widgets import Button

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, inference_detector, init_detector, show_result_pyplot
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from matplotlib import pyplot as plt


# =========================
# Dataset Registry
# =========================
@DATASETS.register_module(force=True)
class BCCDDataset(CocoDataset):
    CLASSES = ('WBC', 'RBC', 'Platelets')


# =========================
# User Settings (EDIT ME)
# =========================
CONFIG_FILE = r"D:\BCCDViwer\mmdetection\runs\my_custom_cfg.yaml"   # mmDetection cfg 경로
WORK_DIR    = r"D:\BCCDViwer\mmdetection\runs"                      # 체크포인트 저장 폴더 (cfg.work_dir과 일치 권장)
IMG_PATH    = r"D:\BCCDViwer\BCCD_Dataset\BCCD\JPEGImages\BloodImage_00007.jpg"  # 테스트 이미지
# 학습된 체크포인트(.pth) 직접 지정하고 싶으면 아래 변수에 넣어도 됨. 빈 문자열이면 가장 최신 체크포인트 자동 선택
CHECKPOINT_FILE = r"D:\BCCDViwer\mmdetection\runs\epoch_12.pth"


# =========================
# Device Helper (CUDA→CPU 자동 폴백)
# =========================
def pick_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


# =========================
# Utility: 최신 체크포인트 찾기
# =========================
def find_latest_checkpoint(work_dir):
    pths = glob.glob(osp.join(work_dir, "*.pth"))
    if not pths:
        return None
    pths.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return pths[0]


# =========================
# Build / Train
# =========================
def build_everything(config_file: str):
    if not osp.isfile(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")

    cfg = Config.fromfile(config_file)

    # work_dir 보정 (cfg에 없거나 다른 곳이면 WORK_DIR 사용)
    cfg.work_dir = WORK_DIR
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # 데이터셋/모델 빌드
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    # 클래스 셋팅
    model.CLASSES = datasets[0].CLASSES
    return cfg, datasets, model


def train(config_file: str):
    cfg, datasets, model = build_everything(config_file)

    # 러너/로그/시드 등은 cfg에서 제어
    print(f"[Train] work_dir = {cfg.work_dir}")
    print(f"[Train] epochs  = {getattr(cfg.runner, 'max_epochs', 'cfg.runner에서 설정')}")
    print(f"[Train] classes = {model.CLASSES}")

    # 학습
    train_detector(model, datasets, cfg, distributed=False, validate=True)

    print("[Train] Done. Checkpoints saved to:", cfg.work_dir)


# =========================
# Inference
# =========================
def load_model_for_inference(config_file: str, checkpoint_file: str = ""):
    device = pick_device()
    print(f"[Predict] Using device: {device}")

    # 체크포인트 결정
    ckpt = checkpoint_file if checkpoint_file else find_latest_checkpoint(WORK_DIR)
    if ckpt is None or not osp.isfile(ckpt):
        raise FileNotFoundError(
            "No checkpoint found. "
            "Set CHECKPOINT_FILE or ensure there is a .pth in WORK_DIR."
        )

    # 가장 간단/안전: init_detector 사용 (CPU/ GPU 자동 적용)
    model = init_detector(config_file, ckpt, device=device)

    # (선택) 클래스 이름 강제 지정: cfg 데이터셋과 동일해야 시각화 라벨이 맞음
    if not hasattr(model, 'CLASSES') or not model.CLASSES:
        # cfg에서 train dataset을 다시 읽어와서 클래스 가져오기
        cfg = Config.fromfile(config_file)
        ds = build_dataset(cfg.data.test if 'test' in cfg.data else cfg.data.val)
        model.CLASSES = getattr(ds, 'CLASSES', ('WBC', 'RBC', 'Platelets'))

    return model


def predict_image(config_file: str, img_path: str, checkpoint_file: str = "", show: bool = True, out_file: str = ""):
    if not osp.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    model = load_model_for_inference(config_file, checkpoint_file)

    # --- 이미지 목록 준비 (같은 폴더의 .jpg 정렬) ---
    img_dir = osp.dirname(img_path)
    all_imgs = sorted(
        [p for p in glob.glob(osp.join(img_dir, "*.jpg")) + glob.glob(osp.join(img_dir, "*.png"))]
    )
    if not all_imgs:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    # 시작 인덱스 결정
    try:
        idx = all_imgs.index(osp.abspath(img_path))
    except ValueError:
        idx = 0

    # --- 간단한 박스 오버레이 도우미 (bbox-only; Mask R-CNN도 bbox는 그려짐) ---
    def draw_overlay(img_bgr, result, classes, score_thr=0.3, alpha=0.45):
        overlay = img_bgr.copy()
        # 결과 포맷: list[np.ndarray] or (bbox_result, segm_result)
        bbox_result = result[0] if isinstance(result, tuple) else result

        # ★ 클래스별 고정 색상 (BGR)
        class_colors = {
            'WBC': (0, 255, 0),       # 초록
            'RBC': (0, 0, 255),       # 빨강
            'Platelets': (255, 0, 0)  # 파랑
        }

        for cls_idx, bboxes in enumerate(bbox_result):
            if bboxes is None or len(bboxes) == 0:
                continue

            cls_name = classes[cls_idx] if classes and cls_idx < len(classes) else str(cls_idx)
            color = class_colors.get(cls_name, (255, 255, 255))  # 정의 외 클래스는 흰색

            for bx in bboxes:
                x1, y1, x2, y2, score = bx.astype(float)
                if score < score_thr:
                    continue
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(overlay, p1, p2, color, 2)
                label = f"{cls_name} {score:.2f}"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (p1[0], p1[1]-th-bl), (p1[0]+tw, p1[1]), color, -1)
                cv2.putText(
                    overlay, label, (p1[0], p1[1]-bl),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
        return overlay
        
    # --- 한 장 추론 ---
    classes = getattr(model, 'CLASSES', ('WBC', 'RBC', 'Platelets'))
    def infer_one(path):
        img_bgr = cv2.imread(path)
        result = inference_detector(model, img_bgr)
        overlay = draw_overlay(img_bgr, result, classes, score_thr=0.3, alpha=0.45)
        # matplotlib 표시 위해 RGB로 변환
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), result

    # --- Matplotlib UI 구성: 이미지 + 좌/우 버튼 ---
    if not show:
        # show=False면 첫 장만 저장하고 끝
        img_rgb, result = infer_one(all_imgs[idx])
        if out_file:
            # mmdet 기본 저장
            model.show_result(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), result, out_file=out_file, score_thr=0.3)
            print(f"[Predict] Saved:", out_file)
        return result

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # 버튼 영역 확보

    img_rgb, result = infer_one(all_imgs[idx])
    imshow_obj = ax.imshow(img_rgb)
    ax.set_title(f"{osp.basename(all_imgs[idx])}  [{idx+1}/{len(all_imgs)}]")
    ax.axis('off')

    # 버튼 영역
    axprev = plt.axes([0.25, 0.05, 0.15, 0.08])
    axnext = plt.axes([0.60, 0.05, 0.15, 0.08])
    bprev = Button(axprev, '◀ Prev')
    bnext = Button(axnext, 'Next ▶')

    def update_display():
        nonlocal imshow_obj, idx
        img_rgb2, _ = infer_one(all_imgs[idx])
        imshow_obj.set_data(img_rgb2)
        ax.set_title(f"{osp.basename(all_imgs[idx])}  [{idx+1}/{len(all_imgs)}]")
        fig.canvas.draw_idle()

    def on_prev(event):
        nonlocal idx
        idx = (idx - 1) % len(all_imgs)
        update_display()

    def on_next(event):
        nonlocal idx
        idx = (idx + 1) % len(all_imgs)
        update_display()

    bprev.on_clicked(on_prev)
    bnext.on_clicked(on_next)

    # 키보드 화살표도 지원
    def on_key(event):
        if event.key in ['left', 'a']:
            on_prev(None)
        elif event.key in ['right', 'd']:
            on_next(None)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=True)  # 창 유지

    # 마지막으로 본 결과를 반환 (필요 시)
    return result


# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) 학습하려면 주석 해제
    # train(CONFIG_FILE)

    # 2) 추론
    res = predict_image(
        config_file=CONFIG_FILE,
        img_path=IMG_PATH,
        checkpoint_file=CHECKPOINT_FILE,  # 빈 값이면 WORK_DIR에서 최신 .pth 자동 탐색
        show=True,                        # GUI 없는 서버라면 False
        out_file=osp.join(WORK_DIR, "pred_result.jpg")
    )
    print("[Predict] Done. Result type:", type(res))
