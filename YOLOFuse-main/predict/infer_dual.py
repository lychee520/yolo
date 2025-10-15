from ultralytics import YOLO
import os


def predict_images(config, output_paths):
    MODEL_PATH = config["MODEL_PATH"]
    PREDICT_CONF = config["PREDICT_CONF"]

    # 确保路径指向 LLVIP/images 文件夹
    INPUT_IMAGES_DIR = os.path.join(config["BASE_OUTPUT_DIR"], "LLVIP", "images")  # 使用切片生成的RGB图像文件夹

    model = YOLO(MODEL_PATH)

    model.predict(
        source=INPUT_IMAGES_DIR,  # 使用 LLVIP/images 文件夹中的切片图像
        save=True,  # 保存结果
        imgsz=640,  # 输入图像大小
        conf=PREDICT_CONF,  # 置信度阈值
        iou=0.45,  # IoU 阈值
        show=False,
        project=output_paths["predict"],  # 保存预测结果的文件夹
        name=f"predict_{PREDICT_CONF}",  # 使用 PREDICT_CONF 命名
        save_txt=True,
        save_conf=True,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        line_width=1,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
    )
