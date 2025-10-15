# main_controller.py
import os
from cut import slice_images
from infer_dual import predict_images
from ronghe import merge_predictions

# =================================================================================
# --- 配置区域：您需要修改的所有参数都在这里 ---
# =================================================================================
config = {
    # --- 1. 主要文件路径 ---
    "LARGE_IMAGE_RGB_PATH": r"E:\arcgis S2 winter RGB\143-1.tif",
    "LARGE_IMAGE_IR_PATH": r"E:\s1 winter 8bit\143-1.tif",
    "MODEL_PATH": "./runs_newdata/多模态/11-elff-LFEM-RGB-pose-1007/weights/best.pt",
    "BASE_OUTPUT_DIR": "./yolo_pipeline_results-4",#修改名字

    # --- 2. 【核心】为 OWT 和 WPS 设置独立的置信度阈值 ---
    "PREDICT_CONF_OWT": 0.51,  # OWT 的保留阈值
    "PREDICT_CONF_WPS": 0.35,  # WPS 的保留阈值 (您可以根据需要设置得更低)

    # --- 3. 其他处理参数 ---
    "TILE_SIZE": 640,
    "OVERLAP_RATIO": 0.2,
}


# 自动生成输出文件名 (已更新以反映两个阈值)
def generate_output_names():
    # 文件名后缀现在会包含两个阈值，方便追溯
    conf_suffix = f"-owt{config['PREDICT_CONF_OWT']}-wps{config['PREDICT_CONF_WPS']}"

    # 预测时使用的项目名称
    # 注意：预测时我们会使用较低的阈值，所以项目文件夹名也反映这一点
    min_conf_for_predict = min(config['PREDICT_CONF_OWT'], config['PREDICT_CONF_WPS'])
    predict_project_name = f"predict_{min_conf_for_predict}"

    return {
        "RGB": os.path.join(config["BASE_OUTPUT_DIR"], f"LLVIP/images{conf_suffix}"),
        "IR": os.path.join(config["BASE_OUTPUT_DIR"], f"LLVIP/imagesIR{conf_suffix}"),
        "predict": os.path.join(config["BASE_OUTPUT_DIR"], "predict_output"),  # 预测的根目录
        "visual": os.path.join(config["BASE_OUTPUT_DIR"], f"merged_visual_result{conf_suffix}.jpg"),
        "label": os.path.join(config["BASE_OUTPUT_DIR"], f"merged_labels{conf_suffix}.txt")
    }


def main():
    """主函数，按顺序执行切分、预测、融合三个步骤。"""
    output_paths = generate_output_names()

    # 确保基础输出目录存在
    os.makedirs(config["BASE_OUTPUT_DIR"], exist_ok=True)

    # 步骤 1: 切片影像
    print("--- [步骤 1/3] 开始执行图像切分 ---")
    slice_images(config, output_paths)
    print("--- 图像切分完成 ---\n")

    # 步骤 2: 执行预测
    # 创建一个用于预测的临时配置，使用两个阈值中较低的那个
    # 这是“宽进严出”策略，确保所有可能的目标都被检测出来，以便后续精确筛选
    predict_config = config.copy()
    min_conf = min(config["PREDICT_CONF_OWT"], config["PREDICT_CONF_WPS"])
    predict_config["PREDICT_CONF"] = min_conf
    print(f"--- [步骤 2/3] 开始执行模型预测 (使用最低阈值 {min_conf}) ---")
    predict_images(predict_config, output_paths)
    print("--- 模型预测完成 ---\n")

    # 步骤 3: 合并标签和预测结果
    # 融合时会使用各自独立的阈值进行精确筛选
    print("--- [步骤 3/3] 开始执行结果融合与筛选 ---")
    merge_predictions(config, output_paths)
    print("--- 结果融合完成 ---\n")
    print("🎉 全部步骤成功完成！")


if __name__ == "__main__":
    main()