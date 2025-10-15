# ronghe.py
import os
import cv2
import torch
import numpy as np
import rasterio
from torchvision.ops import nms
from tqdm import tqdm
import re


def merge_predictions(config, output_paths):
    """
    融合所有标注，根据独立阈值和类别进行筛选和美化。
    """
    # 动态生成预测文件夹路径
    # 使用预测时采用的最低阈值来定位正确的文件夹
    min_conf_for_predict = min(config['PREDICT_CONF_OWT'], config['PREDICT_CONF_WPS'])
    predict_folder_name = f"predict_{min_conf_for_predict}"
    LABELS_DIR = os.path.join(output_paths["predict"], predict_folder_name, "labels")

    LARGE_IMAGE_PATH = config["LARGE_IMAGE_RGB_PATH"]
    NMS_IOU_THRESHOLD = 0.1
    TILE_SIZE = config["TILE_SIZE"]
    OVERLAP_RATIO = config["OVERLAP_RATIO"]
    step_size = TILE_SIZE * (1 - OVERLAP_RATIO)

    all_boxes, all_scores, all_keypoints, all_class_ids = [], [], [], []

    if not os.path.exists(LABELS_DIR):
        print(f"错误：找不到标签文件夹: {LABELS_DIR}")
        return

    txt_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]

    # 获取独立的阈值
    conf_owt = config["PREDICT_CONF_OWT"]
    conf_wps = config["PREDICT_CONF_WPS"]
    class_names = ["OWT", "WPS"]  # ID 0 -> OWT, ID 1 -> WPS

    # 1. 读取所有标签，并根据独立阈值进行二次筛选
    print(f"正在使用独立阈值进行筛选 (OWT > {conf_owt}, WPS > {conf_wps})...")
    for filename in tqdm(txt_files, desc="正在读取与筛选标签"):
        match = re.search(r'(\d+)_(\d+)', filename)
        if not match: continue

        row_idx, col_idx = map(int, match.groups())
        x_offset, y_offset = col_idx * step_size, row_idx * step_size

        with open(os.path.join(LABELS_DIR, filename), 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                score = parts[-1]

                # 根据类别ID和独立阈值进行筛选
                if class_id == 0 and score < conf_owt:
                    continue
                if class_id == 1 and score < conf_wps:
                    continue

                all_class_ids.append(class_id)
                all_scores.append(score)

                cx_norm, cy_norm, w_norm, h_norm = parts[1:5]
                xmin, ymin = (cx_norm - w_norm / 2) * TILE_SIZE + x_offset, (
                            cy_norm - h_norm / 2) * TILE_SIZE + y_offset
                xmax, ymax = (cx_norm + w_norm / 2) * TILE_SIZE + x_offset, (
                            cy_norm + h_norm / 2) * TILE_SIZE + y_offset
                all_boxes.append([xmin, ymin, xmax, ymax])

                kpt_parts = parts[5:-1]
                kpts_with_vis = []
                for i in range(len(kpt_parts) // 3):
                    kpt_x, kpt_y = kpt_parts[i * 3] * TILE_SIZE + x_offset, kpt_parts[i * 3 + 1] * TILE_SIZE + y_offset
                    kpts_with_vis.append([kpt_x, kpt_y, kpt_parts[i * 3 + 2]])
                all_keypoints.append(kpts_with_vis)

    print(f"\n标签筛选与转换完成。共收集到 {len(all_boxes)} 个有效检测框（合并前）。")

    # 2. NMS 合并
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32).view(-1, 4)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    nms_indices = nms(boxes_tensor, scores_tensor, NMS_IOU_THRESHOLD)

    nms_boxes = boxes_tensor[nms_indices].numpy()
    nms_scores = scores_tensor[nms_indices].numpy()
    nms_keypoints = [all_keypoints[i] for i in nms_indices]
    nms_class_ids = [all_class_ids[i] for i in nms_indices]
    print(f"NMS合并完成。共 {len(nms_boxes)} 个独立目标。")

    # 3. 双重过滤逻辑
    final_boxes, final_scores, final_class_ids, fully_filtered_keypoints = nms_boxes, nms_scores, nms_class_ids, []
    print("正在对结果进行双重过滤...")
    for i in range(len(final_boxes)):
        box, keypoints = final_boxes[i], nms_keypoints[i]
        xmin, ymin, xmax, ymax = box
        geometrically_valid_kpts = [k for k in keypoints if xmin <= k[0] <= xmax and ymin <= k[1] <= ymax]
        if len(geometrically_valid_kpts) > 2:
            geometrically_valid_kpts.sort(key=lambda k: k[2], reverse=True)
            final_kpts_for_this_box = geometrically_valid_kpts[:2]
        else:
            final_kpts_for_this_box = geometrically_valid_kpts

        output_kpts = final_kpts_for_this_box + [[0, 0, 0]] * (len(keypoints) - len(final_kpts_for_this_box))
        fully_filtered_keypoints.append(output_kpts)
    print("双重过滤完成。")

    # 4. 可视化与保存
    try:
        with rasterio.open(LARGE_IMAGE_PATH) as src:
            large_height, large_width = src.height, src.width
            img_data = src.read([1, 2, 3])
            output_image = np.moveaxis(img_data, 0, -1).copy()
            if output_image.dtype != np.uint8:
                output_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 定义颜色和字体
        class_colors = [(3, 132, 252), (252, 3, 132)]  # OWT: 蓝色, WPS: 紫红色
        text_color = (255, 255, 255)
        kpt_color_1, kpt_color_2 = (0, 0, 255), (0, 255, 255)
        font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2

        print("正在绘制最终可视化结果...")
        for i in range(len(final_boxes)):
            box, score, class_id = final_boxes[i], final_scores[i], final_class_ids[i]
            keypoints = fully_filtered_keypoints[i]

            box_color = class_colors[class_id] if class_id < len(class_colors) else (0, 255, 0)

            pt1, pt2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(output_image, pt1, pt2, box_color, font_thickness)

            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            label = f"{class_name}: {score:.2f}"

            (w, h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(output_image, (pt1[0], pt1[1] - h - 10), (pt1[0] + w, pt1[1] - 10), box_color, -1)
            cv2.putText(output_image, label, (pt1[0], pt1[1] - 5), font, font_scale, text_color, font_thickness)

            valid_kpts = [k for k in keypoints if k[2] > 0]
            if len(valid_kpts) >= 1: cv2.circle(output_image, (int(valid_kpts[0][0]), int(valid_kpts[0][1])), 5,
                                                kpt_color_1, -1)
            if len(valid_kpts) >= 2: cv2.circle(output_image, (int(valid_kpts[1][0]), int(valid_kpts[1][1])), 5,
                                                kpt_color_2, -1)

        cv2.imwrite(output_paths["visual"], cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        print(f"可视化图像已保存至: {output_paths['visual']}")

        with open(output_paths["label"], 'w') as f:
            for i in range(len(final_boxes)):
                class_id, box, score, keypoints = final_class_ids[i], final_boxes[i], final_scores[i], \
                fully_filtered_keypoints[i]
                xmin, ymin, xmax, ymax = box
                norm_cx, norm_cy = ((xmin + xmax) / 2) / large_width, ((ymin + ymax) / 2) / large_height
                norm_w, norm_h = (xmax - xmin) / large_width, (ymax - ymin) / large_height
                line_parts = [f"{class_id}", f"{norm_cx:.6f}", f"{norm_cy:.6f}", f"{norm_w:.6f}", f"{norm_h:.6f}"]
                for kpt in keypoints:
                    line_parts.extend([f"{kpt[0] / large_width:.6f}", f"{kpt[1] / large_height:.6f}", f"{kpt[2]:.6f}"])
                line_parts.append(f"{score:.6f}")
                f.write(" ".join(line_parts) + "\n")
        print(f"最终标签文件已保存至: {output_paths['label']}")

    except Exception as e:
        print(f"\n在处理大图或保存文件时发生错误: {e}")