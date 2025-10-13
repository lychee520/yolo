# ultralytics/models/yolo/pose/val.py 的最终完整代码

from pathlib import Path
import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
# 导入DetMetrics用于只评估Box的类别
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, DetMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class PoseValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"

        # 为每个类别和每个关键点创建独立的指标计算器和统计数据存储
        self.per_class_metrics = {}
        self.per_class_stats = {}
        self.per_kpt_metrics = {}
        self.per_kpt_stats = {}

        # 兼容性：保留旧的 self.metrics 以免框架其他部分报错
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        # 使用 .get() 安全地处理可能不存在 "keypoints" 键的批次
        batch["keypoints"] = batch.get("keypoints", torch.empty(0)).to(self.device).float()
        return super().preprocess(batch)

    def init_metrics(self, model):
        """ 重构后的初始化函数，为所有评估任务准备独立的计算器和存储空间 """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        self.metrics.names = self.names

        # VVVVVV  请在这里定义哪些类别是包含关键点的 VVVVVV
        self.pose_classes_names = ["OWT"]
        # ^^^^^^  请在这里定义哪些类别是包含关键点的 ^^^^^^

        # 为每个类别创建独立的统计数据字典和指标计算器
        for i, name in self.names.items():
            self.per_class_stats[name] = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[])
            if name in self.pose_classes_names:
                self.per_class_metrics[name] = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
            else:
                self.per_class_metrics[name] = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
            self.per_class_metrics[name].names = self.names

        # 只为Pose类别创建分关键点评估器
        for name in self.pose_classes_names:
            self.per_kpt_metrics[name] = {}
            self.per_kpt_stats[name] = {}
            for k in range(self.kpt_shape[0]):
                self.per_kpt_metrics[name][k] = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
                self.per_kpt_metrics[name][k].names = self.names
                self.per_kpt_stats[name][k] = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[])

        # 兼容性：为旧的 self.stats 赋值以避免报错
        self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def _prepare_batch(self, si, batch):
        """ 健壮地准备批处理数据，处理可能没有关键点的批次 """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch.get("keypoints")
        if kpts is not None and len(kpts) > 0:
            kpts = kpts[batch["batch_idx"] == si]
            if len(kpts) > 0:
                h, w = pbatch["imgsz"]
                kpts = kpts.clone()
                kpts[..., 0] *= w
                kpts[..., 1] *= h
                pbatch["kpts"] = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"],
                                                  ratio_pad=pbatch["ratio_pad"])
            else:
                pbatch["kpts"] = torch.empty(0, self.kpt_shape[0], self.kpt_shape[1], device=pbatch['imgsz'].device)
        else:
            pbatch["kpts"] = torch.empty(0, self.kpt_shape[0], self.kpt_shape[1], device=pbatch['imgsz'].device)
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """ 健壮地准备预测数据，处理没有预测的情况 """
        predn = super()._prepare_pred(pred, pbatch)
        if predn is None or len(predn) == 0:
            empty_kpts = torch.zeros((0, self.kpt_shape[0], self.kpt_shape[1]), device=pred.device)
            return predn if predn is not None else torch.zeros((0, 6), device=pred.device), empty_kpts

        nk = self.kpt_shape[0]
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_kpts

    def update_metrics(self, preds, batch):
        """ 按类别分发统计数据 """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            npr, nl = len(pred), len(cls)
            if nl == 0: continue

            predn, pred_kpts = self._prepare_pred(pred, pbatch)
            if len(predn) == 0: continue

            # 填充总体的统计数据，用于get_stats
            self.stats['conf'].append(predn[:, 4])
            self.stats['pred_cls'].append(predn[:, 5])
            self.stats['target_cls'].append(cls)
            tp_all, tp_p_dict_all = self._process_batch(predn, bbox, cls, pred_kpts, pbatch.get("kpts"))
            if tp_all is not None: self.stats['tp'].append(tp_all)
            if tp_p_dict_all and tp_p_dict_all.get("overall") is not None: self.stats['tp_p'].append(
                tp_p_dict_all.get("overall"))

            # 按类别分发统计数据
            for i, name in self.names.items():
                gt_mask = cls == i
                pred_mask = predn[:, 5] == i
                class_gt_cls = cls[gt_mask]
                class_predn = predn[pred_mask]

                if len(class_predn) == 0 or len(class_gt_cls) == 0: continue

                stats = self.per_class_stats[name]
                stats['conf'].append(class_predn[:, 4])
                stats['pred_cls'].append(class_predn[:, 5])
                stats['target_cls'].append(class_gt_cls)

                class_gt_bbox = bbox[gt_mask]
                tp_class, tp_p_dict_class = self._process_batch(
                    class_predn, class_gt_bbox, class_gt_cls,
                    pred_kpts[pred_mask] if name in self.pose_classes_names else None,
                    pbatch.get("kpts")[gt_mask] if name in self.pose_classes_names else None)

                if tp_class is not None: stats['tp'].append(tp_class)

                if name in self.pose_classes_names and tp_p_dict_class:
                    if tp_p_dict_class.get("overall") is not None: stats['tp_p'].append(tp_p_dict_class.pop("overall"))
                    for k, tp_p_k in tp_p_dict_class.items():
                        if tp_p_k is not None:
                            kpt_stats = self.per_kpt_stats[name][k]
                            kpt_stats['conf'].append(class_predn[:, 4])
                            kpt_stats['pred_cls'].append(class_predn[:, 5])
                            kpt_stats['target_cls'].append(class_gt_cls)
                            kpt_stats['tp'].append(tp_class)
                            kpt_stats['tp_p'].append(tp_p_k)

    def get_stats(self):
        """ 重写 get_stats，汇总所有类别的统计数据以计算总体指标 """
        stats_np = self._concat_stats(self.stats)
        if stats_np:
            # 确保 tp_p 存在且不为空，否则传入一个空的 numpy 数组
            tp_p_data = stats_np.get('tp_p', np.empty((0, self.niou), dtype=np.bool_))
            self.metrics.process(stats_np.get('tp'), tp_p_data, stats_np.get('conf'), stats_np.get('pred_cls'),
                                 stats_np.get('target_cls'))
        return self.metrics.results_dict

    def print_results(self):
        """ 按照 OWT -> OWT分项 -> WPS 的顺序打印报告 """
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info(" " * 28 + "CUSTOM VALIDATION REPORT")
        LOGGER.info("=" * 80)

        class_order = ["OWT", "WPS"]
        keypoint_names = {0: "底座 (base)", 1: "转轴中心 (rotor_center)"}  # VVVVVV 在这里定义您的关键点名称 VVVVVV

        for name in class_order:
            if name not in self.names.values(): continue

            is_pose_class = name in self.pose_classes_names
            metrics = self.per_class_metrics[name]
            stats_np = self._concat_stats(self.per_class_stats[name])

            if stats_np:
                if is_pose_class:
                    LOGGER.info(f"\n--- Metrics for Class: {name} ---")
                    tp_p_data = stats_np.get('tp_p', np.empty((0, self.niou), dtype=np.bool_))
                    metrics.process(stats_np.get('tp'), tp_p_data, stats_np.get('conf'), stats_np.get('pred_cls'),
                                    stats_np.get('target_cls'))
                    self._print_table(name, metrics, self.per_class_stats[name])
                else:
                    LOGGER.info(f"\n--- Metrics for Class: {name} (Box Only) ---")
                    metrics.process(stats_np.get('tp'), stats_np.get('conf'), stats_np.get('pred_cls'),
                                    stats_np.get('target_cls'))
                    self._print_table(name, metrics, self.per_class_stats[name], is_box_only=True)

            if is_pose_class:
                LOGGER.info(f"\n--- Per-Keypoint Metrics for Class: {name} ---")
                for k in range(self.kpt_shape[0]):
                    kpt_name = keypoint_names.get(k, f"Keypoint_{k}")
                    k_metrics = self.per_kpt_metrics[name][k]
                    k_stats_np = self._concat_stats(self.per_kpt_stats[name][k])
                    if k_stats_np:
                        tp_p_data = k_stats_np.get('tp_p', np.empty((0, self.niou), dtype=np.bool_))
                        k_metrics.process(k_stats_np.get('tp'), tp_p_data, k_stats_np.get('conf'),
                                          k_stats_np.get('pred_cls'), k_stats_np.get('target_cls'))
                        self._print_table(kpt_name, k_metrics, self.per_kpt_stats[name][k], is_kpt_metric=True)

        LOGGER.info("=" * 80 + "\n")

    def _concat_stats(self, stats_dict):
        """ 辅助函数，将统计数据列表合并成单一的Numpy数组 """
        if not stats_dict or not stats_dict.get('conf') or not any(len(x) for x in stats_dict['conf']): return None
        filtered_stats = {k: v for k, v in stats_dict.items() if isinstance(v, list) and v}
        if not filtered_stats: return None
        final_stats = {k: [t for t in v if t.numel() > 0] for k, v in filtered_stats.items()}
        final_stats = {k: v for k, v in final_stats.items() if v}
        if not final_stats: return None
        return {k: torch.cat(v, 0).cpu().numpy() for k, v in final_stats.items()}

    def _print_table(self, name, metrics, stats_list, is_box_only=False, is_kpt_metric=False):
        """ 辅助函数，用于打印格式化的表格 """
        pf_header_full = "%22s" + "%11s" * 10
        pf_data_full = "%22s" + "%11i" * 2 + "%11.3g" * 8
        pf_header_box = "%22s" + "%11s" * 6
        pf_data_box = "%22s" + "%11i" * 2 + "%11.3g" * 4

        if is_kpt_metric: LOGGER.info(f"\n>> Metrics for {name}:")

        instances = len(torch.cat(stats_list['target_cls'])) if stats_list.get('target_cls') else 0

        if is_box_only:
            LOGGER.info(pf_header_box % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)"))
            box = metrics.box
            LOGGER.info(pf_data_box % (name, self.seen, instances, box.mp, box.mr, box.map50, box.map))
        else:
            LOGGER.info(pf_header_full % (
            "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)", "Pose(P", "R", "mAP50", "mAP50-95)"))
            box, pose = metrics.box, metrics.pose
            LOGGER.info(pf_data_full % (
            name, self.seen, instances, box.mp, box.mr, box.map50, box.map, pose.mp, pose.mr, pose.map50, pose.map))

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        if len(detections) == 0:
            tp_box = torch.empty(0, self.niou, dtype=torch.bool, device=self.device)
            tp_p_dict = None
            if gt_kpts is not None:
                tp_p_dict = {}
            return tp_box, tp_p_dict

        tp_box = self.match_predictions(detections[:, 5], gt_cls, box_iou(gt_bboxes, detections[:, :4]))
        tp_p_dict = None
        if pred_kpts is not None and gt_kpts is not None and len(gt_kpts) > 0 and len(pred_kpts) > 0:
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            iou_overall = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
            tp_p_overall = self.match_predictions(detections[:, 5], gt_cls, iou_overall)
            tp_p_dict = {"overall": tp_p_overall}
            for k in range(self.kpt_shape[0]):
                iou_k = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area, kpt_idx=k)
                tp_p_k = self.match_predictions(detections[:, 5], gt_cls, iou_k)
                tp_p_dict[k] = tp_p_k
        return tp_box, tp_p_dict

    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        images = batch["img"]
        # 兼容6通道数据绘图
        plot_images(images[:, :3], batch["batch_idx"], batch["cls"].squeeze(-1), batch["bboxes"],
                    kpts=batch.get("keypoints"), paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels_rgb.jpg", names=self.names, on_plot=self.on_plot)
        if images.shape[1] == 6:
            plot_images(images[:, 3:], batch["batch_idx"], batch["cls"].squeeze(-1), batch["bboxes"],
                        kpts=batch.get("keypoints"), paths=batch["im_file"],
                        fname=self.save_dir / f"val_batch{ni}_labels_ir.jpg", names=self.names, on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        if len(preds) == 0 or all(len(p) == 0 for p in preds): return
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds if len(p) > 0], 0)
        images = batch["img"]
        # 兼容6通道数据绘图
        plot_images(images[:, :3], *output_to_target(preds, max_det=self.args.max_det), kpts=pred_kpts,
                    paths=batch["im_file"], fname=self.save_dir / f"val_batch{ni}_pred_rgb.jpg", names=self.names,
                    on_plot=self.on_plot)
        if images.shape[1] == 6:
            plot_images(images[:, 3:], *output_to_target(preds, max_det=self.args.max_det), kpts=pred_kpts,
                        paths=batch["im_file"], fname=self.save_dir / f"val_batch{ni}_pred_ir.jpg", names=self.names,
                        on_plot=self.on_plot)

    def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results
        Results(np.zeros((shape[0], shape[1]), dtype=np.uint8), path=None, names=self.names, boxes=predn[:, :6],
                keypoints=pred_kpts, ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Converts YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {"image_id": image_id, "category_id": self.class_map[int(p[5])], "bbox": [round(x, 3) for x in b],
                 "keypoints": p[6:], "score": round(p[4], 5), })

    def eval_json(self, stats):
        """Evaluates object detection model using COCO JSON format."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "keypoints")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[:2]
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats