# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#a
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class PoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseValidator

        args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
        validator = PoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"

        ### --- 修改开始 --- ###
        # 主指标计算器，用于计算总体结果
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        # 为每个关键点创建独立的指标计算器
        self.per_kpt_metrics = {}
        ### --- 修改结束 --- ###

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

        ### --- 修正开始 --- ###
        # 主统计数据字典
        self.stats = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

        # 将从数据文件加载的names字典同步给主metrics对象
        self.metrics.names = self.names

        # 为每个关键点创建独立的统计数据字典和指标计算器
        self.per_kpt_stats = {}
        for k in range(self.kpt_shape[0]):
            self.per_kpt_stats[k] = dict(tp_p=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
            self.per_kpt_metrics[k] = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
            # 关键修复：将names字典也同步给每个独立的metrics对象
            self.per_kpt_metrics[k].names = self.names
        ### --- 修正结束 --- ###
    def _prepare_batch(self, si, batch):
        """Prepares a batch for processing by converting keypoints to float and moving to device."""
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["kpts"] = kpts
        return pbatch

    def _prepare_pred(self, pred, pbatch):
        """Prepares and scales keypoints in a batch for pose processing."""
        predn = super()._prepare_pred(pred, pbatch)
        nk = pbatch["kpts"].shape[1]
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        ops.scale_coords(pbatch["imgsz"], pred_kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        return predn, pred_kpts

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    # 为主 stats 和每个 per_kpt_stats 填充空数据
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    for k_idx in range(self.kpt_shape[0]):
                        # 只填充必要的key以避免错误
                        for key in ("tp_p", "tp", "conf", "pred_cls", "target_cls", "target_img"):
                            self.per_kpt_stats[k_idx][key].append(stat[key])
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_kpts = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"], tp_p_dict = self._process_batch(predn, bbox, cls, pred_kpts, pbatch["kpts"])

                # 将总体的 tp_p 分配给主 stat
                stat["tp_p"] = tp_p_dict.pop("overall")

                # 将每个关键点的 tp_p 和其他统计数据分配给对应的 per_kpt_stats
                for k, tp_p_k in tp_p_dict.items():
                    kpt_stat = {
                        "conf": stat["conf"],
                        "pred_cls": stat["pred_cls"],
                        "target_cls": stat["target_cls"],
                        "tp_p": tp_p_k,
                        "tp": stat["tp"],
                        "target_img": stat["target_img"],
                    }
                    for key, val in kpt_stat.items():
                        self.per_kpt_stats[k][key].append(val)

            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            # 只为主 stats 填充数据
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_kpts,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        """
        Return TPs for box, overall pose and per-keypoint metrics.
        """
        # Box IoU
        iou = box_iou(gt_bboxes, detections[:, :4])
        tp_box = self.match_predictions(detections[:, 5], gt_cls, iou)

        if pred_kpts is not None and gt_kpts is not None:
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53

            # 1. 计算总体的 OKS IoU 和 TP
            iou_overall = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
            tp_p_overall = self.match_predictions(detections[:, 5], gt_cls, iou_overall)

            tp_p_results = {"overall": tp_p_overall}

            # 2. 为每个关键点独立计算 OKS IoU 和 TP
            for k in range(self.kpt_shape[0]):
                iou_k = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area, kpt_idx=k)
                tp_p_k = self.match_predictions(detections[:, 5], gt_cls, iou_k)
                tp_p_results[k] = tp_p_k

            return tp_box, tp_p_results

        return tp_box, None

    def print_results(self):
        """Prints evaluation results for overall and each keypoint."""

        # 辅助函数，用于将统计数据列表合并成单一的Numpy数组
        def concat_stats(stats_dict):
            # 检查字典和其中的'conf'键是否存在且不为空
            if not stats_dict or not stats_dict.get('conf') or not any(len(x) for x in stats_dict['conf']):
                return None  # 如果没有统计数据，则返回None

            # 使用torch.cat合并列表中的所有张量，然后转移到CPU并转为numpy数组
            return {k: torch.cat(v, 0).cpu().numpy() for k, v in stats_dict.items() if isinstance(v, list) and v}

        # 1. 处理总体的Box和Pose指标
        stats_np = concat_stats(self.stats)
        if stats_np:
            self.metrics.process(stats_np['tp'], stats_np['tp_p'], stats_np['conf'], stats_np['pred_cls'],
                                 stats_np['target_cls'])

        ### --- 修正开始：为表头和数据行定义不同的格式 --- ###
        # 表头格式：全部是字符串
        pf_header = "%22s" + "%11s" * 10
        # 数据行格式：字符串 + 2个整数 + 8个浮点数
        pf_data = "%22s" + "%11i" * 2 + "%11.3g" * 8

        LOGGER.info(pf_header % (
        "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)", "Pose(P", "R", "mAP50", "mAP50-95)"))
        ### --- 修正结束 --- ###

        # 提取并打印总体结果
        box = self.metrics.box
        pose = self.metrics.pose
        total_instances = len(torch.cat(self.stats['target_cls'])) if self.stats['target_cls'] else 0
        LOGGER.info(pf_data % (
        "all", self.seen, total_instances, box.mp, box.mr, box.map50, box.map, pose.mp, pose.mr, pose.map50, pose.map))

        # 2. 接下来，独立处理并打印每个关键点的指标
        LOGGER.info("\n--- Per-Keypoint Pose Metrics ---")

        # VVVVVV  请根据您的模型自定义这里的关键点名称 VVVVVV
        keypoint_names = {0: "底座 (base)", 1: "转轴中心 (rotor_center)"}
        # ^^^^^^  请根据您的模型自定义这里的关键点名称 ^^^^^^

        for k in range(self.kpt_shape[0]):
            k_stats = self.per_kpt_stats[k]
            k_metrics = self.per_kpt_metrics[k]

            k_stats_np = concat_stats(k_stats)
            if k_stats_np:
                # 使用对应关键点的统计数据来计算指标
                k_metrics.process(k_stats_np['tp'], k_stats_np['tp_p'], k_stats_np['conf'], k_stats_np['pred_cls'],
                                  k_stats_np['target_cls'])

            # 准备打印
            kpt_name = keypoint_names.get(k, f"Keypoint_{k}")
            LOGGER.info(f"\n>> Metrics for {kpt_name}:")

            # 打印表头
            LOGGER.info(pf_header % (
            "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)", "Pose(P", "R", "mAP50", "mAP50-95)"))

            # 提取该关键点的 Box 和 Pose 指标
            k_box = k_metrics.box
            k_pose = k_metrics.pose
            k_total_instances = len(torch.cat(k_stats['target_cls'])) if k_stats['target_cls'] else 0
            LOGGER.info(pf_data % (
            kpt_name, self.seen, k_total_instances, k_box.mp, k_box.mr, k_box.map50, k_box.map, k_pose.mp, k_pose.mr,
            k_pose.map50, k_pose.map))

    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            kpts=batch["keypoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            kpts=pred_kpts,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            keypoints=pred_kpts,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Converts YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "keypoints": p[6:],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates object detection model using COCO JSON format."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

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
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
                                                                                       :2
                                                                                       ]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats