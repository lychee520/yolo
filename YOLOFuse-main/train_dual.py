from ultralytics import YOLO


#############多模态训练脚本#############
if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v11/yolo11-elff-RCB-pose.yaml")
    model.train(
        data="ultralytics/cfg/datasets/lll-pose.yaml",
        ch=6, # 多模态时设置为 6 ，单模态时设置为 3
        imgsz=640,
        epochs=2,
        batch=32,
        close_mosaic=0,
        workers=0,
        device="0",
        optimizer="SGD",
        patience=0,
        amp=False,
        cache=True, # disk 硬盘，速度稍快精度可复现；ram/True 内存，速度快但精度不可复现
        project="runs/多模态pose",
        name="test",   #每次修改name
        resume=False,
        fraction=1, # 只用全部数据的 ？% 进行训练 (0.1-1)
    )
# from ultralytics import YOLO
#
# # 加载已保存的部分训练模型
# model = YOLO('runs/Easy-level-Fusion/Easy-fuse/weights/last.pt') # 使用你的模型路径
#
# # 继续训练
# results = model.train(resume=True)