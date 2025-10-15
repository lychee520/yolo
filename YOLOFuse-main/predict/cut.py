import os
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm


def slice_images(config, output_paths):
    # 配置参数
    LARGE_IMAGE_RGB_PATH = config["LARGE_IMAGE_RGB_PATH"]
    LARGE_IMAGE_IR_PATH = config["LARGE_IMAGE_IR_PATH"]
    BASE_OUTPUT_DIR = config["BASE_OUTPUT_DIR"]
    TILE_SIZE = config["TILE_SIZE"]
    OVERLAP_RATIO = config["OVERLAP_RATIO"]

    # 创建 LLVIP 目录，并确保 images 和 imagesIR 文件夹存在
    LLVIP_DIR = os.path.join(BASE_OUTPUT_DIR, "LLVIP")
    OUTPUT_DIR_RGB = os.path.join(LLVIP_DIR, "images")  # 保存 RGB 切片的文件夹
    OUTPUT_DIR_IR = os.path.join(LLVIP_DIR, "imagesIR")  # 保存 IR 切片的文件夹

    os.makedirs(OUTPUT_DIR_RGB, exist_ok=True)
    os.makedirs(OUTPUT_DIR_IR, exist_ok=True)

    # 计算步长，确保20%的重叠
    step_size = int(TILE_SIZE * (1 - OVERLAP_RATIO))

    try:
        with rasterio.open(LARGE_IMAGE_RGB_PATH) as src_rgb, rasterio.open(LARGE_IMAGE_IR_PATH) as src_ir:
            assert src_rgb.height == src_ir.height and src_rgb.width == src_ir.width, "Error: Images must have the same size!"
            height, width = src_rgb.height, src_rgb.width

            num_tiles = len(range(0, height, step_size)) * len(range(0, width, step_size))
            with tqdm(total=num_tiles, desc="Slicing images") as pbar:
                for y in range(0, height, step_size):
                    for x in range(0, width, step_size):
                        row_idx = y // step_size
                        col_idx = x // step_size
                        window = Window(x, y, TILE_SIZE, TILE_SIZE)

                        transform_rgb = rasterio.windows.transform(window, src_rgb.transform)
                        transform_ir = rasterio.windows.transform(window, src_ir.transform)

                        meta_rgb = src_rgb.meta.copy()
                        meta_ir = src_ir.meta.copy()

                        meta_rgb.update({"height": TILE_SIZE, "width": TILE_SIZE, "transform": transform_rgb})
                        meta_ir.update({"height": TILE_SIZE, "width": TILE_SIZE, "transform": transform_ir})

                        filename = f"tile_{row_idx}_{col_idx}.tif"
                        rgb_filepath = os.path.join(OUTPUT_DIR_RGB, filename)
                        ir_filepath = os.path.join(OUTPUT_DIR_IR, filename)

                        data_rgb = src_rgb.read(window=window)
                        data_ir = src_ir.read(window=window)

                        padded_rgb = np.zeros((src_rgb.count, TILE_SIZE, TILE_SIZE), dtype=src_rgb.dtypes[0])
                        padded_ir = np.zeros((src_ir.count, TILE_SIZE, TILE_SIZE), dtype=src_ir.dtypes[0])

                        padded_rgb[:, :data_rgb.shape[1], :data_rgb.shape[2]] = data_rgb
                        padded_ir[:, :data_ir.shape[1], :data_ir.shape[2]] = data_ir

                        with rasterio.open(rgb_filepath, 'w', **meta_rgb) as dst_rgb:
                            dst_rgb.write(padded_rgb)

                        with rasterio.open(ir_filepath, 'w', **meta_ir) as dst_ir:
                            dst_ir.write(padded_ir)

                        pbar.update(1)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
