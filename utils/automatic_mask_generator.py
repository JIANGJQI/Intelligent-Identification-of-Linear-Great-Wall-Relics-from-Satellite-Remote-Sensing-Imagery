import json
import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


np.random.seed(3)


def show_anns(anns, borders=True, output_path=None):
    """
    显示分割注释的函数，将分割掩码以不同颜色可视化显示

    参数:
        anns: 分割注释列表，每个注释应包含"segmentation"和"area"字段
        borders: 布尔值，是否绘制分割区域的边界轮廓，默认为True
        output_path: 字符串，保存结果图像的路径，如果为None则不保存，默认为None

    返回值:
        无返回值
    """
    if len(anns) == 0:
        return
    # 按照面积从大到小排序分割注释
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # 创建一个RGBA图像用于显示分割掩码
    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    # 遍历所有分割注释，为每个区域分配随机颜色
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.3]])
        img[m] = color_mask
        # 如果需要绘制边界，则计算并绘制轮廓
        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # 对轮廓进行近似平滑处理
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.5), thickness=1)

            # 如果指定了输出路径，则保存结果图像
            if output_path is not None:
                cv2.imwrite(output_path, img * 255)
    ax.imshow(img)


def overlay_mask_on_image_and_save(image, anns, output_path=None, borders=True):
    """
    将分割掩码叠加到原始图像上并保存结果图像

    参数:
        image: 原始图像，numpy数组格式
        anns: 分割注释列表，每个注释包含"segmentation"和"area"字段
        output_path: 输出图像的保存路径，如果为None则不保存
        borders: 是否绘制掩码边界轮廓

    返回值:
        无返回值
    """
    if len(anns) == 0:
        return

    # 按面积从大到小排序注释，确保大面积的掩码先绘制
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    img = np.zeros_like(image)

    # 遍历所有注释，为每个分割区域绘制掩码和边界
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate(
            [np.random.random(3), [0.5]]
        )  # 生成随机颜色，alpha透明度设为0.5
        img[m] = color_mask[:3] * 255  # 应用RGB值，不使用alpha通道

        if borders:
            # 查找并绘制轮廓边界
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # 对轮廓进行近似平滑处理
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=1)

    # 将掩码叠加到原始图像上
    overlayed_img = cv2.addWeighted(image, 1, img, 0.5, 0)

    # 保存结果图像
    if output_path is not None:
        cv2.imwrite(output_path, overlayed_img)


def save_mask_as_json(masks, output_dir, image_name):
    """
    将掩码数据保存为COCO RLE格式的JSON文件

    参数:
        masks: 包含掩码信息的列表，每个元素是一个字典，包含分割信息、面积、边界框等
        output_dir: 输出目录路径
        image_name: 原始图像文件名

    返回值:
        无返回值
    """
    # 创建输出目录
    # output_dir = "output_masks_coco"
    os.makedirs(output_dir, exist_ok=True)

    # 准备 JSON 数据
    json_data = []

    for i, mask_info in enumerate(masks):
        # 提取 RLE 编码
        rle = mask_info["segmentation"]

        # 将 RLE 转换为二值掩码（可选：用于调试或可视化）
        # binary_mask = mask_utils.decode(rle)

        # 构造要保存的数据结构
        mask_entry = {
            "id": i,
            "area": int(mask_info["area"]),
            "bbox": [int(x) for x in mask_info["bbox"]],  # 转换为整数列表
            "point_coords": mask_info["point_coords"],
            "predicted_iou": float(mask_info["predicted_iou"]),
            "stability_score": float(mask_info["stability_score"]),
            "rle": {"size": rle["size"], "counts": rle["counts"]},
        }

        json_data.append(mask_entry)

    # 获取文件名除了后缀
    image_name = os.path.splitext(image_name)[0]
    # 保存为 JSON 文件
    output_json_path = os.path.join(output_dir, f"{image_name}.json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"掩码已保存为 COCO RLE 格式的 JSON 文件：{output_json_path}")
