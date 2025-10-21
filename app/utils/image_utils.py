"""
图像处理工具函数
包括图像验证、保存、预处理等
"""
from fastapi import UploadFile
from PIL import Image
import os
import uuid
from typing import Optional
from loguru import logger

from app.config import settings


def validate_image(filename: str) -> bool:
    """
    验证文件是否为允许的图像格式

    参数:
        filename: 文件名

    返回:
        是否为合法图像格式
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in settings.ALLOWED_EXTENSIONS


async def save_upload_file(
    upload_file: UploadFile,
    prefix: str = "image"
) -> str:
    """
    保存上传的文件

    参数:
        upload_file: FastAPI上传文件对象
        prefix: 文件名前缀

    返回:
        保存后的文件路径

    【TODO】后端团队成员可选优化:
    1. 添加图像压缩功能（减少存储空间）
    2. 添加图像格式转换（统一转为PNG或JPG）
    3. 添加DICOM文件支持（医学影像标准格式）
    """
    try:
        # 生成唯一文件名
        ext = os.path.splitext(upload_file.filename)[1].lower()
        unique_filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

        # 读取并保存文件
        contents = await upload_file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"💾 文件已保存: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"❌ 文件保存失败: {str(e)}")
        raise


def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> Image.Image:
    """
    图像预处理（用于模型输入）

    参数:
        image_path: 图片路径
        target_size: 目标尺寸

    返回:
        预处理后的PIL图像对象

    【TODO】后端团队成员根据模型需求实现:
    ```python
    from PIL import Image
    import numpy as np

    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 调整大小
    image = image.resize(target_size)

    # 归一化（根据模型训练时的预处理方式）
    image = np.array(image) / 255.0

    # 标准化（使用ImageNet均值和方差）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    return image
    ```
    """
    logger.warning("⚠️  图像预处理逻辑待实现 (image_utils.py:92)")

    # 简单实现：加载图像
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return image


def validate_image_quality(image_path: str) -> dict:
    """
    检查图像质量（分辨率、清晰度等）

    【TODO】后端团队成员可选实现:
    - 检查图像分辨率是否足够
    - 检测图像模糊度
    - 检查图像对比度
    """
    try:
        image = Image.open(image_path)
        width, height = image.size

        return {
            "valid": True,
            "width": width,
            "height": height,
            "format": image.format,
            "mode": image.mode
        }
    except Exception as e:
        logger.error(f"❌ 图像质量检查失败: {str(e)}")
        return {
            "valid": False,
            "error": str(e)
        }
