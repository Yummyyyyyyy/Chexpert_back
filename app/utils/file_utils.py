"""
文件处理工具
"""

from fastapi import UploadFile
from pathlib import Path
import aiofiles
from typing import Set

from app.config import Config


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


async def save_upload_file(
    upload_file: UploadFile,
    file_id: str,
    destination: Path
) -> Path:
    """
    异步保存上传的文件
    
    Args:
        upload_file: FastAPI上传文件对象
        file_id: 文件唯一标识
        destination: 目标目录
        
    Returns:
        保存的文件路径
    """
    # 获取原始文件扩展名
    ext = upload_file.filename.rsplit('.', 1)[1].lower()
    
    # 构建保存路径
    filename = f"original_{file_id}.{ext}"
    file_path = destination / filename
    
    # 异步写入文件
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path