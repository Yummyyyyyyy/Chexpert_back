"""
病症标签API路由
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import uuid
from pathlib import Path

from app.services.pathology_service import PathologyService
from app.config import Config
from app.utils.file_utils import save_upload_file, allowed_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/pathology', tags=['病症标签'])

# 初始化服务
try:
    pathology_service = PathologyService()
except Exception as e:
    logger.error(f"Failed to initialize PathologyService: {e}")
    pathology_service = None


@router.post("/analyze")
async def analyze_pathology(
    file: UploadFile = File(...),
    top_n: Optional[int] = Query(None, description="返回Top N结果", ge=1, le=14),
    threshold: Optional[float] = Query(None, description="概率阈值", ge=0.0, le=1.0)
):
    """
    分析X光片的病症标签
    
    参数:
    - file: 上传的图像文件
    - top_n: 返回前N个病症（可选，1-14）
    - threshold: 概率阈值（可选，0.0-1.0）
    
    返回:
    - 病症标签分析结果
    """
    if pathology_service is None:
        raise HTTPException(
            status_code=503,
            detail="病症标签服务未初始化，请检查模型文件是否存在"
        )
    
    try:
        # 验证文件类型
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型。允许的类型: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            )
        
        # 保存上传文件
        file_id = str(uuid.uuid4())
        saved_path = await save_upload_file(file, file_id, Config.UPLOAD_FOLDER)
        
        logger.info(f"Processing pathology analysis for file: {saved_path}")
        
        # 执行病症分析
        result = pathology_service.analyze_image(
            image_path=str(saved_path),
            top_n=top_n,
            threshold=threshold
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', '分析失败'))
        
        # 添加文件信息
        result['file_info'] = {
            'file_id': file_id,
            'filename': file.filename,
            'file_path': str(saved_path.relative_to(Config.BASE_DIR))
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pathology analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/labels")
async def get_pathology_labels():
    """
    获取所有病症标签列表
    
    返回:
    - 14类病症标签及其中文翻译
    """
    from app.models.pathology_model import PathologyModel
    
    labels = []
    for idx, label in enumerate(PathologyModel.PATHOLOGY_LABELS):
        labels.append({
            'index': idx,
            'label': label,
            'label_cn': PathologyModel.LABEL_TRANSLATIONS.get(label, label)
        })
    
    return JSONResponse(content={
        'labels': labels,
        'total_count': len(labels)
    })


@router.post("/batch-analyze")
async def batch_analyze_pathology(
    files: list[UploadFile] = File(...),
    top_n: Optional[int] = Query(5, description="返回Top N结果")
):
    """
    批量分析多张X光片
    
    参数:
    - files: 上传的图像文件列表
    - top_n: 每张图返回前N个病症
    
    返回:
    - 批量分析结果
    """
    if pathology_service is None:
        raise HTTPException(status_code=503, detail="病症标签服务未初始化")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="单次最多上传10张图片")
    
    results = []
    
    for file in files:
        try:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': '不支持的文件类型'
                })
                continue
            
            file_id = str(uuid.uuid4())
            saved_path = await save_upload_file(file, file_id, Config.UPLOAD_FOLDER)
            
            result = pathology_service.analyze_image(
                image_path=str(saved_path),
                top_n=top_n
            )
            
            result['filename'] = file.filename
            result['file_id'] = file_id
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
    
    return JSONResponse(content={
        'batch_results': results,
        'total_files': len(files),
        'success_count': sum(1 for r in results if r.get('success', False))
    })