"""
自定义异常处理
"""


class ModelLoadError(Exception):
    """模型加载失败异常"""
    pass


class ImageProcessingError(Exception):
    """图像处理失败异常"""
    pass


class APICallError(Exception):
    """第三方API调用失败异常"""
    pass


class InvalidImageError(Exception):
    """无效图像异常"""
    pass
