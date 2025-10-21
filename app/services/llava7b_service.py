"""
LLaVA-7B模型服务 - 通过 ngrok API 远程调用
参考 model_llava/deploy/test_ngork.py 的调用方式
"""
from loguru import logger
from typing import Tuple, Optional
import time
import requests
from PIL import Image
import io
import re

from app.config import settings


class LLaVA7BService:
    """LLaVA-7B模型调用服务 - 参考test_ngork.py实现"""

    def __init__(self):
        """初始化服务"""
        self.api_url = settings.LLAVA_7B_API_URL
        self.api_timeout = settings.API_TIMEOUT

    async def generate_report(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        support_info: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        生成医学报告 - 通过 LLAVA-7B API

        参数:
            image_path: 图片路径
            prompt: 用户自定义提示词 (可选)
            support_info: 分类结果支持信息 (可选,预留给未来/api/v1/image/analyze接口)

        返回:
            (report, processing_time): 报告文本和处理时间
        """
        # 去除路径前导斜杠
        image_path = image_path.lstrip('/')

        # 检查 API 是否配置
        if not self.api_url:
            raise ValueError("LLAVA-7B API URL 未配置!")

        try:
            logger.info(f"📝 开始调用 LLAVA-7B API: {image_path}")
            start_time = time.time()

            # 1. 读取图片并转换为PIL对象
            image = Image.open(image_path).convert("RGB")

            # 2. 将图片转为JPEG字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=90)
            img_byte_arr = img_byte_arr.getvalue()

            # 3. 构建prompt (参考prompt.txt)
            # 注意: support_info当前为空,但保留占位符供将来使用
            if not prompt:
                # 使用默认的prompt模板,参考 model_llava/deploy/prompt.txt
                # 当support_info未来实现时,将被插入到clinical context中
                if support_info:
                    clinical_context = f"Based on the chest X-ray image and clinical context: {support_info}"
                else:
                    # 当前/api/v1/image/analyze接口未实现,忽略support_info
                    clinical_context = "Based on the chest X-ray image"

                question = (
                    f"<image>\nYou are an experienced radiologist. {clinical_context}, generate a report with 3 sections. "
                    f"Prioritize accuracy—only describe what is clearly observed; if no relevant findings exist, state 'No evidence of...' or 'Unremarkable'.\n\n"
                    f"1. FINDINGS: Objectively describe all observed details, including:\n"
                    f"   - If present: Implants/devices (e.g., central lines, drains, stents, surgical clips)\n"
                    f"   - Abnormalities (e.g., edema, effusions, atelectasis) and normal findings, using standard radiological terminology.\n\n"
                    f"2. IMPRESSION: Concisely summarize key diagnostic conclusions based on FINDINGS, highlighting the most clinically significant abnormalities (e.g., 'Pulmonary vascular congestion without frank edema').\n\n"
                    f"3. SUMMARY: Note significance of findings. If no actionable issues, state 'No findings requiring immediate action'."
                )
            else:
                question = f"<image>\n{prompt}"

            prompt_text = f"User: {question}\nAssistant:"

            # 4. 准备multipart/form-data请求 (参考test_ngork.py)
            files = {
                "image": ("image.jpg", img_byte_arr, "image/jpeg")
            }
            data = {
                "question": prompt_text
            }

            # 5. 调用API
            logger.info(f"🌐 调用 LLAVA-7B API: {self.api_url}")
            response = requests.post(
                self.api_url,
                files=files,
                data=data,
                timeout=self.api_timeout
            )

            # 6. 检查响应状态
            logger.info(f"状态码: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"API返回错误: {response.text}")
                raise RuntimeError(f"LLAVA-7B API 调用失败: {response.status_code}")

            # 7. 解析响应
            try:
                result = response.json()
                raw_report = result.get("response", "")
            except Exception as e:
                logger.error(f"解析JSON失败: {response.text}")
                raise RuntimeError(f"无法解析API响应: {str(e)}")

            # 8. 清理报告 - 去除重复句子
            cleaned_report = self._remove_duplicate_sentences(raw_report)

            # 去除prompt残留
            if 'Assistant:' in cleaned_report:
                cleaned_report = cleaned_report.split('Assistant:')[-1].strip()

            processing_time = time.time() - start_time
            logger.success(f"✅ LLAVA-7B 报告生成完成! 耗时: {processing_time:.2f}秒")

            return cleaned_report, processing_time

        except requests.exceptions.Timeout:
            logger.error(f"❌ API 调用超时 (>{self.api_timeout}秒)")
            raise RuntimeError("LLAVA-7B API 调用超时")
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ 无法连接到 LLAVA-7B API")
            raise RuntimeError("无法连接到 LLAVA-7B API, 请检查服务是否在运行")
        except FileNotFoundError:
            logger.error(f"❌ 图片文件不存在: {image_path}")
            raise
        except Exception as e:
            logger.error(f"❌ LLAVA-7B 报告生成失败: {str(e)}")
            raise

    def _remove_duplicate_sentences(self, text: str) -> str:
        """
        去除重复出现的句子
        有时候模型会重复输出同一句话,需要清理

        参数:
            text: 原始文本

        返回:
            清理后的文本
        """
        if not text:
            return text

        # 按句子分割 (使用句号、问号、感叹号)
        sentences = re.split(r'(?<=[.!?。!?])\s+', text.strip())

        # 去重逻辑: 检测连续重复的句子
        cleaned_sentences = []
        previous_sentence = None
        repeat_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 检查是否与前一句相同
            if sentence == previous_sentence:
                repeat_count += 1
                # 如果连续重复超过1次,跳过这句
                if repeat_count > 1:
                    continue
            else:
                repeat_count = 0
                previous_sentence = sentence

            cleaned_sentences.append(sentence)

        # 重新组合
        result = ' '.join(cleaned_sentences)

        # 记录清理信息
        original_count = len(sentences)
        cleaned_count = len(cleaned_sentences)
        if original_count > cleaned_count:
            logger.info(f"🧹 清理重复句子: {original_count} -> {cleaned_count}")

        return result


# 全局单例
_llava7b_service_instance = None


def get_llava7b_service() -> LLaVA7BService:
    """获取 LLaVA-7B 服务单例"""
    global _llava7b_service_instance
    if _llava7b_service_instance is None:
        _llava7b_service_instance = LLaVA7BService()
    return _llava7b_service_instance
