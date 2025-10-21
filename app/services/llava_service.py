"""
LLaVA模型服务 - 通过 Colab API 远程调用
调用部署在 Colab 上的 LLaVA 模型生成医学报告
"""
from loguru import logger
from typing import Tuple, Optional
import time
import httpx
import base64
import os
from datetime import datetime
from pathlib import Path

from app.config import settings


class LLaVAService:
    """LLaVA模型调用服务 - 远程 API 版本"""

    def __init__(self):
        """初始化服务"""
        self.colab_api_url = settings.COLAB_API_URL
        self.api_timeout = settings.API_TIMEOUT

    async def generate_report(
        self,
        image_path: str,
        prompt: str = "",
        pathology_labels: list = None
    ) -> Tuple[str, float]:
        """
        生成医学报告 - 通过 Colab API

        参数:
            image_path: 图片路径
            prompt: 用户自定义提示词 (可选)
            pathology_labels: 病症标签列表 (可选,推荐使用Top3标签)

        返回:
            (report, processing_time): 报告文本和处理时间
        """
        # 去除路径前导斜杠
        image_path = image_path.lstrip('/')

        # 检查 Colab API 是否配置
        if not self.colab_api_url:
            raise ValueError("Colab API URL 未配置!")

        try:
            logger.info(f"📝 开始调用 Colab API: {image_path}")
            start_time = time.time()

            # 读取图片并转为 base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')

            # 准备请求数据
            if not prompt:
                # 构建默认prompt,如果有病症标签则包含进去
                pathology_context = ""
                if pathology_labels and len(pathology_labels) > 0:
                    labels_str = ", ".join(pathology_labels[:3])  # 只使用前3个
                    pathology_context = f"\nDetected pathologies: {labels_str}\n"

                default_prompt = f"""You are an experienced radiologist. Analyze this chest X-ray image and generate a diagnostic report with EXACTLY these three sections:{pathology_context}

FINDINGS: [Only mention clinically significant findings relevant to diagnosis - abnormalities, lesions, or pathological changes. Skip normal anatomical descriptions unless diagnostically relevant]

IMPRESSION: [List key diagnoses as numbered items: 1) diagnosis one 2) diagnosis two]

SUMMARY: [Brief clinical summary and recommendations]

STRICT REQUIREMENTS:
- Use ONLY plain text - NO markdown formatting, NO asterisks, NO bold, NO italics, NO special symbols
- Use ONLY these three section headers: FINDINGS, IMPRESSION, SUMMARY
- In FINDINGS: Be concise, only describe abnormalities or diagnostically relevant observations
- Do NOT include: COMPARISON, TECHNIQUE, HISTORY, CLINICAL INDICATION, or any other sections
- Do NOT compare with previous studies
- Start directly with "FINDINGS:" without any preamble
- Use simple numbered lists with parentheses: 1) 2) 3)
- Output clean medical text only"""

                request_data = {
                    "image": image_base64,
                    "prompt": default_prompt
                }
            else:
                request_data = {
                    "image": image_base64,
                    "prompt": prompt
                }

            # 调用 Colab API
            logger.info(f"🌐 调用 Colab API: {self.colab_api_url}")
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.post(self.colab_api_url, json=request_data)

            # 检查响应
            response.raise_for_status()  # 自动处理非 200 状态码

            # 解析响应
            result = response.json()
            raw_report = result.get("report", "")

            # 提取 assistant 后面的内容并格式化
            report = self._format_report(raw_report)

            processing_time = time.time() - start_time
            logger.success(f"✅ 报告生成完成! 耗时: {processing_time:.2f}秒")

            # 保存报告到本地存档
            report_path = self._save_report(report, image_path, pathology_labels)
            if report_path:
                logger.info(f"📁 报告已保存: {report_path}")

            return report, processing_time

        except httpx.TimeoutException:
            logger.error(f"❌ API 调用超时 (>{self.api_timeout}秒)")
            raise RuntimeError("Colab API 调用超时")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ API 返回错误: {e.response.status_code}")
            raise RuntimeError(f"Colab API 调用失败: {e.response.status_code}")
        except httpx.ConnectError:
            logger.error(f"❌ 无法连接到 Colab API")
            raise RuntimeError("无法连接到 Colab API, 请检查 Colab 是否在运行")
        except FileNotFoundError:
            logger.error(f"❌ 图片文件不存在: {image_path}")
            raise
        except Exception as e:
            logger.error(f"❌ 报告生成失败: {str(e)}")
            raise

    def _format_report(self, raw_report: str) -> str:
        """
        格式化报告：提取 assistant 后的内容，并每句话一行

        参数:
            raw_report: 原始报告文本

        返回:
            格式化后的报告
        """
        import re

        # 提取 assistant 后面的内容
        # 先按行分割，找到 assistant 所在行
        lines = raw_report.split('\n')
        assistant_index = -1

        for i, line in enumerate(lines):
            if line.strip() == 'assistant':
                assistant_index = i
                break

        if assistant_index >= 0 and assistant_index + 1 < len(lines):
            # 提取 assistant 之后的所有行
            report_lines = lines[assistant_index + 1:]
            report = ' '.join(line.strip() for line in report_lines if line.strip())
        elif "assistant" in raw_report:
            # 备用方案：直接分割
            report = raw_report.split("assistant", 1)[1].strip()
        else:
            report = raw_report.strip()

        # 将编号项 1. 2. 3. 改成 1) 2) 3)
        report = re.sub(r'(\d+)\.', r'\1)', report)

        # 按句号、问号、感叹号分割
        sentences = re.split(r'(?<=[.!?。！？])\s*', report)

        # 过滤空行
        formatted_lines = [s.strip() for s in sentences if s.strip()]

        return '\n'.join(formatted_lines)

    def _save_report(
        self,
        report_text: str,
        image_path: str,
        pathology_labels: Optional[list] = None
    ) -> Optional[str]:
        """
        保存报告到本地文件

        参数:
            report_text: 报告内容
            image_path: 图像路径
            pathology_labels: 病症标签列表

        返回:
            保存的文件路径，失败返回None
        """
        try:
            # 创建reports目录
            reports_dir = Path("reports/llava")
            reports_dir.mkdir(parents=True, exist_ok=True)

            # 生成文件名：时间戳 + 图像文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = os.path.basename(image_path).split('.')[0]
            report_filename = f"{timestamp}_{image_filename}.txt"
            report_path = reports_dir / report_filename

            # 构建报告内容（包含元数据）
            full_content = []
            full_content.append("=" * 80)
            full_content.append("LLaVA Medical Report")
            full_content.append("=" * 80)
            full_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            full_content.append(f"Image: {image_path}")

            if pathology_labels and len(pathology_labels) > 0:
                full_content.append(f"Detected Pathologies: {', '.join(pathology_labels)}")

            full_content.append("=" * 80)
            full_content.append("")
            full_content.append(report_text)
            full_content.append("")
            full_content.append("=" * 80)

            # 写入文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(full_content))

            return str(report_path)

        except Exception as e:
            logger.error(f"❌ 保存报告失败: {str(e)}")
            return None


# 全局单例
_llava_service_instance = None


def get_llava_service() -> LLaVAService:
    """获取 LLaVA 服务单例"""
    global _llava_service_instance
    if _llava_service_instance is None:
        _llava_service_instance = LLaVAService()
    return _llava_service_instance
