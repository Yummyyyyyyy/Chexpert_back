# 后端团队开发指南

> 给团队成员的详细开发说明

---

## 项目现状

框架已经搭建完成，所有接口都能正常运行并返回**模拟数据**。现在需要团队成员**替换模拟逻辑为真实实现**。

---

## 需要修改的文件清单

### 🔴 核心文件（必须实现）

#### 1. `app/models/model_manager.py`

**任务**: 实现模型加载逻辑

**需要修改的函数**:
- `load_classification_model()`: 加载疾病分类模型
- `load_llava_model()`: 加载LLaVA多模态模型

**当前状态**: 返回占位符字符串

**修改示例**:
```python
async def load_classification_model(self):
    import torch
    from your_model import CheXpertClassifier

    model_path = os.path.join(settings.MODEL_BASE_DIR, "classification_model.pth")
    self.classification_model = CheXpertClassifier()
    self.classification_model.load_state_dict(torch.load(model_path))
    self.classification_model.to(self.device)
    self.classification_model.eval()
    logger.success("✅ 分类模型加载成功")
```

**搜索关键词**: `⚠️  分类模型加载逻辑待实现`

---

#### 2. `app/models/heatmap_generator.py`

**任务**: 实现热力图生成和疾病分类

**需要修改的函数**:
- `generate()`: 生成热力图并返回分类结果

**当前状态**: 返回模拟热力图路径和模拟分类结果

**修改要点**:
1. 使用Grad-CAM或类似算法生成热力图
2. 调用分类模型获取真实分类结果
3. 保存热力图到 `uploads/` 目录
4. 返回真实数据

**参考库**: `pytorch-grad-cam`, `opencv-python`

**搜索关键词**: `⚠️  热力图生成逻辑待实现`

---

#### 3. `app/services/llava_service.py`

**任务**: 实现LLaVA模型推理，生成医学报告

**需要修改的函数**:
- `generate_report()`: 调用LLaVA模型生成报告

**当前状态**: 返回模拟的医学报告

**修改示例**:
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

# 加载图像
image = Image.open(image_path)

# 准备输入
model = model_manager.get_llava_model()
processor = AutoProcessor.from_pretrained(settings.LLAVA_MODEL_PATH)

full_prompt = self.prompt_template.format(user_prompt=prompt)
inputs = processor(text=full_prompt, images=image, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

report = processor.decode(outputs[0], skip_special_tokens=True)
```

**搜索关键词**: `⚠️  LLaVA报告生成逻辑待实现`

---

#### 4. `app/services/third_party_service.py`

**任务**: 调用第三方API获取知识图谱

**需要修改的函数**:
- `query_disease()`: 调用真实的第三方API

**当前状态**: 返回模拟的知识图谱数据

**修改步骤**:
1. 在 `app/config.py` 中配置 `THIRD_PARTY_API_URL` 和 `THIRD_PARTY_API_KEY`
2. 实现HTTP请求逻辑
3. 处理API响应并转换为标准格式
4. 添加错误处理和重试机制

**修改示例**:
```python
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{self.api_url}/knowledge_graph",
        headers={"Authorization": f"Bearer {self.api_key}"},
        json={"disease": disease_name, "language": language},
        timeout=self.timeout
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API调用失败: {response.status_code}")
```

**搜索关键词**: `⚠️  第三方API调用逻辑待实现`

---

### 🟡 配置文件（必须填写）

#### 5. `app/config.py`

**任务**: 配置实际的参数

**需要修改的配置项**:
```python
# 前端地址（根据实际情况修改）
CORS_ORIGINS: list = [
    "http://localhost:3000",
    "http://your-frontend-domain.com"  # 添加生产环境地址
]

# 模型路径（填写真实路径）
LLAVA_MODEL_PATH: str = "/path/to/your/llava/model"
ADAPTER_PATH: str = "/path/to/your/adapter"

# 第三方API配置（填写真实密钥）
THIRD_PARTY_API_URL: str = "https://api.example.com"
THIRD_PARTY_API_KEY: str = "your_secret_key"
```

**或者**: 在 `.env` 文件中配置（推荐）

---

#### 6. `.env` 文件

**任务**: 复制 `.env.example` 为 `.env`，并填写真实配置

```bash
cp .env.example .env
```

然后编辑 `.env`:
```env
# 模型路径
LLAVA_MODEL_PATH=/path/to/your/llava/model
ADAPTER_PATH=/path/to/your/adapter
DEVICE=cuda  # 或 cpu / mps

# 第三方API
THIRD_PARTY_API_URL=https://api.example.com
THIRD_PARTY_API_KEY=your_secret_key_here

# CORS配置
# CORS_ORIGINS=["http://localhost:3000", "http://your-frontend.com"]
```

---

### 🟢 可选优化（不影响基础功能）

#### 7. `app/utils/image_utils.py`

**可选优化**:
- `preprocess_image()`: 根据模型需求实现图像预处理
- `validate_image_quality()`: 添加图像质量检查

---

## 开发流程建议

### 阶段1: 环境搭建（第1天）

1. 克隆代码
2. 创建虚拟环境: `python -m venv venv`
3. 安装依赖: `pip install -r requirements.txt`
4. 启动服务验证框架: `python app/main.py`
5. 访问 http://localhost:8000/docs 查看API文档

### 阶段2: 模型集成（第2-3天）

1. 准备模型文件
   - 将模型放到 `app/models/base_models/` (此目录已在.gitignore中)
   - 或配置模型路径到 `.env`

2. 修改 `model_manager.py`
   - 实现模型加载逻辑
   - 测试模型是否成功加载

3. 修改 `heatmap_generator.py`
   - 实现Grad-CAM
   - 测试热力图生成

4. 修改 `llava_service.py`
   - 实现LLaVA推理
   - 测试报告生成

### 阶段3: API集成（第4天）

1. 获取第三方API密钥
2. 在 `.env` 中配置API信息
3. 修改 `third_party_service.py`
4. 测试API调用

### 阶段4: 联调测试（第5天）

1. 与前端团队联调
2. 测试完整流程：上传图片 → 查看热力图 → 生成报告 → 查询知识图谱
3. 修复bug

---

## 如何快速定位待修改代码

所有待实现的代码都标记了 `【TODO】` 和 `⚠️` 警告。

**方法1**: 全局搜索
```bash
# 搜索所有TODO标记
grep -r "TODO" app/

# 搜索所有警告标记
grep -r "⚠️" app/
```

**方法2**: 查看日志

运行服务后，所有未实现的功能都会打印警告日志：
```
⚠️  分类模型加载逻辑待实现 (model_manager.py:43)
⚠️  热力图生成逻辑待实现 (heatmap_generator.py:60)
⚠️  LLaVA报告生成逻辑待实现 (llava_service.py:71)
⚠️  第三方API调用逻辑待实现 (third_party_service.py:65)
```

---

## 测试指南

### 1. 测试健康检查
```bash
curl http://localhost:8000/health
```

### 2. 测试图像上传
```bash
curl -X POST "http://localhost:8000/api/v1/image/analyze" \
  -F "file=@test_image.jpg"
```

### 3. 测试报告生成
```bash
curl -X POST "http://localhost:8000/api/v1/report/generate" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/uploads/original_xxx.jpg", "prompt": "分析这张X光片"}'
```

### 4. 测试知识图谱
```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '{"disease_name": "Cardiomegaly", "language": "zh"}'
```

---

## 常见问题

### Q1: 模型文件太大，能否纳入Git？

**不能**！模型文件已被 `.gitignore` 排除。

**解决方案**:
1. 将模型存储在团队共享盘或云存储
2. 在 `.env` 中配置模型路径
3. 或使用Hugging Face模型，运行时自动下载

### Q2: 如何调试模型推理？

在对应的服务文件中添加日志：
```python
logger.debug(f"模型输入: {inputs}")
logger.debug(f"模型输出: {outputs}")
```

### Q3: 前端联调时遇到CORS错误？

确保 `app/config.py` 中的 `CORS_ORIGINS` 包含前端地址。

### Q4: 如何添加新的API接口？

1. 在 `app/api/v1/endpoints/` 创建新文件
2. 定义路由和处理函数
3. 在 `app/api/v1/router.py` 中注册路由

---

## Git提交建议

```bash
# 提交模型集成
git add app/models/
git commit -m "feat: 实现模型加载和热力图生成"

# 提交服务层实现
git add app/services/
git commit -m "feat: 实现LLaVA推理和第三方API调用"

# 提交配置修改
git add app/config.py .env.example
git commit -m "chore: 更新配置文件"
```

**注意**: 不要提交 `.env` 文件（包含密钥），已被 `.gitignore` 排除。

---

## 联系方式

如有问题，请联系：
- 项目负责人: [联系方式]
- 技术讨论群: [群号/链接]

---

**祝开发顺利！有问题随时沟通。**
