# CheXpert Backend API

> 基于FastAPI的医学影像分析后端服务，提供X光片分析、报告生成和知识图谱查询功能

---

## 项目概述

本项目为 CheXpert 前端应用提供后端API服务，主要功能包括：

1. **图像分析**: 上传X光片，返回疾病分类结果和热力图
2. **报告生成**: 基于自研LLaVA多模态模型生成医学报告
3. **知识图谱**: 调用第三方API获取疾病知识图谱

---

## 技术栈

- **框架**: FastAPI 0.104+
- **深度学习**: PyTorch 2.1+ (待配置)
- **图像处理**: Pillow, OpenCV
- **HTTP客户端**: httpx
- **日志**: loguru
- **文档**: 自动生成Swagger UI

---

## 项目结构

```
CheXpert_back/
├── app/
│   ├── main.py                 # 应用入口
│   ├── config.py               # 配置文件
│   ├── api/v1/                 # API路由
│   │   ├── endpoints/
│   │   │   ├── upload.py       # 图像上传与分析
│   │   │   ├── llava_report.py # 报告生成
│   │   │   └── third_party.py  # 知识图谱查询
│   │   └── router.py           # 路由汇总
│   ├── models/                 # 模型管理
│   │   ├── model_manager.py    # 模型加载器
│   │   └── heatmap_generator.py # 热力图生成
│   ├── services/               # 业务逻辑
│   │   ├── llava_service.py    # LLaVA模型服务
│   │   └── third_party_service.py # 第三方API服务
│   └── utils/                  # 工具函数
│       ├── image_utils.py      # 图像处理
│       └── exception_handler.py # 异常处理
├── requirements.txt            # Python依赖
├── .env.example                # 环境变量示例
└── README.md                   # 本文件
```

---

## 快速开始

### 1. 环境准备

```bash
# Python版本要求: 3.9+
python --version

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填写实际配置
# 特别注意：
# - CORS_ORIGINS: 前端地址
# - LLAVA_MODEL_PATH: 模型路径
# - THIRD_PARTY_API_KEY: 第三方API密钥
```

### 3. 启动服务

```bash
# 方式1: 使用uvicorn命令
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 方式2: 直接运行main.py
python -m app.main

# 方式3: 使用Python直接运行
python app/main.py
```

### 4. 访问API文档

启动成功后，访问以下地址：

- **Swagger UI文档**: http://localhost:8000/docs
- **ReDoc文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/health

---

## API接口说明

### 基础接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 欢迎页面 |
| `/health` | GET | 健康检查 |

### 功能接口（/api/v1）

#### 1. 图像分析

- **接口**: `POST /api/v1/image/analyze`
- **功能**: 上传X光片，返回分类结果和热力图
- **请求**: `multipart/form-data`, 字段名 `file`
- **响应**:
  ```json
  {
    "success": true,
    "message": "图像分析完成",
    "original_image_url": "/uploads/original_xxx.jpg",
    "heatmap_image_url": "/uploads/heatmap_xxx.jpg",
    "classifications": [
      {
        "label": "Cardiomegaly",
        "confidence": 0.87,
        "description": "心脏肥大"
      }
    ]
  }
  ```

#### 2. 报告生成

- **接口**: `POST /api/v1/report/generate`
- **功能**: 基于图片生成医学报告
- **请求**:
  ```json
  {
    "image_path": "/uploads/original_xxx.jpg",
    "prompt": "请分析这张X光片"
  }
  ```
- **响应**:
  ```json
  {
    "success": true,
    "report": "详细的医学报告...",
    "processing_time": 1.5
  }
  ```

#### 3. 知识图谱查询

- **接口**: `POST /api/v1/knowledge/query`
- **功能**: 查询疾病知识图谱
- **请求**:
  ```json
  {
    "disease_name": "Cardiomegaly",
    "language": "zh"
  }
  ```
- **响应**:
  ```json
  {
    "success": true,
    "disease_name": "Cardiomegaly",
    "knowledge_graph": {
      "description": "...",
      "symptoms": [...],
      "treatments": [...]
    }
  }
  ```

---

## 前后端协作指南

### CORS配置

后端已配置CORS中间件，前端可以直接调用API。如果前端地址变更，需要在 `app/config.py` 中更新 `CORS_ORIGINS`。

### 前端调用示例

```javascript
// 1. 图像上传
const formData = new FormData();
formData.append('file', imageFile);

const uploadResponse = await fetch('http://localhost:8000/api/v1/image/analyze', {
  method: 'POST',
  body: formData
});
const result = await uploadResponse.json();

// 2. 生成报告
const reportResponse = await fetch('http://localhost:8000/api/v1/report/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image_path: result.original_image_url,
    prompt: '请详细分析'
  })
});

// 3. 查询知识图谱
const knowledgeResponse = await fetch('http://localhost:8000/api/v1/knowledge/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    disease_name: 'Cardiomegaly',
    language: 'zh'
  })
});
```

---

## 后端团队待实现功能

当前框架已搭建完成，以下功能需要后端团队成员实现：

### 🔴 高优先级（核心功能）

1. **模型集成** (`app/models/model_manager.py`)
   - [ ] 加载分类模型
   - [ ] 加载LLaVA模型
   - [ ] 实现GPU/CPU自动切换

2. **热力图生成** (`app/models/heatmap_generator.py`)
   - [ ] 实现Grad-CAM算法
   - [ ] 整合分类推理逻辑

3. **LLaVA推理** (`app/services/llava_service.py`)
   - [ ] 实现模型推理
   - [ ] 优化提示词模板

4. **第三方API** (`app/services/third_party_service.py`)
   - [ ] 配置API密钥
   - [ ] 实现API调用逻辑

### 🟡 中优先级（性能优化）

5. **图像预处理** (`app/utils/image_utils.py`)
   - [ ] 根据模型需求实现预处理
   - [ ] 添加DICOM格式支持

6. **异常处理**
   - [ ] 完善错误提示
   - [ ] 添加重试机制

### 🟢 低优先级（可选优化）

7. **性能优化**
   - [ ] 添加模型推理缓存
   - [ ] 批量推理支持
   - [ ] 异步处理优化

8. **日志监控**
   - [ ] 集成Sentry等监控工具
   - [ ] 添加性能指标统计

---

## 部署说明

### 开发环境

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 生产环境

```bash
# 1. 修改配置
# 在 .env 中设置 DEBUG=False

# 2. 使用Gunicorn部署
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# 3. 或使用Docker
# 【TODO】后续添加Dockerfile
```

---

## 常见问题

### Q1: 如何测试API是否正常？

访问 http://localhost:8000/health，应返回：
```json
{"status": "healthy", "app_name": "CheXpert Backend API"}
```

### Q2: 前端无法调用API（CORS错误）？

检查 `app/config.py` 中的 `CORS_ORIGINS`，确保包含前端地址。

### Q3: 模型加载失败？

1. 检查 `.env` 中的 `LLAVA_MODEL_PATH` 是否正确
2. 确认模型文件存在
3. 检查CUDA/GPU配置

### Q4: 图片上传失败？

1. 确认 `uploads/` 目录存在且有写权限
2. 检查文件大小是否超过限制（默认10MB）
3. 确认文件格式是否支持

---

## 贡献指南

1. 克隆仓库
2. 创建功能分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -m 'Add some feature'`
4. 推送分支: `git push origin feature/your-feature`
5. 创建Pull Request

---

## 许可证

本项目仅供学习交流使用

---

## 联系方式

如有问题，请联系项目负责人或在仓库中提Issue。

**祝开发顺利！**
