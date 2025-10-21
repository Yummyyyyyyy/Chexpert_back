# CheXpert Backend API 接口文档

> 供前端团队参考的详细接口说明

---

## 基础信息

- **Base URL**: `http://localhost:8000`
- **API Version**: `v1`
- **API Prefix**: `/api/v1`
- **Content-Type**: `application/json` (除文件上传外)

---

## 通用响应格式

### 成功响应
```json
{
  "success": true,
  "message": "操作成功",
  "data": { ... }
}
```

### 错误响应
```json
{
  "success": false,
  "error": "错误类型",
  "detail": "详细错误信息"
}
```

---

## 接口列表

## 1. 健康检查

### GET /health

检查后端服务是否正常运行。

**请求示例**:
```bash
curl http://localhost:8000/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "app_name": "CheXpert Backend API",
  "version": "1.0.0",
  "debug_mode": true
}
```

---

## 2. 图像分析接口

### POST /api/v1/image/analyze

上传X光片图像，返回疾病分类结果和热力图。

**请求参数**:
- `file` (file, required): X光片图像文件（支持jpg/jpeg/png/dcm格式）

**请求示例** (JavaScript):
```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/api/v1/image/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

**请求示例** (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/image/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

**响应示例**:
```json
{
  "success": true,
  "message": "图像分析完成",
  "original_image_url": "/uploads/original_a1b2c3d4.jpg",
  "heatmap_image_url": "/uploads/heatmap_a1b2c3d4.jpg",
  "classifications": [
    {
      "label": "Cardiomegaly",
      "confidence": 0.87,
      "description": "心脏肥大"
    },
    {
      "label": "Edema",
      "confidence": 0.65,
      "description": "肺水肿"
    },
    {
      "label": "No Finding",
      "confidence": 0.12,
      "description": "未发现异常"
    }
  ]
}
```

**字段说明**:
- `original_image_url`: 原始图片的URL路径
- `heatmap_image_url`: 热力图的URL路径
- `classifications`: 疾病分类结果数组
  - `label`: 疾病类别（英文）
  - `confidence`: 置信度（0-1之间）
  - `description`: 疾病描述（中文）

**错误响应**:
```json
{
  "success": false,
  "error": "参数验证失败",
  "detail": "不支持的文件格式，仅支持: {'.jpg', '.jpeg', '.png', '.dcm'}"
}
```

---

## 3. 报告生成接口

### POST /api/v1/report/generate

基于上传的图片，调用LLaVA模型生成详细的医学报告。

**请求参数**:
```json
{
  "image_path": "string (required)",  // 图片路径（通常是上一步返回的路径）
  "prompt": "string (optional)"        // 自定义提示词，默认为标准医学报告提示
}
```

**请求示例** (JavaScript):
```javascript
const response = await fetch('http://localhost:8000/api/v1/report/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    image_path: '/uploads/original_a1b2c3d4.jpg',
    prompt: '请详细分析这张胸部X光片，重点关注心脏和肺部'
  })
});

const result = await response.json();
console.log(result.report);
```

**请求示例** (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/report/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/uploads/original_a1b2c3d4.jpg",
    "prompt": "请分析这张X光片"
  }'
```

**响应示例**:
```json
{
  "success": true,
  "message": "报告生成成功",
  "report": "**胸部X光片分析报告**\n\n【影像观察】\n- 心影大小：轻度增大，心胸比约0.55\n- 肺野：双肺纹理增多\n...",
  "processing_time": 1.52
}
```

**字段说明**:
- `report`: 生成的医学报告（Markdown格式）
- `processing_time`: 处理耗时（秒）

**错误响应**:
```json
{
  "success": false,
  "error": "文件未找到",
  "detail": "图片文件不存在"
}
```

---

## 4. 知识图谱查询接口

### POST /api/v1/knowledge/query

根据疾病名称，查询相关的知识图谱信息。

**请求参数**:
```json
{
  "disease_name": "string (required)",  // 疾病名称（英文或中文）
  "language": "string (optional)"       // 语言，"zh"中文或"en"英文，默认"zh"
}
```

**请求示例** (JavaScript):
```javascript
const response = await fetch('http://localhost:8000/api/v1/knowledge/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    disease_name: 'Cardiomegaly',
    language: 'zh'
  })
});

const result = await response.json();
console.log(result.knowledge_graph);
```

**请求示例** (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '{
    "disease_name": "Cardiomegaly",
    "language": "zh"
  }'
```

**响应示例**:
```json
{
  "success": true,
  "message": "知识图谱查询成功",
  "disease_name": "Cardiomegaly",
  "knowledge_graph": {
    "disease_name": "Cardiomegaly",
    "language": "zh",
    "description": "心脏肥大是指心脏体积异常增大...",
    "symptoms": [
      "呼吸困难",
      "胸闷",
      "心悸",
      "乏力"
    ],
    "causes": [
      "高血压",
      "冠心病",
      "心脏瓣膜病"
    ],
    "treatments": [
      "药物治疗：利尿剂、血管扩张剂",
      "手术治疗：心脏瓣膜置换"
    ],
    "prevention": [
      "控制血压",
      "定期体检",
      "健康饮食"
    ],
    "related_diseases": [
      "心力衰竭",
      "肺动脉高压"
    ],
    "graph_visualization": {
      "nodes": [
        {"id": "1", "label": "Cardiomegaly", "type": "disease"},
        {"id": "2", "label": "高血压", "type": "cause"}
      ],
      "edges": [
        {"from": "2", "to": "1", "label": "引起"}
      ]
    }
  }
}
```

**字段说明**:
- `description`: 疾病描述
- `symptoms`: 症状列表
- `causes`: 病因列表
- `treatments`: 治疗方案列表
- `prevention`: 预防措施列表
- `related_diseases`: 相关疾病列表
- `graph_visualization`: 图谱可视化数据（可用于前端渲染知识图谱）

---

## 前端集成示例

### React示例

```jsx
import React, { useState } from 'react';

const API_BASE_URL = 'http://localhost:8000';

function ImageUpload() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      // 1. 上传图片并分析
      const response = await fetch(`${API_BASE_URL}/api/v1/image/analyze`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.success) {
        setResult(data);

        // 2. 生成报告
        const reportResponse = await fetch(`${API_BASE_URL}/api/v1/report/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_path: data.original_image_url
          })
        });
        const reportData = await reportResponse.json();

        // 3. 查询知识图谱（以第一个检测到的疾病为例）
        if (data.classifications.length > 0) {
          const knowledgeResponse = await fetch(`${API_BASE_URL}/api/v1/knowledge/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              disease_name: data.classifications[0].label,
              language: 'zh'
            })
          });
          const knowledgeData = await knowledgeResponse.json();
          console.log('知识图谱:', knowledgeData);
        }
      }
    } catch (error) {
      console.error('上传失败:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleUpload} accept=".jpg,.jpeg,.png" />
      {loading && <p>处理中...</p>}
      {result && (
        <div>
          <img src={`${API_BASE_URL}${result.heatmap_image_url}`} alt="热力图" />
          <ul>
            {result.classifications.map((item, index) => (
              <li key={index}>
                {item.description}: {(item.confidence * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
```

---

## 错误码说明

| HTTP状态码 | 说明 |
|-----------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误（如文件格式不支持） |
| 404 | 资源不存在（如图片文件未找到） |
| 422 | 参数验证失败 |
| 500 | 服务器内部错误 |

---

## 注意事项

1. **CORS**: 后端已配置CORS，前端可以直接调用
2. **文件大小**: 默认限制10MB，超过将返回错误
3. **文件格式**: 仅支持 `.jpg`, `.jpeg`, `.png`, `.dcm`
4. **图片路径**: 上传接口返回的路径可以直接用于报告生成接口
5. **异步处理**: 所有接口都是异步的，建议使用`async/await`

---

## 在线API文档

访问 http://localhost:8000/docs 可查看交互式API文档（Swagger UI），支持在线测试。

---

**更新日期**: 2025-10-16
**文档版本**: v1.0
