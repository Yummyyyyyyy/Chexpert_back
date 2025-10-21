# 快速启动指南

> 5分钟内启动CheXpert后端服务

---

## 第一次运行

### 1. 安装依赖（仅第一次）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境（仅第一次）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env（可选，使用默认配置也能运行）
# 注意：如果要运行真实模型，需要配置模型路径
```

### 3. 启动服务

**方式1: 使用启动脚本（推荐）**
```bash
# macOS/Linux
./start.sh

# Windows
start.bat
```

**方式2: 直接运行**
```bash
# 激活虚拟环境后
python app/main.py

# 或
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 验证服务

打开浏览器访问：
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

---

## 后续运行

```bash
# 1. 激活虚拟环境
source venv/bin/activate  # macOS/Linux

# 2. 启动服务
python app/main.py
```

---

## 测试API

### 方法1: 使用测试脚本
```bash
python test_api.py
```

### 方法2: 使用Swagger UI
访问 http://localhost:8000/docs，在线测试所有接口

### 方法3: 使用curl
```bash
# 健康检查
curl http://localhost:8000/health

# 测试知识图谱（不需要上传文件）
curl -X POST "http://localhost:8000/api/v1/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '{"disease_name": "Cardiomegaly", "language": "zh"}'
```

---

## 前端对接

### 前端需要知道的信息

1. **后端地址**: `http://localhost:8000`
2. **API前缀**: `/api/v1`
3. **三个主要接口**:
   - 图像分析: `POST /api/v1/image/analyze`
   - 报告生成: `POST /api/v1/report/generate`
   - 知识图谱: `POST /api/v1/knowledge/query`

### 前端调用示例

```javascript
// 1. 上传图片并分析
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/api/v1/image/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

详细API文档请查看 [API_DOCS.md](./API_DOCS.md)

---

## 常见问题

### Q: 端口8000已被占用？
修改端口：编辑 `.env` 文件，修改 `PORT=8001`

### Q: 前端调用出现CORS错误？
在 `app/config.py` 中添加前端地址到 `CORS_ORIGINS`

### Q: 想要使用真实模型？
查看 [TEAM_GUIDE.md](./TEAM_GUIDE.md)，按步骤实现模型集成

---

## 项目文档

- **README.md**: 项目完整说明
- **API_DOCS.md**: 前端对接文档
- **TEAM_GUIDE.md**: 后端团队开发指南

---

**启动成功后，访问 http://localhost:8000/docs 查看完整API文档！**
