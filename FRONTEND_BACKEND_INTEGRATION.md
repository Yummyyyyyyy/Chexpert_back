# 前后端联调成功指南

## 当前状态

✅ **后端服务**: http://localhost:8000 (运行中)
✅ **前端服务**: http://localhost:3000 (运行中)
✅ **API已连接**: 所有三个功能模块已连接到后端

---

## 已完成的配置

### 1. 后端API端点

- 健康检查: http://localhost:8000/health
- 图像分析: http://localhost:8000/api/v1/image/analyze
- 报告生成: http://localhost:8000/api/v1/report/generate
- 知识图谱: http://localhost:8000/api/v1/knowledge/query
- API文档: http://localhost:8000/docs

### 2. 前端API配置

已创建 `src/config/api.js` 统一管理后端地址：
- API_BASE_URL: http://localhost:8000
- 已配置CORS，前端可以直接调用

### 3. 已更新的前端API模块

✅ **upload/api.js**: 图像上传与分析
✅ **llava-report/api.js**: 医学报告生成  
✅ **third-party-api/api.js**: 疾病知识图谱

---

## 如何测试前后端连接

### 方式1: 使用浏览器（推荐）

1. 打开浏览器访问: http://localhost:3000
2. 上传一张X光片图像（jpg/png格式）
3. 系统会自动：
   - 调用后端分析图像
   - 显示分类结果和热力图
   - 生成医学报告
   - 查询知识图谱

### 方式2: 手动测试单个接口

**测试图像分析:**
```bash
curl -X POST "http://localhost:8000/api/v1/image/analyze" \
  -F "file=@test_image.jpg"
```

**测试知识图谱:**
```bash
curl -X POST "http://localhost:8000/api/v1/knowledge/query" \
  -H "Content-Type: application/json" \
  -d '"disease_name": "Cardiomegaly", "language": "zh"'
```

---

## 预期工作流程

1. **用户上传图片** → 前端调用 `uploadAndAnalyzeImage()`
   - 后端返回分类结果和热力图URL
   - 前端显示结果

2. **生成医学报告** → 前端调用 `generateMedicalReport()`
   - 使用上一步返回的图片路径
   - 后端调用LLaVA模型（当前返回模拟报告）
   - 前端显示报告内容

3. **查询知识图谱** → 前端调用 `fetchKnowledgeGraph()`
   - 使用检测到的疾病名称
   - 后端返回知识图谱数据（当前返回模拟数据）
   - 前端渲染知识图谱可视化

---

## 当前返回的是模拟数据

⚠️ **重要提示**: 

后端目前返回的是**模拟数据**，因为真实的模型逻辑尚未实现。

要使用真实的AI模型，后端团队需要：

1. 在 `app/models/model_manager.py` 中加载真实模型
2. 在 `app/models/heatmap_generator.py` 中实现Grad-CAM
3. 在 `app/services/llava_service.py` 中实现LLaVA推理
4. 在 `app/services/third_party_service.py` 中配置真实API

详见: `/Users/yangminyue/coding_coding/CheXpert_back/TEAM_GUIDE.md`

---

## 查看日志

### 后端日志
```bash
tail -f /Users/yangminyue/coding_coding/CheXpert_back/logs/app.log
```

### 前端日志
```bash
tail -f /tmp/chexpert_frontend.log
```

或直接在浏览器开发者工具的Console中查看

---

## 停止服务

### 停止后端
```bash
lsof -ti:8000 | xargs kill -9
```

### 停止前端
```bash
lsof -ti:3000 | xargs kill -9
```

---

## 重启服务

### 重启后端
```bash
cd /Users/yangminyue/coding_coding/CheXpert_back
python run.py
```

### 重启前端
```bash
cd "/Users/yangminyue/github projects/Chexpert_front"
npm start
```

---

## 常见问题

### Q: 前端调用后端出现CORS错误？
A: 已配置CORS，如果仍有问题，检查后端 `app/config.py` 中的 `CORS_ORIGINS` 是否包含 `http://localhost:3000`

### Q: 图片上传失败？
A: 检查：
1. 图片格式是否支持（jpg/jpeg/png/dcm）
2. 图片大小是否超过10MB
3. 后端 `uploads/` 目录是否存在

### Q: API返回404？
A: 检查：
1. 后端服务是否运行: `curl http://localhost:8000/health`
2. API路径是否正确
3. 查看前端Console和后端日志

---

## 下一步

1. ✅ 在浏览器中测试完整流程
2. ✅ 检查前端Console是否有错误
3. ✅ 查看后端返回的模拟数据
4. ⏳ 后端团队实现真实的模型逻辑
5. ⏳ 前端根据真实数据调整UI展示

---

**祝调试顺利！有问题随时查看文档或日志。**
