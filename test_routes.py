#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：列出所有注册的FastAPI路由
运行方式: python test_routes.py
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.main import app

print("=" * 80)
print("已注册的 FastAPI 路由:")
print("=" * 80)

routes = []
for route in app.routes:
    if hasattr(route, 'methods') and hasattr(route, 'path'):
        methods = ','.join(route.methods)
        path = route.path
        name = route.name if hasattr(route, 'name') else ''
        routes.append((methods, path, name))

# 按路径排序
routes.sort(key=lambda x: x[1])

for methods, path, name in routes:
    print(f"{methods:10} {path:50} [{name}]")

print("=" * 80)
print(f"总共 {len(routes)} 个路由")
print("=" * 80)

# 检查关键路由
print("\n关键路由检查:")
key_routes = [
    "/api/v1/report/generate",
    "/api/v1/report/generate-v2",
    "/api/v1/report/generate-markdown",
]

for key_route in key_routes:
    found = any(path == key_route for _, path, _ in routes)
    status = "✅ 已注册" if found else "❌ 未注册"
    print(f"{status:15} {key_route}")

print("=" * 80)
