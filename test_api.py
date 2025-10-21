"""
快速测试脚本
用于验证API接口是否正常工作
运行: python test_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查接口"""
    print("\n=== 测试健康检查接口 ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200


def test_image_analyze():
    """测试图像分析接口（需要提供测试图片）"""
    print("\n=== 测试图像分析接口 ===")

    # 【TODO】替换为实际的测试图片路径
    test_image_path = "test_image.jpg"

    try:
        with open(test_image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/api/v1/image/analyze", files=files)
            print(f"状态码: {response.status_code}")
            print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
            return response.status_code == 200
    except FileNotFoundError:
        print(f"⚠️  测试图片不存在: {test_image_path}")
        print("请准备一张测试图片，或跳过此测试")
        return False


def test_report_generate():
    """测试报告生成接口"""
    print("\n=== 测试报告生成接口 ===")

    data = {
        "image_path": "/uploads/test.jpg",
        "prompt": "You are an experienced radiologist. Analyze this chest X-ray image and generate a comprehensive diagnostic report including FINDINGS, IMPRESSION AND SUMMARY sections."
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/report/generate",
        json=data,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {response.status_code}")
    result = response.json()

    if response.status_code == 200:
        print(f"成功: {result.get('success')}")
        print(f"报告摘要: {result.get('report', '')[:200]}...")
    else:
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

    return response.status_code in [200, 404]  # 404是正常的（文件不存在）


def test_knowledge_query():
    """测试知识图谱查询接口"""
    print("\n=== 测试知识图谱查询接口 ===")

    data = {
        "disease_name": "Cardiomegaly",
        "language": "zh"
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/knowledge/query",
        json=data,
        headers={"Content-Type": "application/json"}
    )

    print(f"状态码: {response.status_code}")
    result = response.json()

    if response.status_code == 200:
        print(f"成功: {result.get('success')}")
        print(f"疾病名称: {result.get('disease_name')}")
        kg = result.get('knowledge_graph', {})
        print(f"描述: {kg.get('description', '')[:100]}...")
        print(f"症状数量: {len(kg.get('symptoms', []))}")
    else:
        print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

    return response.status_code == 200


def main():
    """运行所有测试"""
    print("=" * 60)
    print("  CheXpert Backend API 测试脚本")
    print("=" * 60)
    print(f"目标服务: {BASE_URL}")
    print("请确保服务已启动！")

    results = {
        "健康检查": False,
        "图像分析": False,
        "报告生成": False,
        "知识图谱": False
    }

    try:
        results["健康检查"] = test_health()
        results["图像分析"] = test_image_analyze()
        results["报告生成"] = test_report_generate()
        results["知识图谱"] = test_knowledge_query()
    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到服务器！")
        print("请先启动服务: python app/main.py")
        return

    # 打印测试结果
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查服务状态")


if __name__ == "__main__":
    main()
