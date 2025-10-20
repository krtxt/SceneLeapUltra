import re

# 读取报告文件
with open('/home/xiantuo/source/grasp/SceneLeapUltra/tests/grasp_analysis_results/grasp_distribution_analysis_report.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找所有包含场景和抓取数的行
pattern = r'场景: ([^,]+), 物体索引: \d+, 名称: [^,]+, 抓取数: (\d+)'
matches = re.findall(pattern, content)

# 过滤出抓取数少于512的物体，并收集其场景
low_grasp_scenes = set()
for match in matches:
    scene, grasp_count = match
    if int(grasp_count) < 512:
        low_grasp_scenes.add(scene)

# 输出结果
print("拥有抓取数少于512个的物体的场景:")
for scene in sorted(low_grasp_scenes):
    print(f"- {scene}")