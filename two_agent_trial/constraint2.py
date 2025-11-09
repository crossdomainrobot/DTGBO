# @File : organize_info.py
# Author: Hou Chenfei (adapted)
# Time: 2025-11-06

from openai import OpenAI

# ==============================
# 初始化模型客户端（兼容 DashScope）
# ==============================
llm = OpenAI(
    api_key="sk-f0a47a747acd44199169c01831af65e3",  # 你的密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ==============================
# 文件路径设置
# ==============================
text_path = r"D:\Aresearch\大模型+优化器_雪车论文（在投）\雪车论文2\数据库\constraints.txt"

# ==============================
# 提示词（用于组织信息）
# ==============================
organize_prompt = """
You are an information organizer. I will provide content related to the bobsleigh track centerline.
Please summarize and organize the information by grouping similar content together and removing irrelevant parts (such as sources, credibility notes, or polite expressions).

You must output exactly two concise pieces of key information:

First paragraph: Information related to the track centerline length.
Second paragraph: Information related to the cumulative elevation.

Your response must be concise, precise, and focused — exclude all content unrelated to centerline length or cumulative elevation.
"""

# ==============================
# 从文件读取文本
# ==============================
def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.strip()

# ==============================
# 主函数
# ==============================
def organize_centerline_info(text_input):
    messages = [
        {"role": "system", "content": "You are a precise and structured information organizer."},
        {"role": "user", "content": organize_prompt},
        {"role": "user", "content": f"Here is the provided content:\n\n{text_input}"},
    ]

    print("\n🚀 正在整理信息，请稍候……\n")

    response = llm.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        stream=False
    )

    if hasattr(response, "choices") and response.choices:
        output_text = getattr(response.choices[0].message, "content", "") or ""
    else:
        output_text = str(response)

    print("======= 整理结果 =======\n")
    print(output_text)
    print("========================\n")


# ==============================
# 程序入口
# ==============================
if __name__ == "__main__":
    try:
        text_content = read_text_file(text_path)
        organize_centerline_info(text_content)
    except FileNotFoundError:
        print("❌ 未找到指定的文件，请检查路径是否正确。")
    except Exception as e:
        print(f"⚠️ 发生错误：{e}")
