# @File : constraints.py
# Author: Hou Chenfei (adapted)
# Time: 2025-11-06

from openai import OpenAI
import os
import PyPDF2

output_txt = r"D:\Aresearch\大模型+优化器_雪车论文（在投）\雪车论文2\数据库\constraints.txt"
pdf_path = r"D:\Aresearch\大模型+优化器_雪车论文（在投）\雪车论文2\数据库\Whistler Sliding Centre Sled Trajectory and Track Construction Study(1).pdf"
urls = [
    "https://www.ibsf.org/de/unser-sport/bob",
    "https://www.ibsf.org/de/bahn/24/whistler?cHash=cf5738db772e877df852a731480430c0"
]

llm1 = OpenAI(
    api_key="sk-f0a47a747acd44199169c01831af65e3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

llm2 = OpenAI(
    api_key="sk-8b32b10486ec484a899f864283d27294",  # 第二个 LLM 的 key，可改成你的第二个
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def read_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text[:30000]


extract_prompt = """
You are an information extractor. I will provide a PDF file and two website links. You need to extract any explicit constraints, implicit constraints, or empirical constraints related to the track centerline length and the cumulative elevations. For example: “The athlete’s acceleration should not exceed 10 m/s²,” or “The track length should not be too long, and the total completion time should not exceed 60 minutes.”
From file X: ….. (explicit, implicit, or empirical content related to the track centerline length and elevations)
From webpage Y: ….. (explicit, implicit, or empirical content related to the track centerline length and elevations)
From webpage Z: ….. (explicit, implicit, or empirical content related to the track centerline length and elevations)

Ensure accuracy. If the provided files do not contain the relevant content, please clearly state that no such information is available in those files. If you're not sure whether it meets the requirements, you could include it as well. Avoid irrelevant descriptions and keep it concise, clear, and to the point. Lastly, please think through each step carefully.
"""

summary_prompt = """
I will provide you with some content about the bobsleigh track centerline. 
Please summarize and organize it by grouping similar information together and removing irrelevant parts (such as information sources, credibility statements, or polite expressions).

You must output only two pieces of information:

The first paragraph should be about the track centerline length.

The second paragraph should be about the cumulative elevation.

Be concise and to the point — do not include any content unrelated to centerline length or cumulative elevation.
"""

def extract_constraints():
    pdf_text = read_pdf_text(pdf_path)

    # ---------- 第一阶段：提取 ----------
    messages1 = [
        {"role": "system", "content": "You are a precise and concise technical analyst."},
        {"role": "user", "content": extract_prompt},
        {"role": "user", "content": f"The two webpages are:\n{urls[0]}\n{urls[1]}"},
        {"role": "user", "content": f"Here is the text extracted from the PDF file:\n\n{pdf_text}"},
    ]

    print("\n🚀 Starting LLM-1 Extraction...\n")

    response1 = llm1.chat.completions.create(
        model="qwen-plus",
        messages=messages1,
        stream=False
    )

    if hasattr(response1, "choices") and response1.choices:
        llm1_output = getattr(response1.choices[0].message, "content", "") or ""
    else:
        llm1_output = str(response1)

    print("======= LLM-1 Output =======\n")
    print(llm1_output)

    messages2 = [
        {"role": "system", "content": "You are a precise summarizer focusing only on track length and elevation."},
        {"role": "user", "content": summary_prompt},
        {"role": "user", "content": f"Here is the extracted content from the first LLM:\n\n{llm1_output}"},
    ]

    print("🚀 Starting LLM-2 Summarization...\n")

    response2 = llm2.chat.completions.create(
        model="qwen-plus",
        messages=messages2,
        stream=False
    )

    if hasattr(response2, "choices") and response2.choices:
        llm2_output = getattr(response2.choices[0].message, "content", "") or ""
    else:
        llm2_output = str(response2)

    print("======= LLM-2 Final Summary =======\n")
    print(llm2_output)

    try:
        with open(output_txt, "a", encoding="utf-8") as f:  # 以“追加”模式打开文件
            f.write("\n\n======= New Summary (LLM-2) =======\n")
            f.write(llm2_output.strip())
            f.write("\n==============================\n")
    except Exception as e:
        print(f"\n error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!：{e}")



if __name__ == "__main__":
    extract_constraints()
