# @File : optimizer2 .py
# @File : optimizer_height.py
import os
import re
import numpy as np
import sys
from openai import OpenAI

# ================== 日志目录 ===================
LOG_DIR = r"D:\Aresearch\雪车论文\雪车论文2\数据\OPRO_distributed\22_comparision_SDS_textgrad"
os.makedirs(LOG_DIR, exist_ok=True)

# ================== 阶段配置（对应 Table 1） ===================
STAGE_STEP_CONFIG = {
    "early": {
        "min_abs_step": 0.0,    # 表 1: [-4, 4]，最小可为 0
        "max_abs_step": 4.0,
    },
    "mid": {
        "min_abs_step": 0.001,  # 表 1: [0.001, 1]
        "max_abs_step": 1.0,
    },
    "late": {
        "min_abs_step": 0.001,  # 表 1: [0.001, 0.1]
        "max_abs_step": 0.1,
    },
}


# ================== 一些基础函数 ===================

def compute_segment_distances(x_coord, y_coord, segment_points):
    """
    根据 2D 轨迹和每段的点数，计算每个 segment 的水平距离。
    """
    n_segments = len(segment_points)
    distances = np.zeros(n_segments)
    for i in range(n_segments):
        start_idx = np.sum(segment_points[:i]) if i > 0 else 0
        end_idx = np.sum(segment_points[:i+1])
        dx = x_coord[end_idx - 1] - x_coord[start_idx]
        dy = y_coord[end_idx - 1] - y_coord[start_idx]
        distances[i] = np.sqrt(dx**2 + dy**2)
    return distances


def extract_first_float(text: str) -> float:
    """
    从 LLM 输出中抽取第一个浮点数。
    如果没抽到，就抛异常。
    """
    pattern = r'-?\d+(?:\.\d+)?'
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"No float found in LLM output: {text}")
    return float(m.group(0))


def extract_stage_label(text: str) -> str:
    """
    从 LLM 输出中抽取阶段标签：early / mid / late
    如果没抽到，就默认 early。
    """
    low = text.lower()
    if "late" in low:
        return "late"
    if "mid" in low or "middle" in low:
        return "mid"
    return "early"


# ================== 轨道 cost 函数实现 ===================

def build_z_from_height_differences(x, y, segment_points, height_differences, H_target):
    """
    根据每段的高度差，构造 z 坐标。
    """
    n = len(x)
    z = np.zeros_like(x)

    cumulative_height = H_target - np.cumsum(
        np.concatenate(([0.0], height_differences.astype(float)))
    )

    start_idx = 0
    for i, seg_len in enumerate(segment_points):
        end_idx = start_idx + seg_len
        z[start_idx:end_idx] = np.linspace(
            cumulative_height[i],
            cumulative_height[i+1],
            seg_len
        )
        start_idx = end_idx

    return z


def compute_full_cost(
    x, y, segment_points, height_differences,
    L_target=1270,
    H_target=140,
    max_slope=0.204,
    mean_slope=0.116,
):
    """
    完整 cost 函数（与 prompt 思路一致：长度/落差=硬边界+软偏好；坡度超限软化）。
    返回 cost 和若干中间量（方便 debug）。
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    height_differences = np.asarray(height_differences, dtype=float)
    n = len(x)

    # ================== 构造 z 坐标 ===================
    z = build_z_from_height_differences(x, y, segment_points,
                                        height_differences, H_target)

    # ================== 一阶/二阶导数 + 曲率 ===================
    dx1 = np.zeros(n); dy1 = np.zeros(n); dz = np.zeros(n)
    ddx = np.zeros(n); ddy = np.zeros(n); ddz = np.zeros(n)
    curvature_2d = np.zeros(n); curvature_3d = np.zeros(n)
    curvature_difference = np.zeros(n)

    for i in range(1, n - 1):
        dx1[i] = (x[i+1] - x[i-1]) / 2
        dy1[i] = (y[i+1] - y[i-1]) / 2
        dz[i]  = (z[i+1] - z[i-1]) / 2
        ddx[i] = x[i+1] - 2 * x[i] + x[i-1]
        ddy[i] = y[i+1] - 2 * y[i] + y[i-1]
        ddz[i] = z[i+1] - 2 * z[i] + z[i-1]

    dx1[0],  dy1[0],  dz[0]  = x[1]-x[0],   y[1]-y[0],   z[1]-z[0]
    dx1[-1], dy1[-1], dz[-1] = x[-1]-x[-2], y[-1]-y[-2], z[-1]-z[-2]
    ddx[0],  ddy[0],  ddz[0] = dx1[1]-dx1[0], dy1[1]-dy1[0], dz[0]-dz[1]
    ddx[-1], ddy[-1], ddz[-1] = dx1[-1]-dx1[-2], dy1[-1]-dy1[-2], dz[-1]-dz[-2]

    for i in range(n):
        dx1_sq, dy1_sq, dz1_sq = dx1[i]**2, dy1[i]**2, dz[i]**2
        ddx_sq, ddy_sq, ddz_sq = ddx[i]**2, ddy[i]**2, ddz[i]**2

        norm_d  = dx1_sq + dy1_sq + dz1_sq
        norm_dd = ddx_sq + ddy_sq + ddz_sq
        dot_dd  = dx1[i]*ddx[i] + dy1[i]*ddy[i] + dz[i]*ddz[i]

        denom3d = (np.sqrt(norm_d)**3 + 1e-8)
        num3d   = norm_dd * norm_d - dot_dd**2
        if not np.isfinite(num3d) or num3d < 0:
            num3d = 0.0

        curvature_2d[i] = abs(dx1[i]*ddy[i] - ddx[i]*dy1[i]) / ((dx1_sq + dy1_sq)**1.5 + 1e-8)
        curvature_3d[i] = np.sqrt(num3d) / denom3d
        curvature_difference[i] = curvature_2d[i] - curvature_3d[i]

    curvature_difference_sum = np.sum(curvature_difference)

    # ================== 坡度与均值 ===================
    distances1 = np.zeros(n)
    slope1 = np.zeros(n)
    for i in range(1, n):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dist = np.sqrt(dx**2 + dy**2)
        distances1[i] = distances1[i-1] + dist
        slope1[i] = (z[i-1] - z[i]) / (dist + 1e-8)
    slope_mean = np.sum(slope1) / n

    # ================== 总长度与总落差 ===================
    segment_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    total_length = np.sum(segment_lengths)
    current_height = np.sum(height_differences)

    # ================== 分层代价 ===================
    def huber(r, delta):
        ar = np.abs(r)
        return np.where(ar <= delta, 0.5 * r**2, delta * (ar - 0.5 * delta))

    # 长度：硬上界 + 软目标
    tau_L = 20.0
    L = total_length
    C_L_hard = (np.maximum(0.0, L - 1300.0) ** 2) / (1300.0 ** 2)
    C_L_soft = huber(L - 1270.0, tau_L) / (1270.0 ** 2)
    wL_hard, wL_soft = 1.0, 0.2
    C_L = wL_hard * C_L_hard + wL_soft * C_L_soft

    # 落差：硬区间 + 带状偏好
    H = current_height
    C_H_hard = (np.maximum(0.0, 120.0 - H) ** 2) / (120.0 ** 2) \
             + (np.maximum(0.0, H - 150.0) ** 2) / (150.0 ** 2)
    band_dev = np.maximum(0.0, np.maximum(130.0 - H, H - 140.0))
    C_H_band = (band_dev ** 2) / (10.0 ** 2)
    wH_hard, wH_band = 1.0, 0.4
    C_H = wH_hard * C_H_hard + wH_band * C_H_band

    # 曲率差
    C_C = curvature_difference_sum / n

    # 坡度超限软化
    smax = max_slope
    excess = np.maximum(0.0, np.abs(slope1) - smax)
    C_S = np.mean(excess ** 2) / (smax ** 2)

    # 平均坡度偏差
    C_S_S = slope_mean - mean_slope

    # 合成总 cost（外层乘 10000，让整体 cost 大一点）
    cost = 10000 * (
        0.25 * C_L
        + 0.35 * C_H
        + 0.04 * np.abs(C_C)
        + 0.35 * np.abs(C_S)
        + 0.22 * np.abs(C_S_S)
    )

    return float(cost), dict(
        total_length=total_length,
        current_height=current_height,
        C_L=C_L,
        C_H=C_H,
        C_C=C_C,
        C_S=C_S,
        C_S_S=C_S_S,
        slope_mean=slope_mean,
        curvature_difference_sum=curvature_difference_sum,
        C_L_hard=C_L_hard,
        C_L_soft=C_L_soft,
        C_H_hard=C_H_hard,
        C_H_band=C_H_band,
    )


# ================== 构造给 LLM 的提示 ===================

def build_system_prompt():
    """
    固定的 system prompt，不再训练它。
    让模型扮演“基于 cost 的优化器”，专门优化第二段高度差。
    """
    system_prompt = (
        " (Background) In bobsleigh races, athletes must steer the sled along a predefined track as quickly as possible to reach the finish line. "
        " (Background) The key characteristic of the track is mainly determined by its 3D centerline, which defines the length, curvature, slopes, and other features of the track. "
        " (Background) However, obtaining 3D centerline data is impossible in most cases. (Background) Based on race requirements and experience, converting 2D track data into 3D data is a highly cost-effective approach. "
        " (Background) I have formulated this conversion process  as a mathematical optimization problem, where the optimization variables are the height differences, i.e. height_differences, along the track segments. "
        " (Role) Your role is to act as a gradient-based optimizer that, given my cost feedback, optimizes the 2nd height and minimizes the cost. "
        " (Role) The optimization process is described in the following code: "
        " python "
        # 这里是你原来写在 prompt 里的 cost 代码（略），我保持不动
        "def compute_full_cost("
        "    x, y, segment_points, height_differences,"
        "    max_slope=0.204,"
        "    mean_slope=0.116,"
        "):"
        "    ... "
        " print(f\"\\n✅ Cost = {cost:.6f}\") "
        " # ================== End of Code =================== "
        " (Task) You need to understand how the cost is calculated and how the height differences influence the cost in the optimization process. "
        " (Task) Based on this understanding, you will optimize the 2nd `height_differences`. "
        " (Task) You should directly optimizing the 2nd segment. I trust you ability! "
        " Steps: "
        " (Step) 1. Optimize the 2nd element, which means you need to output only one element each time."
        " (Step) 2. Iteratively optimize the 2nd height difference. During each iteration, the height difference may only change by ±4. Once the direction of convergence becomes clear, gradually reduce the step size. The final value must reach a precision of 0.001 and the final cost must less than 0.1. "
        " (Step) 3. You are strictly forbidden to stop optimizing a segment unless both of the following conditions are fully and simultaneously satisfied: "
        "     (Condition A) The overall cost computed by the reference cost function is strictly less than 0.1. "
        "     (Condition B) The optimization for all segments has reached a step size less than or equal to 0.001. "
        "     Only when both conditions are satisfied at the same time, you are allowed to terminate optimization for that segment and output the sentence: 'This is my final decision!' "
        "     If either condition is not satisfied, you must continue the iterative optimization without interruption. No exceptions are allowed. "
        "     Premature use of the phrase 'This is my final decision!' is strictly prohibited and must never occur under any circumstances until both conditions are met. "
        " (Step) 4. Combine all optimized values to update the complete set of height_differences. "
        " Remember that the final value must reach a precision of 0.001. Please carefully think through the decomposition and optimization process step by step. "
        " Never say 'This is my final decision!' because once you say that, the program will treat it as the end"
        "of the iteration. Just keep it in mind — there's no need to say anything like : we're still far from being able to say 'This is my final decision!' either."
        "Please think step by step"
    )
    return system_prompt


def build_formatted_input(
    initial_height_differences,
    current_heights,
    segment_idx,
    current_cost,
    iteration_idx,
    stage,
    min_abs_step,
    max_abs_step,
    prev_summary=None,
):
    """
    把当前状态包装成 user prompt，发给“高度优化大模型”。
    会包含：
      - 当前阶段 stage
      - 本阶段允许的步长区间 [min_abs_step, max_abs_step]
      - 上一轮由大模型 C 凝练的 summary（若有）
    """
    old_h2 = float(current_heights[segment_idx])

    common_tail = (
        f"\nStage & step constraints:\n"
        f"- Current stage: {stage}\n"
        f"- Allowed step for this stage (Δ = new_h2 - old_h2): "
        f"abs(Δ) ∈ [{min_abs_step:.3f}, {max_abs_step:.3f}]\n"
        f"- Current old_h2 = {old_h2:.6f}\n"
        f"- Current total cost = {current_cost:.6f}\n\n"
        "Task:\n"
        "- You are optimizing ONLY the 2nd element of height_differences (index 1).\n"
        "- Based on the current heights, the cost, the stage-specific step-size constraint,\n"
        "  and the condensed summary of the previous iteration (if provided),\n"
        "  propose a NEW value for height_differences[1] (call it new_h2).\n"
        "- The update step is Δ = new_h2 - old_h2, and you MUST ensure abs(Δ) is within the interval above.\n"
        "- You MUST output only ONE floating-point number (no extra text, no explanation).\n"
        "- The number you output will be directly used as the new height_differences[1].\n"
    )

    if iteration_idx % 10 == 0:
        msg = (
            " (Background) In bobsleigh races, athletes must steer the sled along a predefined track as quickly as possible to reach the finish line.\n"
            " (Background) The key characteristic of the track is mainly determined by its 3D centerline, which defines the length, curvature, slopes, and other features of the track.\n"
            " (Background) However, obtaining 3D centerline data is impossible in most cases. (Background) Based on race requirements and experience, converting 2D track data into 3D data is a highly cost-effective approach.\n"
            " (Background) I have formulated this conversion process as a mathematical optimization problem, where the optimization variables are the height differences, i.e. height_differences, along the track segments.\n"
            " (Role) Your role is to act as a gradient-based optimizer that, given my cost feedback, optimizes the 2nd height and minimizes the cost.\n\n"
            f"Iteration: {iteration_idx}\n"
            f"Initial height_differences (distance-proportional initialization):\n"
            f"{initial_height_differences.tolist()}\n\n"
            f"We are optimizing the height_differences for segment index {segment_idx} (0-based index).\n"
            f"Current height_differences:\n{current_heights.tolist()}\n"
            f"The current total cost is {current_cost:.6f}.\n"
        )
        if prev_summary is not None:
            msg += (
                "\nCondensed summary of the previous iteration:\n"
                f"{prev_summary}\n"
            )
        msg += common_tail
    else:
        msg = (
            f"Iteration: {iteration_idx}\n"
            f"Initial height_differences (distance-proportional initialization):\n"
            f"{initial_height_differences.tolist()}\n\n"
            f"We are optimizing the height_differences for segment index {segment_idx} (0-based index).\n"
            f"Current height_differences:\n{current_heights.tolist()}\n"
            f"The current total cost is {current_cost:.6f}.\n"
        )
        if prev_summary is not None:
            msg += (
                "\nCondensed summary of the previous iteration:\n"
                f"{prev_summary}\n"
            )
        msg += common_tail

    return msg


# ================== LLM 调用封装 ===================

def call_llm_for_height(
    client,
    model_name,
    system_prompt,
    user_text,
):
    """
    调用 qwen-plus 一次，返回一个 float（新的 height_differences[1]）。
    """
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    new_val = extract_first_float(content)
    return new_val, content


def call_llm_for_stage(
    client,
    model_name,
    iteration_idx,
    current_cost,
):
    """
    大模型 A：根据当前迭代次数 t 和 cost，判定当前处于 early/mid/late。
    """
    system_prompt = (
        "You are a controller for an iterative optimization algorithm.\n"
        "Your only job is to decide which stage the optimization is currently in.\n"
        "There are exactly three stages: 'early', 'mid', and 'late'.\n\n"
        "Guidelines (from prior knowledge K and Table 1):\n"
        "- early stage:\n"
        "  1) Avoid getting trapped in local optima;\n"
        "  2) Identify the direction for optimization;\n"
        "  3) Make bold adjustments to the decision variable e;\n"
        "  4) Convergence is prohibited.\n"
        "  Step size range: abs(step) in [0, 4].\n\n"
        "- mid stage:\n"
        "  1) Optimize e in the predefined promising direction;\n"
        "  2) Make cautious adjustments to e;\n"
        "  3) Convergence is not permitted in principle.\n"
        "  Step size range: abs(step) in [0.001, 1].\n\n"
        "- late stage:\n"
        "  1) Fine-tune e;\n"
        "  2) Terminate once the convergence criterion is met.\n"
        "  Step size range: abs(step) in [0.001, 0.1].\n\n"
        "Use both the iteration index t and the current total cost I_cost(t)\n"
        "to decide which stage best matches the current situation.\n"
        "You MUST output exactly one word: 'early', 'mid', or 'late'."
    )

    user_prompt = (
        f"Current iteration index t = {iteration_idx}.\n"
        f"Current total cost I_cost(t) = {current_cost:.6f}.\n\n"
        "Decide the current stage p in {early, mid, late}.\n"
        "Output only one word: early, mid, or late.\n"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    content = resp.choices[0].message.content
    stage = extract_stage_label(content)
    return stage, content


def call_llm_for_iteration_summary(
    client,
    model_name,
    iteration_idx,
    stage,
    old_h2,
    new_h2,
    step,
    old_cost,
    new_cost,
    user_text,
    model_output,
):
    """
    大模型 C：对“上一轮的输入 + 输出”做凝练，总结成短文本，
    供下一轮作为高层语义记忆使用。
    """
    system_prompt = (
        "You are a summarizer for an iterative optimization process.\n"
        "Your goal is to concisely summarize ONE iteration of optimization\n"
        "based on the optimizer's input prompt and the model's numeric output.\n\n"
        "The summary will be used as high-level memory for the NEXT iteration.\n"
        "Focus on:\n"
        "- the stage (early/mid/late),\n"
        "- how the step (Δ = new_h2 - old_h2) changed the decision variable,\n"
        "- how the cost changed (better or worse),\n"
        "- whether the direction seems promising or not.\n\n"
        "Requirements:\n"
        "- Output 1–3 short sentences.\n"
        "- Do NOT include any instructions to the next model.\n"
        "- Do NOT ask questions.\n"
        "- No bullet points, no markdown, just plain text.\n"
    )

    user_prompt = (
        f"Iteration index: {iteration_idx}\n"
        f"Stage: {stage}\n"
        f"Old h2: {old_h2:.6f}\n"
        f"New h2: {new_h2:.6f}\n"
        f"Step Δ = new_h2 - old_h2 = {step:.6f}\n"
        f"Previous cost: {old_cost:.6f}\n"
        f"New cost: {new_cost:.6f}\n\n"
        "Below is the optimizer input prompt used in this iteration:\n"
        "----- OPTIMIZER INPUT BEGIN -----\n"
        f"{user_text}\n"
        "----- OPTIMIZER INPUT END -----\n\n"
        "Below is the raw numeric output from the optimizer model:\n"
        "----- MODEL OUTPUT BEGIN -----\n"
        f"{model_output}\n"
        "----- MODEL OUTPUT END -----\n\n"
        "Now, summarize this iteration according to the requirements."
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    summary = resp.choices[0].message.content.strip()
    return summary


# ================== 主优化过程 ===================

def main():
    # ===== 协同模式入口（供主控程序调用） =====
    if "--coop-helper" in sys.argv:
        target = os.environ.get("COOP_TARGET_ID", "?")          # 被协助者编号
        prev_input = os.environ.get("COOP_PREV_INPUT", "")      # 对方上一轮输入
        prev_output = os.environ.get("COOP_PREV_OUTPUT", "")    # 对方上一轮输出
        print(f"[optimizer2] Assisting optimizer{target}")
        print(f"[optimizer2] Received previous input: {prev_input}")
        print(f"[optimizer2] Received previous output: {prev_output}")
        print(f"[optimizer2] Suggestion: Try smaller step size and smooth adjustment")
        return
    # ========================================

    prev_summary = None  # 上一轮的凝练文本

    # ---------- 1. 读取数据 ----------
    data = np.loadtxt('../2-xy-www1.txt')
    x_coord = data[:, 0]
    y_coord = data[:, 1]

    segment_points = np.loadtxt('../segment_lengthswww2.txt', dtype=int)
    target_height_differences = np.loadtxt('../target_height_differences.txt')

    n_segments = len(segment_points)

    # ---------- 2. 计算按距离比例分配的初始高度差 ----------
    distances = compute_segment_distances(x_coord, y_coord, segment_points)
    total_distance = np.sum(distances)
    H_target = 138.3
    L_target = 1277.9

    initial_height_differences = H_target * (distances / total_distance)

    current_heights = initial_height_differences.copy()
    segment_idx = 1  # 只优化第 2 段（下标 1）

    # ---------- 3. 设置 DashScope 客户端 ----------
    DASHSCOPE_BASE_URL = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    QWEN_PLUS_KEY = "sk-4af60e09c8fc4c01982bc5e089d24499"

    if QWEN_PLUS_KEY is None:
        raise RuntimeError("请先在环境变量中设置 QWEN_PLUS_KEY 或直接在代码中填写。")

    forward_client = OpenAI(
        api_key=QWEN_PLUS_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )
    forward_model_name = "qwen-plus"

    # ---------- 4. 固定 system prompt ----------
    system_prompt = build_system_prompt()

    # ---------- 5. 计算初始 cost ----------
    current_cost, detail = compute_full_cost(
        x_coord, y_coord, segment_points, current_heights,
        L_target=L_target,
        H_target=H_target,
    )

    print(f"Initial cost = {current_cost:.6f}")
    print(f"Initial heights: {current_heights.tolist()}")

    # ---------- 6. 迭代优化高度 ----------
    max_iterations = 30
    cost_threshold = 0.1
    step_threshold = 0.001

    log_file = os.path.join(LOG_DIR, "optimizer2_log.txt")

    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write("iter,stage,current_cost,new_h2,step,raw_llm_output\n")

    for it in range(1, max_iterations + 1):
        old_cost = current_cost

        # 6.1 判定阶段（大模型 A）
        stage, stage_raw = call_llm_for_stage(
            forward_client,
            forward_model_name,
            iteration_idx=it,
            current_cost=current_cost,
        )

        stage_cfg = STAGE_STEP_CONFIG.get(stage, STAGE_STEP_CONFIG["early"])
        min_abs_step = stage_cfg["min_abs_step"]
        max_abs_step = stage_cfg["max_abs_step"]

        # 6.2 构造发给“大模型 B（高度优化器）”的 user prompt
        user_text = build_formatted_input(
            initial_height_differences,
            current_heights,
            segment_idx,
            current_cost,
            iteration_idx=it,
            stage=stage,
            min_abs_step=min_abs_step,
            max_abs_step=max_abs_step,
            prev_summary=prev_summary,
        )

        try:
            new_h2_raw, raw_output = call_llm_for_height(
                forward_client,
                forward_model_name,
                system_prompt,
                user_text,
            )
        except Exception as e:
            print(f"[Iter {it}] LLM 调用或解析失败：{e}")
            break

        old_h2 = current_heights[segment_idx]
        step = new_h2_raw - old_h2
        abs_step = abs(step)

        # 阶段内最大步长约束
        if abs_step > max_abs_step:
            step = np.sign(step) * max_abs_step
            abs_step = max_abs_step

        # 阶段内最小步长约束（early 允许 0）
        if (min_abs_step > 0.0) and (0.0 < abs_step < min_abs_step):
            step = np.sign(step) * min_abs_step
            abs_step = min_abs_step

        # 全局物理约束：每一步变化不能超过 ±4
        if abs_step > 4.0:
            step = 4.0 * np.sign(step)
            abs_step = 4.0

        new_h2 = old_h2 + step
        new_h2 = float(np.round(new_h2, 3))
        step = new_h2 - old_h2

        current_heights[segment_idx] = new_h2

        # 6.5 更新 cost
        current_cost, detail = compute_full_cost(
            x_coord, y_coord, segment_points, current_heights,
            L_target=L_target,
            H_target=H_target,
        )

        print(
            f"[Iter {it}] stage={stage}, old_h2={old_h2:.6f}, new_h2={new_h2:.6f}, "
            f"step={step:.6f}, cost={current_cost:.6f}"

        )
        if it == 2:
            print("[optimizer2] Manually stopping at iteration 2 (for cooperative test)")
            break

        with open(log_file, "a", encoding="utf-8") as f_log:
            safe_raw = raw_output.replace("\n", "\\n")
            f_log.write(
                f"{it},{stage},{current_cost:.6f},{new_h2:.6f},{step:.6f},{safe_raw}\n"
            )

        # 6.6 大模型 C：对本轮迭代做凝练，供下一轮使用
        try:
            prev_summary = call_llm_for_iteration_summary(
                forward_client,
                forward_model_name,
                iteration_idx=it,
                stage=stage,
                old_h2=old_h2,
                new_h2=new_h2,
                step=step,
                old_cost=old_cost,
                new_cost=current_cost,
                user_text=user_text,
                model_output=raw_output,
            )
        except Exception as e:
            print(f"[Iter {it}] 总结大模型调用失败：{e}")
            prev_summary = None

        # 判断停止条件
        if (current_cost < cost_threshold) and (abs(step) <= step_threshold):
            print(
                f"Converged at iter {it}: stage={stage}, cost={current_cost:.6f}, "
                f"step={step:.6f}, h2={new_h2:.6f}"
            )
            break

    # ---------- 7. 打印最终结果 ----------
    print("Final heights:", current_heights.tolist())
    print(f"Final cost = {current_cost:.6f}")
    np.savetxt(
        os.path.join(LOG_DIR, "optimized_height_differences.txt"),
        current_heights,
        fmt="%.6f"
    )
    print("Optimized height_differences saved.")


if __name__ == "__main__":
    """
    两种模式：
    - 正常模式：直接跑 main()，就是你现在的优化流程
    - helper 模式：由 orchestrator_experiment.py 启动，只读取邻居上一轮的信息，打印一下就退出
    """
    mode = os.getenv("COOP_MODE", "normal")

    if mode == "helper" or "--helper" in sys.argv:
        # 这是“协同辅助”模式：读取邻居上一轮的信息，基于它生成一条新输出（而不是只打印几句就退出）
        target_id = os.getenv("COOP_TARGET_ID", "?")
        coop_msg = os.getenv("COOP_MSG", "")
        coop_iter = os.getenv("COOP_ITER", "?")

        print(f"[optimizer2-helper] input_from_optimizer{target_id}: {coop_msg}")

        # =============== 解析邻居的迭代行（提取 old_h2/new_h2/cost） ===============
        import re


        def _extract_float(pattern, text):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else None


        old_h2 = _extract_float(r"old_h2=([-\d\.]+)", coop_msg)
        new_h2_neighbor = _extract_float(r"new_h2=([-\d\.]+)", coop_msg)
        cost_neighbor = _extract_float(r"cost=([-\d\.]+)", coop_msg)

        # 组一段给 LLM 的简短 prompt（如果有可用的 KEY），否则用规则 fallback 也能产出
        use_llm = False
        try:
            DASHSCOPE_BASE_URL = os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            QWEN_PLUS_KEY = os.getenv("QWEN_PLUS_KEY") or os.getenv("DASHSCOPE_API_KEY") or None
            if QWEN_PLUS_KEY:
                from openai import OpenAI

                client = OpenAI(api_key=QWEN_PLUS_KEY, base_url=DASHSCOPE_BASE_URL)
                use_llm = True
            else:
                use_llm = False
        except Exception:
            use_llm = False

        # 我们约定 helper 的输出格式为 单行：
        # [Coop 2->1 | Iter k] new_h2=..., step=..., note=...
        # （2->1 意味着 optimizer2 正在帮助 optimizer1；你也可以只写成 [optimizer2-helper] ... ）
        if use_llm:
            system_prompt = (
                "You assist a cooperative optimizer. "
                "Given the neighbor's iteration log line, propose a new value for h2. "
                "Output one single line exactly in the format: "
                "[Coop 2->{TARGET} | Iter {ITER}] new_h2={VALUE} step={STEP} note={SHORT_REASON]. "
                "Do not add extra lines."
            )
            user_prompt = (
                f"Neighbor log line:\n{coop_msg}\n\n"
                "Constraints:\n"
                "- The per-step change |Δ| should be <= 4.0.\n"
                "- If the last cost increased, reduce step magnitude.\n"
                "- If the last cost decreased, keep the direction but gradually shrink the step.\n"
                "- Only propose a single new_h2."
            )
            try:
                resp = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system",
                         "content": system_prompt.replace("{TARGET}", str(target_id)).replace("{ITER}",
                                                                                              str(coop_iter))},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                line = resp.choices[0].message.content.strip().splitlines()[0]
                # 兜底：如果模型没按格式来，就自己拼一次
                if not line.startswith("[Coop"):
                    # 简单兜底
                    # 估个步长
                    step_guess = 0.5
                    if old_h2 is not None and new_h2_neighbor is not None:
                        last_step = new_h2_neighbor - old_h2
                        # 若上一次代价变大，反向减半；若变小，则同向减半
                        if cost_neighbor is not None and cost_neighbor > 10:  # 这里阈值可调
                            step_guess = -0.5 * (1 if last_step >= 0 else -1)
                        else:
                            step_guess = 0.5 * (1 if last_step >= 0 else -1)
                        new_h2 = round(new_h2_neighbor + step_guess, 3)
                        line = f"[Coop 2->{target_id} | Iter {coop_iter}] new_h2={new_h2} step={round(step_guess, 3)} note=fallback-reformat"
                    print(line)
                else:
                    print(line)
            except Exception as e:
                # LLM 失败就用规则 fallback
                pass
        if not use_llm:
            # =============== 规则 fallback：完全不依赖 API KEY，也能产出一条规范的新输出 ===============
            # 规则：沿用邻居方向，但把步长缩小到 0.5；若看起来 cost 变大（或很大），就反向 0.5
            step_guess = 0.5
            if old_h2 is not None and new_h2_neighbor is not None:
                last_step = new_h2_neighbor - old_h2
                if cost_neighbor is not None and cost_neighbor > 10:
                    # 成本很高/上升：反向尝试
                    step_guess = -0.5 if last_step >= 0 else 0.5
                else:
                    # 成本 OK：同向微调
                    step_guess = 0.5 if last_step >= 0 else -0.5
                new_h2 = round(new_h2_neighbor + step_guess, 3)
                print(
                    f"[Coop 2->{target_id} | Iter {coop_iter}] new_h2={new_h2} step={round(step_guess, 3)} note=rule-fallback")
            else:
                # 实在解析不到，就给个温和建议
                print(f"[Coop 2->{target_id} | Iter {coop_iter}] new_h2=3.870 step=0.000 note=insufficient-context")
        # helper 完成一个输出就退出
        sys.exit(0)


