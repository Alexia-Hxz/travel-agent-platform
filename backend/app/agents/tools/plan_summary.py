import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.agents.tools.llm import MODEL_NAME

logger = logging.getLogger(__name__)

# This is the only model call in the project that used to leave thinking on, and it cost more
# than time. Measured on one request and payload: thinking on answered in 28.8s with 340
# visible characters, thinking off in 11.2s with 820 -- the reasoning had been spending the
# 700-token output budget on text nobody sees. With a real plan's larger context the call then
# ran past the shared 30s timeout twice (61.2s measured in a full run), landed in the bare
# `except` below, and every traveller got the local template instead of a written summary.
# Thinking off, with its own bound: this is the last step of a plan and must not be able to
# hold it open for a minute waiting for an answer that is already available locally.
SUMMARY_TIMEOUT = 20.0
SUMMARY_MAX_RETRIES = 1
SUMMARY_BODY = {"enable_thinking": False}


def _model_client() -> OpenAI | None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=SUMMARY_TIMEOUT,
        max_retries=SUMMARY_MAX_RETRIES,
    )


def _fallback_summary(context: dict[str, Any]) -> str:
    trip = context.get("trip", {})
    destination = trip.get("destination", "目的地")
    start_date = trip.get("start_date", "出发日")
    end_date = trip.get("end_date", "返程日")
    places = context.get("saved_places", [])
    saved_place_count = context.get("saved_place_count") or len(places)
    itinerary = context.get("itinerary", [])
    generated_plan = context.get("generated_daily_plan", [])
    budget = context.get("budget", {})
    conflicts = context.get("conflicts", [])
    transport = context.get("transport_advice") or {}
    traveler_profile = context.get("traveler_profile") or {}
    weather = context.get("weather") or {}

    sections = [f"本次 {destination} 行程为 {start_date} 至 {end_date}。"]
    if traveler_profile.get("total_people"):
        sections.append(f"同行共 {traveler_profile['total_people']} 人，其中成人 {traveler_profile.get('total_adults', 0)} 人、儿童 {traveler_profile.get('total_children', 0)} 人；项目建议优先考虑全员可参与和年龄限制。")
    if weather.get("days"):
        indoor_days = [day.get("date") for day in weather["days"] if day.get("indoor_priority")]
        if indoor_days:
            sections.append(f"天气预报提示 {', '.join(indoor_days)} 需要优先准备室内或酒店备选方案。")
    weather_advice = context.get("weather_advice") or []
    if weather_advice:
        # Each wet day gets a concrete line naming the affected items.
        for entry in weather_advice:
            sections.append(f"{entry.get('message', '')}{entry.get('suggestion', '')}")
    if places:
        # Every saved place must appear, with the detail that makes it actionable:
        # what it is, where it is, when it is open, and what it costs.
        sections.append(f"你收藏的 {saved_place_count} 个地点全部纳入本次行程：")
        for place in places:
            name = str(place.get("name") or "地点")
            details: list[str] = []
            if place.get("type_label"):
                details.append(str(place["type_label"]))
            if place.get("address"):
                details.append(f"地址：{place['address']}")
            if place.get("opening_hours"):
                details.append(f"开放时间：{place['opening_hours']}")
            if place.get("rating") is not None:
                details.append(f"评分 {place['rating']}")
            cost = place.get("estimated_cost")
            if cost is not None:
                currency_label = "元" if budget.get("currency", "CNY").upper() == "CNY" else budget.get("currency", "CNY")
                details.append("免费" if int(cost) <= 0 else f"预算 {int(cost)}{currency_label}")
            if place.get("note"):
                details.append(f"选择理由：{place['note']}")
            suffix = f"（{'；'.join(details)}）" if details else ""
            sections.append(f"  - {name}{suffix}")
    else:
        sections.append("目前还没有收藏地点，建议先在“生成规划建议”里勾选项目并加入收藏地点。")
    selected = context.get("selected_recommendations") or []
    if selected:
        selected_names = "、".join(str(item.get("name")) for item in selected[:6])
        selected_total = sum(int(item.get("estimated_cost") or 0) for item in selected)
        sections.append(
            f"你在规划建议中选中的推荐项目有 {len(selected)} 个：{selected_names}，"
            f"合计约 {selected_total} {budget.get('currency', 'CNY')}；"
            # Ticked recommendations are candidates until they are actually scheduled -
            # previously this line claimed they were already folded into the plan.
            f"这些目前还只是候选，需要排入每日行程后才是确定安排。"
        )
    adjustment = (context.get("adjustment_summary") or "").strip()
    if adjustment:
        sections.append(f"对话式调整行程的结论：{adjustment}")
    # 每日行程 is the single source of truth, so the summary lists it day by day
    # instead of only reporting a count while describing the planner's draft.
    if itinerary:
        sections.append(f"每日行程已排定 {len(itinerary)} 个行程项，这是本次确定的安排：")
        for item in sorted(itinerary, key=lambda entry: (str(entry.get("date")), str(entry.get("start_time")))):
            date = str(item.get("date") or "")
            start = str(item.get("start_time") or "")[:5]
            end = str(item.get("end_time") or "")[:5]
            name = str(item.get("place_name") or "行程项")
            line = f"  - {date} {start}-{end} {name}"
            if item.get("note"):
                line += f"（{item['note']}）"
            sections.append(line)
    elif generated_plan:
        generated_count = sum(len(day.get("items", [])) for day in generated_plan)
        sections.append(
            f"每日行程目前还是空的，下面只是建议草稿：联网推荐可覆盖第一天约 {generated_count} 段时间安排，"
            "确认排入每日行程后才会成为确定安排。"
        )
    else:
        sections.append("当前还没有每日行程项，建议先确定每天的主目的地和出发时间。")
    if transport.get("recommendation"):
        sections.append(
            f"交通方面，从{transport.get('origin') or '出发地'}到{transport.get('destination') or '目的地'}，"
            f"建议：{transport['recommendation']}"
        )
    total = budget.get("total_estimate")
    currency = budget.get("currency", "CNY")
    if total is not None:
        itemized = budget.get("itemized_summary")
        sections.append(f"预算明细：{itemized}。" if itemized else f"按当前收藏地点估算，基础预算约为 {total} {currency}。")
    if budget.get("remaining_budget") is not None:
        sections.append(f"预算余量约为 {budget['remaining_budget']} {currency}，请为交通与临时消费预留空间。")
    if conflicts:
        sections.append(f"发现 {len(conflicts)} 个时间冲突，出发前请调整重叠安排。")
    else:
        sections.append("当前未发现行程时间重叠；实时票价、营业时间和交通班次仍需在出发前确认。")
    return "".join(sections)


def generate_final_summary(context: dict[str, Any]) -> tuple[str, str]:
    """Generate a grounded final summary and preserve a useful local fallback."""
    client = _model_client()
    if client is None:
        return _fallback_summary(context), "fallback"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一名旅行规划助理。请基于提供的结构化旅行资料，用简洁中文输出一段最终旅行总结。"
                        "必须覆盖行程概览、交通建议、预算情况、冲突或缺失信息和出行注意事项。"
                        # "每日行程" (itinerary) is the single source of truth. Everything
                        # else in the context is a proposal that the user has not accepted,
                        # and the summary used to blur the two - it described the planner's
                        # own draft day as though it were the agreed plan, next to a
                        # different draft from the adjustment panel.
                        "**itinerary 是用户已经确定下来的每日行程，是唯一的事实来源："
                        "行程概览必须以它为准，逐日说明已确定的安排与该日的顺序、时段。**"
                        "generated_daily_plan、recommendations 和 selected_recommendations 都只是"
                        "**尚未排入每日行程的候选或草稿**，只能说成“还可以考虑/建议补充”，"
                        "绝对不能说成已经安排好，也不得与 itinerary 混为一谈或重复叙述。"
                        "如果 itinerary 为空，就明确说明“目前还没有排定的行程”，再给出建议。"
                        "必须逐个点名 saved_places 中的每一个收藏地点（不得遗漏、不得合并省略），"
                        "并说明它在行程中的定位与安排；资料里带有的类型、地址、开放时间、评分、"
                        "estimated_cost 和 note（用户当初选择它的理由）必须如实使用，没有的不要编造。"
                        "若资料中出现 adjustment_summary，必须单独用一句话复述该次对话式调整的结论"
                        "（让用户能看出这次调整确实生效了），再说明它如何改变了行程。"
                        "若资料中出现 weather_advice，必须逐条说明哪一天受天气影响、"
                        "当天哪些安排偏户外、建议怎么调整。"
                        "不得编造资料中没有的地点、票价、时间或事实；资料不完整时明确提示。只输出总结正文。"
                        "若资料中出现 traveler_documents，那是用户自己上传的订单、票据或确认单原文，"
                        "每条都带有来源文件名：凡涉及已预订的酒店、航班、入住与离店时间、订单号或已付金额，"
                        "必须以它为准并注明来自哪份资料，不得与之矛盾，也不得把资料里没有的预订说成已经确定。"
                        "预算必须按 itemized_summary 原样保留逐项加法格式，金额为 0 的项目必须写成“免费”。"
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
            ],
            temperature=0.2,
            extra_body=SUMMARY_BODY,
            max_tokens=700,
        )
        summary = (response.choices[0].message.content or "").strip()
        if summary:
            return summary, "completed"
        logger.warning("final summary model call returned no text; using the local summary")
    except Exception:
        # Falling back is a designed outcome, but a silent one is undiagnosable: the 61s this
        # used to burn left nothing in the log, and the plan just looked summary-less.
        logger.warning("final summary model call failed; using the local summary", exc_info=True)
    return _fallback_summary(context), "fallback"
