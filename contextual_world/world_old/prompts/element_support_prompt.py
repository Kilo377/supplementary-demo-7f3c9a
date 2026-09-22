from __future__ import annotations

import json


def build_agent_environment_element_support_prompt(
    *,
    agent_name: str,
    action_proposal: str,
    agent_state_for_support: dict,
    current_agent_facts: list[dict],
    permanent_elements: list[dict],
    temporary_elements: list[dict],
) -> str:
    agent_state_json = json.dumps(agent_state_for_support, ensure_ascii=False, indent=2)
    current_agent_facts_json = json.dumps(current_agent_facts, ensure_ascii=False, indent=2)
    permanent_elements_json = json.dumps(permanent_elements, ensure_ascii=False, indent=2)
    temporary_elements_json = json.dumps(temporary_elements, ensure_ascii=False, indent=2)

    return f"""你在模拟环境中负责判断：agent 的 action proposal 如果要真实发生，需要哪些环境元素支持。

agent 的 action proposal 只是“想做什么”，不是已经发生的事实。
你不生成最终动作反馈，不更新环境状态，也不写下一步动作。
你只判断这一小步动作需要什么元素参与，供后续环境模块真正执行。

环境里有两类元素：

permanent_element 是场景默认存在的陈设级元素。
例如沙发、办公桌、冰箱、花洒、书柜、收纳箱、门、椅子、微波炉。
它们通常不会被创建或销毁，但可以被打开、关闭、清洁、使用、承载、坐在上面，或作为临时对象的来源/挂靠点。

temporary_element 是运行中出现、取出、补全出来的临时对象。
例如食物、牛奶、书、文件、垃圾、抱枕、抹布、杯子里的水。
它们必须有来源或从属关系，通常依附于某个 permanent_element。
它们可以被拿起、放下、吃掉、喝掉、用完、丢弃、收纳，或者放到别的 permanent_element 上。
temporary_element 有生命周期。
temporary_element 只能是小型、可被移动、取用、消费、放置或临时持有的对象。
不要把大型固定设备、家具或电子大件创建成 temporary_element。
例如微波炉、烤箱、冰箱、洗衣机、电脑、电视、沙发、床、柜子、桌子、门、浴缸、花洒都不允许作为新建 temporary_element。
这类对象如果不在 permanent_elements 或 temporary_elements 中，就视为当前场景没有该对象，不要用低惊讶度补全把它造出来。

你要先判断：
如果这个 action proposal 要真实发生，是否需要一个可被建模的元素参与？

如果不需要，比如只是 agent 自身的身体动作、姿势变化、注视变化、身体表面变化，
或者泡沫、气味、热气、灰尘、水汽、光线、声音这类细节效果，那么它不需要元素支持。
这种情况 route=agent_body_action，support_kind=no_element。
这些细节可以放到 context_facts，但不要创建 temporary_element。

如果需要元素支持，再判断它依赖：
- 已有 permanent_element
- 已有 temporary_element
- 新建 temporary_element
- 多者混合

低惊讶度补全的意思是：
当前环境没有显式列出某个对象，但这个对象自然可以从某个 permanent_element 中出现，或者自然附着在某个 permanent_element 上。
例如从冰箱拿食物、从书柜拿书、整理办公桌上的散乱文件、从收纳区拿一个文件夹。
这种情况下可以创建 temporary_element，并指定它依附/来源于哪个 permanent_element。

不要因为环境没有穷举所有细节就机械拒绝合理动作。
但如果动作需要一个高惊讶度对象，或者没有任何合理 permanent_element 作为锚点，就应该 reject。
低惊讶度补全必须同时满足两点：
1. 新 temporary_element 必须挂靠到一个已有 permanent_element，填写 anchor_element_id。
2. 新 temporary_element 与 anchor_element_id 之间必须有足够直接的因果关系或容纳/来源关系。
合理例子：冰箱里的牛奶、锅里的食材、书柜里的书、柜子里的衣服、垃圾桶里的垃圾、洗衣篮里的衣物。
不合理例子：茶几上凭空出现笔记本、洗衣机旁凭空出现电脑、浴缸里凭空出现书。
不合理例子：厨房里没有微波炉时凭空生成微波炉、办公室里没有电脑时凭空生成电脑、卫生间里没有洗衣机时凭空生成洗衣机。
如果只能靠“可能有”来解释，而没有明确容器、来源、表面承载或场景常识关系，不要创建 temporary_element。
如果 action proposal 明确命名了目标对象，例如洗衣篮、微波炉、冰箱、书架、柜子，不要把它替换成另一个相近对象。
如果这个目标对象不在 permanent_elements 或 temporary_elements 中，但在当前场景里低惊讶度存在，可以创建同名 temporary_element。
例如卫生间里没有显式列出洗衣篮，但“把衣服放进洗衣篮”可以补全一个名为“洗衣篮”的 temporary_element；不要改成“放进洗衣机”。
但这个同名补全规则不适用于大型固定设备、家具或电子大件；例如 action proposal 提到微波炉，但 permanent_elements 和 temporary_elements 中没有微波炉，不要创建微波炉。

如果 action proposal 涉及已有 temporary_element，例如吃食物、喝牛奶、放下书、丢掉垃圾，请不要忽略它。
如果食物被吃掉、牛奶被喝掉、垃圾被丢弃，这类 temporary_element 应该返回生命周期更新。

只判断 action proposal 这一小步，不要推进到下一步。
只返回 JSON，不要解释推理过程。

返回格式：
{{
  "route": "agent_body_action | element_interaction | reject",
  "support_kind": "no_element | permanent_element | existing_temporary_element | new_temporary_element | mixed | reject",
  "reason": "一句简短中文理由",

  "permanent_targets": [
    {{
      "element_id": "已有 permanent_element id",
      "element_name": "元素名",
      "role": "这个 permanent_element 在动作中的作用"
    }}
  ],

  "temporary_targets": [
    {{
      "temporary_element_id": "已有 temporary_element id",
      "name": "临时元素名",
      "role": "这个 temporary_element 在动作中的作用"
    }}
  ],

  "temporary_element_creations": [
    {{
      "name": "需要新建的 temporary_element 名",
      "anchor_element_id": "它从属/来源/挂靠的 permanent_element id",
      "anchor_reason": "为什么这个 temporary_element 可以低惊讶度补全",
      "lifecycle": "game | until_consumed | until_disposed | until_used",
      "initial_status": "held | placed | in_use"
    }}
  ],

  "temporary_element_updates": [
    {{
      "temporary_element_id": "已有 temporary_element id",
      "new_status": "held | placed | in_use | consumed | disposed | discarded",
      "anchor_element_id": "如果位置或从属发生变化，填新的 permanent_element id，否则为空",
      "reason": "为什么生命周期或状态这样变化"
    }}
  ],

  "context_facts": [
    "不需要建模为 element 的环境/动作细节"
  ]
}}

例子：

action proposal: {agent_name}用泡沫揉搓身体。
判断：泡沫只是动作细节和身体表面状态，不需要创建 temporary_element。
route=agent_body_action，support_kind=no_element。

action proposal: {agent_name}打开花洒。
判断：需要和已有 permanent_element 花洒交互。
route=element_interaction，support_kind=permanent_element。

action proposal: {agent_name}从冰箱里拿出食物。
判断：冰箱是 permanent_element，食物可以作为低惊讶度 temporary_element 创建。
route=element_interaction，support_kind=new_temporary_element。

action proposal: {agent_name}吃掉手里的食物。
判断：如果食物已经是 temporary_element，就使用 existing_temporary_element，并更新为 consumed。

action proposal: {agent_name}把手里的书放到办公桌上。
判断：这同时涉及已有 temporary_element 书，以及 permanent_element 办公桌。
route=element_interaction，support_kind=mixed。

agent_state_for_support:
{agent_state_json}

current_agent_facts:
{current_agent_facts_json}

permanent_elements:
{permanent_elements_json}

temporary_elements:
{temporary_elements_json}

agent action proposal:
{action_proposal}
""".strip()
