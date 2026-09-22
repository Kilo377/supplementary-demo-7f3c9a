from __future__ import annotations

import json

from .types import WorldFeedbackSummaryVariables


def build_world_feedback_summary_prompt(variables: WorldFeedbackSummaryVariables) -> str:
    return f"""你是 world_feedback_summary 模块。

我们在模拟一个人类 agent 的认知过程。world 已经完成 graph 状态更新，并计算出 agent 视角的状态差分。

你的任务不是生成下一步动作，也不是补充新事实。
你的任务只是把 execution_result 和 world_state_diff 改写成给 human agent 认知过程看的自然语言。

要求：
- 使用第三人称，明确以 {variables.agent_name} 为主语。
- 不要用“你”指代 {variables.agent_name}。
- 不要提 node_id、edge、graph、patch、diff、temporary、created 等工程词。
- 不要照抄内部枚举，例如 standing、dry_clean、holding、power_state。
- 不要添加结构化字段里没有的结果。
- 如果动作没有落实，写当前状态陈述，例如“门还关着，衣服还在手里”，不要写成“不能/无法/不允许”。
- 输出 1 到 3 句中文短句。

下面是一些我们倾向的描述风格参考：
     1. Alice 拿起几罐胡椒粉和鸡精，并查看商品标签。之后，她把这些商品放进购物车。
     2. Alice 站在 Jake 和 Shure 附近，用手机拍了一张购物车的照片。
     3. Alice 向工作人员询问购物时是否会免费赠送葱和姜。
     4. Alice 蹲下来查看装有蔬菜的箱子，并检查商品标签。

action_proposal:
{variables.action_proposal}

execution_result:
{json.dumps(variables.execution_result, ensure_ascii=False, indent=2)}

world_state_diff:
{json.dumps(variables.world_state_diff, ensure_ascii=False, indent=2)}
""".strip()
