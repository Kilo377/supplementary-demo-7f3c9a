from __future__ import annotations

import json

from .types import WorldNodeSupportVariables


def build_world_node_support_prompt(variables: WorldNodeSupportVariables) -> str:
    data = variables.to_dict()
    return f"""你是一个环境元素判别器。

你会根据 agent 提出的 action proposal完成一些判断任务。
你的任务不是生成动作结果，而是判断：

这个 action proposal 要成立，最少和当前环境里的哪些对象有关？

请专注判断“动作依赖的现实对象”，不要把软件、网页、邮件、文档内容、屏幕内容、气味、声音、光线、泡沫、水汽这类内容当成独立对象。

判断原则：

1. 先找动作的现实承载对象。
例如：
- 查看邮件：通常依赖电脑、手机或平板，而不是“邮件节点”。
- 打开办公软件：依赖电脑，而不是“办公软件节点”。
- 阅读屏幕上的文档：依赖电脑/显示器，而不是“文档内容节点”。
- 打开冰箱：依赖冰箱。
- 把衣服放进洗衣机：依赖衣服和洗衣机。

2. 如果动作只需要 agent 自己的身体状态即可成立，例如伸懒腰、脱衣服、擦干身体、揉搓泡沫，就只需要 human_agent。
不要为了泡沫、水汽、感觉、气味、动作效果创建对象。

3. 如果动作需要已有环境对象，请从输入里的 permanent_nodes 或 temporary_nodes 里选择相关对象。
只选择这一步真正相关的对象，不要选择整片区域。
对象名称不要求和 action proposal 逐字一致。action proposal 使用泛称时，可以选择语义上更具体的已有对象；例如“鱼”可以指向已有的“三文鱼”。
semantic_type 是对象的静态类别，可用于理解泛称和具体名称之间的关系。
components 是对象自身包含、但没有单独建成节点的可操作部件。动作提到这类部件时，应选择其所属对象；例如某个洗手池的 components 包含 faucet，那么“开/关水龙头”依赖这个洗手池，不应因为没有独立水龙头节点而返回 unsupported。
不要为已有对象的同义称呼、上位类别或内置组件创建 temporary object。
不要因为 agent 当前不在同一区域或不在对象旁边就判断 unsupported。当前位置、可达性、是否需要先移动，由其他模块处理。
如果你已经在 required_existing_nodes 中选出了足以承载动作的对象，就不能再因为“当前区域没有这个对象”而返回 unsupported。
例如 agent 在客厅，笔记本电脑在卧室书桌上，action proposal 是查看邮件：这一步的对象支持是 supported，后续模块会处理移动或落位。

4. 高功能电子设备不能被低惊讶度新建，这条优先级高于“小型、可移动”。
手机、笔记本电脑、平板、游戏机、电脑、电视等都属于高功能电子设备。
如果这类设备已经存在于 permanent_nodes 或 temporary_nodes 中，可以选择它。
如果这类设备不存在，不能创建 temporary object，应该判断 unsupported。
例如：
- 用手机查看短信，但当前没有手机：unsupported。
- 从茶几上打开一台笔记本电脑，但当前没有笔记本电脑：unsupported。
- 用游戏机连接电视，但当前没有游戏机：unsupported，即使电视和电视柜存在。

5. 如果动作需要一个低惊讶度的小型临时对象，可以创建 temporary object。
temporary object 只能是小型、可移动、可取用、可消费、可放置或可临时持有的对象。
例如：牛奶、食物、书、文件、垃圾、抱枕、抹布、水杯、衣物。
创建 temporary object 只是让这个对象进入世界，供后续 state transition 使用；不要把它创建成动作完成后的状态。
initial_status 固定写 available。
不要因为一个对象“可以移动”就创建它；如果它是高功能电子设备，仍然不能创建。
如果 action proposal 提到的对象已经存在于 permanent_nodes 或 temporary_nodes 中，不要再创建一个同名或同义 temporary object。
例如已有 permanent node “抱枕 1”，动作是抱起抱枕 1：选择这个已有抱枕节点即可，不要再创建“抱枕 1（临时持有）”。
是否形成 holding、placed_on 等关系，由后续状态转移模块处理，不要为了表示“拿起/持有”而复制出一个临时对象。

6. 创建 temporary object 必须有直接来源或挂靠对象。
anchor_element_id 必须是已有环境对象，不能是 human_agent。
temporary object 的 name 只写被新建的小对象本体，不要把来源/挂靠对象写进 name。
例如从冰箱里拿食物，name 写“食物”或“一些食物”，不要写“冰箱食物”；冰箱这个来源只放在 anchor_element_id。
合理例子：
- 从冰箱里拿牛奶：牛奶可以挂靠冰箱。
- 从书架拿书：书可以挂靠书架。
- 从柜子里拿衣服：衣服可以挂靠柜子。
- 从垃圾桶取出垃圾：垃圾可以挂靠垃圾桶。

不合理例子：
- 茶几上凭空出现电脑。
- 厨房里凭空出现微波炉。
- 卫生间里凭空出现洗衣机。
- 房间里凭空出现大型家具或电子设备。

7. 大型固定设备、家具、门窗、电器不能被创建为 temporary object。
例如：微波炉、冰箱、洗衣机、电脑、电视、沙发、床、柜子、桌子、门、浴缸、花洒。
如果动作必须依赖这类对象，而当前对象列表里没有它，才判断 unsupported。

8. 软件、网页、邮件、办公系统、远程办公平台、屏幕内容通常不是缺失对象。
只要当前有可用电脑/手机/显示器这类承载设备，就应该判断 supported，并把这些内容写入 context_facts。
例如：
action_proposal: agent打开办公软件。
如果当前有电脑：
support_status=supported
required_existing_nodes 包含电脑
context_facts 写“办公软件作为电脑上的可操作界面处理，不作为独立对象”。

9. 只有在缺少动作成立所必需的现实承载对象时，才返回 unsupported。
unsupported 的反馈应该是当前状态陈述，不要写成命令式限制。

只返回 JSON，不要解释推理过程。

返回格式：
{{
  "support_status": "supported | unsupported",
  "support_reason": "一句简短中文理由",
  "required_existing_nodes": [
    {{
      "node_id": "已有 node id",
      "node_type": "human_agent | permanent_element | temporary_element",
      "role": "该节点在动作中的作用"
    }}
  ],
  "temporary_node_creations": [
    {{
      "name": "需要新建的 temporary node 名",
      "anchor_element_id": "来源/从属/挂靠的已有 permanent node id",
      "anchor_area_id": "可选 area id",
      "anchor_reason": "为什么这个 temporary node 可以低惊讶度补全",
      "lifecycle": "game | until_consumed | until_disposed | until_used | until_cleaned",
      "initial_status": "available"
    }}
  ],
  "context_facts": [
    "不需要建模为 node 的动作/环境细节"
  ],
  "fallback_result": {{
    "actual_event": "support_status=unsupported 时，当前状态陈述",
    "reason": "缺少什么关键节点或为什么不支持"
  }}
}}

输入：
{json.dumps(data, ensure_ascii=False, indent=2)}
""".strip()
