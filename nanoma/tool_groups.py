"""工具名分组：按语义把工具名聚成集合，供工具策略与各子系统判定使用。

放在独立模块是为了让依赖方向单向：core 与从 core 拆出的子系统模块
（merge_submit、legacy/*）都从这里取，谁也不必反向 import core。
"""

from __future__ import annotations

_CREATE_TOOLS = {"spawn", "spawn_many"}
_COORDINATION_TOOLS = {"send", "deliver_to_parent", "wait", "query", "kill", "transfer", "set_bio"}
_LIFECYCLE_TOOLS = {"get_cost", "set_status", "rebirth", "submit"}
_SHELL_TOOLS = {"shell", "tb_shell"}
_DELIVERY_READ_TOOLS = {
    "ws_read_file", "ws_grep", "ws_code_outline", "ws_read_symbol",
    "tb_read_file", "get_task_context",
}
_DELIVERY_WRITE_TOOLS = {
    "ws_create_file", "ws_append_file", "ws_replace_string",
    "ws_multi_replace", "ws_apply_patch", "tb_write_file",
}
_READ_TOOLS = _DELIVERY_READ_TOOLS | {"query", "get_cost"} | _SHELL_TOOLS
_WORK_TOOLS = _DELIVERY_WRITE_TOOLS | _SHELL_TOOLS | {"batch", "submit"}
_FINISH_TOOLS = {"set_status", "submit", "tb_write_file", "ws_create_file", "ws_append_file"}
