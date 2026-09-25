"""已被取代的历代编排机制。

这里的代码不在当前执行路径上：每一代的开关都默认关闭，且没有任何
benchmark adapter 或 launch 脚本会打开它们。放在这里是为了让主干只
呈现实际运行的逻辑，同时保留复活的可能。

各代的配置字段仍在 core.py 的 _Archived*Config 基类中。
"""

from nanoma.legacy.fixed_topology import FixedTopologyMixin
from nanoma.legacy.supervisor import SupervisorMixin

__all__ = ["FixedTopologyMixin", "SupervisorMixin"]
