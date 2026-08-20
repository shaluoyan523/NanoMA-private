# Topology references

These prompts describe 30 macro-level multi-agent patterns from the original
direct-spawn NanoMA interface. In the current general agent they are conceptual
references, not literal tool scripts: when a prompt says `spawn`, the node
should reach a `task_create` planning point instead. The runtime then asks that
same node model whether to delegate and determines the concrete child split.

Use `examples/run_preset.py` for this compatibility translation. New tasks
normally do not need a preset; node-autonomous planning can form the topology
directly from the task and current context.
