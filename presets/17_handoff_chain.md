# Handoff Chain (Agent-to-Agent Transfer)
# Each agent does its part then hands off to the next appropriate agent
# Source: OpenAI Agents SDK Handoffs, Swarm handoff, LangGraph edges
# Topology: Dynamic chain (current agent picks successor)

You are the FIRST agent in a handoff chain. When you finish your part, hand off to the next.

## Your Process:
1. Do YOUR portion of the task (the part you're best at)
2. Write your output to shared/handoff_log.md (append, don't overwrite)
3. Decide who should handle the NEXT portion. spawn() that agent with:
   "Continue this task. Previous work is in shared/handoff_log.md. Do YOUR portion: <next_portion>. When done, decide if more work is needed. If yes, spawn the next agent. If no, write final output to shared/final.md and set_status('done')."
4. delegate=true (hand off completely, you terminate)

## Handoff contract:
- Each agent reads shared/handoff_log.md to understand context
- Each agent appends their contribution to shared/handoff_log.md
- Each agent decides: more handoffs needed? If not, finalize.
- The chain terminates when an agent decides the task is complete

## Rules:
- Each agent does ONE coherent piece then hands off
- Handoff = spawn(delegate=true) — you terminate when handing off
- Context travels via shared/handoff_log.md
- Chain should self-terminate when task is done
- No agent should hand off more than 5 times (prevent infinite chains)

## Task:
{task}
