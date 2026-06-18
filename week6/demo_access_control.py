"""Week 6 demo: access-control guardrails end to end.

This is the only script that calls the Gemini API. It is deliberately small
because the free tier allows only ~20 requests per model per day and each agent
query makes about two LLM calls. The rate-limit and budget demos at the end use
no LLM at all, so they are instant and quota-free.

Usage:
    cd week6
    source ../week5/.venv/bin/activate   # or your own venv with google-genai
    python demo_access_control.py

What it shows:
    A. Same salary question, two roles -> redacted for engineer, full for HR.
    B. A normal policy question still works for an engineer (happy path).
    C. Rate-limit block (no LLM call).
    D. Budget block (no LLM call).
"""

import logging
import time

from app_starter import Agent, DATA_DIR

# Keep the report output clean: only warnings and errors from the agent.
logging.getLogger().setLevel(logging.WARNING)

# Free tier: 5 requests/minute, ~2 LLM calls per query, so pace LLM queries.
SECONDS_BETWEEN_QUERIES = 25

# A real employee so the salary question has something to redact.
EMPLOYEE = "Brian Yang"


def run_llm_query(agent, label, question, user_id, user_role):
    print("=" * 72)
    print(f"[{label}] role={user_role}  user_id={user_id}")
    print(f"Q: {question}")
    result = agent.query(question, user_id=user_id, user_role=user_role)
    if "error" in result:
        print(f"BLOCKED: {result['error']}")
    else:
        print(f"A: {result['answer']}")
        print(
            f"   tokens={result['tokens_used']}  cost=${result['cost']:.6f}  "
            f"budget_remaining=${result['budget_remaining']:.2f}"
        )
    print()


def main():
    agent = Agent(str(DATA_DIR / "techcorp.db"))
    print("Agent initialized with access control, rate limiting, cost enforcement\n")

    salary_q = f"What is {EMPLOYEE}'s salary?"

    # --- A. Role-based redaction: same question, different roles ---
    run_llm_query(agent, "A1 redacted", salary_q, "eng_user", "engineer")
    time.sleep(SECONDS_BETWEEN_QUERIES)
    run_llm_query(agent, "A2 full", salary_q, "hr_user", "hr")

    # --- B. Happy path: a non-sensitive policy question for an engineer ---
    time.sleep(SECONDS_BETWEEN_QUERIES)
    run_llm_query(
        agent, "B happy-path", "How many PTO days do employees get per year?",
        "eng_user", "engineer",
    )

    # --- C. Rate-limit block (no LLM call) ---
    print("=" * 72)
    print("[C rate-limit] exhausting the per-minute limit for user 'spammer'")
    for _ in range(agent.rate_limiter.max_queries_per_minute):
        agent.rate_limiter.is_allowed("spammer")
    blocked = agent.query("anything", user_id="spammer", user_role="engineer")
    print(f"Result: {blocked}\n")

    # --- D. Budget block (no LLM call) ---
    print("=" * 72)
    print("[D budget] engineer 'broke_user' has already spent their $100 budget")
    agent.cost_enforcer.add_cost("broke_user", "engineer", 100.0)
    blocked = agent.query("anything", user_id="broke_user", user_role="engineer")
    print(f"Result: {blocked}\n")

    # --- Audit trail summary ---
    print("=" * 72)
    log = agent.access_controller.get_audit_log()
    print(f"Audit log: {len(log)} entries. Last 5:")
    for entry in log[-5:]:
        print(f"  {entry}")

    print("\nAgent metrics:", agent.get_metrics())


if __name__ == "__main__":
    main()
