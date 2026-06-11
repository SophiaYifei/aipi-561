"""Run the 10 test queries for the Week 5 report.

Usage:
    cd week5
    source .venv/bin/activate
    python demo_queries.py
"""

import logging
import time

from app_starter import Agent, DATA_DIR

# Keep the report output clean: only warnings and errors from the agent.
logging.getLogger().setLevel(logging.WARNING)

# The free tier allows 5 requests/minute and each agent query makes ~2 LLM
# calls, so we pause between queries to stay under the limit.
SECONDS_BETWEEN_QUERIES = 25

QUERIES = [
    # 1-2: policy retrieval (policy_search)
    "What is the travel policy?",
    "How many PTO days do employees get per year?",
    # 3-4: database lookups (employee_lookup, by name and by id)
    "Who is Brian Yang and what is his job title?",
    "Which department does employee with ID 42 work in?",
    # 5-6: expense limits (expense_query, incl. mapping user wording to roles)
    "What is the expense approval limit for a manager?",
    "I am a senior engineer at IC3 level. How much can I approve?",
    # 7: multi-tool question (employee_lookup + expense_query)
    "What is the job title of employee 1, and what expense limit does a vp have?",
    # 8-10: failure handling and scope
    "Look up the employee named Zebulon Quixote.",
    "What is the expense approval limit for an intern?",
    "What is the weather in Paris today?",
]


def main():
    agent = Agent(str(DATA_DIR / "techcorp.db"))
    print("Agent initialized successfully\n")

    for i, q in enumerate(QUERIES, start=1):
        if i > 1:
            time.sleep(SECONDS_BETWEEN_QUERIES)
        print("=" * 70)
        print(f"Query {i}: {q}")
        result = agent.query(q)
        print(f"Answer: {result['answer']}")
        print(f"Tokens: {result['tokens_used']}   Cost: ${result['cost']:.6f}\n")

    print("=" * 70)
    metrics = agent.get_metrics()
    print("Final metrics:")
    print(f"  total_queries:      {metrics['total_queries']}")
    print(f"  total_tokens:       {metrics['total_tokens']}")
    print(f"  total_cost:         ${metrics['total_cost']:.6f}")
    print(f"  avg_cost_per_query: ${metrics['avg_cost_per_query']:.6f}")


if __name__ == "__main__":
    main()
