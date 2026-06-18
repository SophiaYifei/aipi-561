"""
Week 6 offline tests for the access-control guardrails.

These tests exercise every guardrail WITHOUT calling the Gemini API, so they
can be run freely without touching the free-tier daily quota. The only LLM
calls in the project happen in demo_access_control.py.

Run with: python3 test_access_control.py
"""

import json
import sqlite3

from access_control_starter import AccessController, RateLimiter, CostEnforcer
from app_starter import Agent, DATA_DIR


def _sample_employee_name() -> str:
    """Grab a real employee name from the DB for the redaction tests."""
    conn = sqlite3.connect(str(DATA_DIR / "techcorp.db"))
    name = conn.execute("SELECT name FROM employees LIMIT 1").fetchone()[0]
    conn.close()
    return name


def test_access_controller():
    print("AccessController")
    ac = AccessController(str(DATA_DIR / "access_control.json"))

    # Field-level visibility.
    assert ac.can_view_field("hr", "salary")
    assert ac.can_view_field("manager", "salary")
    assert not ac.can_view_field("engineer", "salary")
    assert not ac.can_view_field("engineer", "ssn")
    assert ac.can_view_field("finance", "ssn")
    # A non-sensitive field is visible to everyone.
    assert ac.can_view_field("engineer", "title")
    print("  field visibility: PASSED")

    # Document-level visibility by sensitivity.
    pub = {"id": "p", "sensitivity": "Public"}
    internal = {"id": "i", "sensitivity": "Internal"}
    conf = {"id": "c", "sensitivity": "Confidential"}
    restricted = {"id": "r", "sensitivity": "Restricted"}
    assert ac.can_view_document("engineer", pub)
    assert ac.can_view_document("engineer", internal)
    assert not ac.can_view_document("engineer", conf)
    assert not ac.can_view_document("engineer", restricted)
    assert ac.can_view_document("manager", conf)
    assert ac.can_view_document("hr", restricted)
    # Unknown sensitivity fails safe (deny).
    assert not ac.can_view_document("executive", {"id": "x"})
    print("  document visibility: PASSED")

    # filter_documents keeps only viewable docs and logs every decision.
    before = len(ac.get_audit_log())
    docs = [pub, internal, conf, restricted]
    visible = ac.filter_documents("engineer", docs)
    assert {d["id"] for d in visible} == {"p", "i"}
    assert len(ac.get_audit_log()) == before + 4  # one log per document
    print("  filter_documents + audit: PASSED")

    # redact_response: value gone, field name kept; allowed role untouched.
    eng = ac.redact_response("engineer", "Her salary is $467,621 and SSN 115-04-4507.")
    assert "$467,621" not in eng and "115-04-4507" not in eng
    assert "salary" in eng.lower()
    hr = ac.redact_response("hr", "Her salary is $467,621 and SSN 115-04-4507.")
    assert "$467,621" in hr and "115-04-4507" in hr
    print("  redact_response: PASSED")

    # Structured employee redaction (retrieval-time, field level).
    emp = {"name": "X", "title": "Eng", "salary": 467621, "ssn": "115-04-4507"}
    safe = ac.redact_employee_fields("engineer", emp)
    assert safe["salary"] == "[REDACTED]" and safe["ssn"] == "[REDACTED]"
    assert safe["name"] == "X" and safe["title"] == "Eng"
    full = ac.redact_employee_fields("hr", emp)
    assert full["salary"] == 467621 and full["ssn"] == "115-04-4507"
    print("  redact_employee_fields: PASSED")


def test_rate_limiter():
    print("RateLimiter")
    rl = RateLimiter(max_queries_per_minute=3)
    assert all(rl.is_allowed("u") for _ in range(3))
    assert not rl.is_allowed("u")
    assert rl.get_remaining_queries("u") == 0
    # Different users have independent budgets.
    assert rl.is_allowed("other")
    print("  per-user limit: PASSED")


def test_cost_enforcer():
    print("CostEnforcer")
    ce = CostEnforcer()
    # New user, role passed in: engineer budget = $100.
    assert ce.can_afford_query("e", 50.0, role="engineer")
    ce.add_cost("e", "engineer", 90.0)
    assert ce.can_afford_query("e", 10.0)
    assert not ce.can_afford_query("e", 11.0)
    assert ce.get_budget_remaining("e") == 10.0
    # Executive has a far larger budget.
    assert ce.can_afford_query("x", 900.0, role="executive")
    print("  budget tracking: PASSED")


def test_tool_level_enforcement():
    """The data tools enforce access at retrieval time, no LLM involved."""
    print("Tool-level enforcement (retrieval time)")
    agent = Agent(str(DATA_DIR / "techcorp.db"))
    name = _sample_employee_name()

    emp_tool = agent.tools["employee_lookup"]
    emp_tool.role = "engineer"
    eng = json.loads(emp_tool.execute(employee_name=name))["employees"][0]
    assert eng["salary"] == "[REDACTED]" and eng["ssn"] == "[REDACTED]", eng

    emp_tool.role = "hr"
    hr = json.loads(emp_tool.execute(employee_name=name))["employees"][0]
    # HR may see salary and ssn; compensation (stock/bonus) stays redacted.
    assert hr["salary"] != "[REDACTED]" and hr["ssn"] != "[REDACTED]", hr
    assert hr["stock_options"] == "[REDACTED]", "HR is not in compensation visibility"
    print("  employee field redaction by role: PASSED")

    pol_tool = agent.tools["policy_search"]
    pol_tool.role = "engineer"
    eng_docs = agent.access_controller.filter_documents("engineer", pol_tool.documents)
    mgr_docs = agent.access_controller.filter_documents("manager", pol_tool.documents)
    assert len(eng_docs) < len(mgr_docs), "Engineer should see fewer documents"
    print(f"  document filtering by role: PASSED "
          f"(engineer {len(eng_docs)} vs manager {len(mgr_docs)} of {len(pol_tool.documents)})")


def test_agent_guardrails_block_before_llm():
    """Rate-limit and budget blocks must return BEFORE any LLM call."""
    print("Agent guardrails (block before LLM)")
    agent = Agent(str(DATA_DIR / "techcorp.db"))

    # Exhaust the rate limiter for this user, then query -> blocked, no LLM.
    for _ in range(agent.rate_limiter.max_queries_per_minute):
        agent.rate_limiter.is_allowed("rl_user")
    blocked = agent.query("anything", user_id="rl_user", user_role="engineer")
    assert blocked.get("error") == "Rate limit exceeded", blocked
    print("  rate-limit block: PASSED")

    # Exhaust the budget for a different user, then query -> blocked, no LLM.
    agent.cost_enforcer.add_cost("budget_user", "engineer", 100.0)
    blocked = agent.query("anything", user_id="budget_user", user_role="engineer")
    assert blocked.get("error") == "Budget exceeded", blocked
    print("  budget block: PASSED")


if __name__ == "__main__":
    test_access_controller()
    test_rate_limiter()
    test_cost_enforcer()
    test_tool_level_enforcement()
    test_agent_guardrails_block_before_llm()
    print("\nAll offline access-control tests passed!")
