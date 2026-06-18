# Week 6 Report: Access Control and Monitoring

## 1. Overview

I added four guardrails around my Week 5 TechCorp agent: role-based access
control, rate limiting, cost enforcement, and an audit log. The guardrail
classes are in `access_control_starter.py`, and I integrated them into a copy of
my Week 5 `app_starter.py`. Everything lives in `week6/`. The rule I followed is
to block a request before it reaches the LLM, not after.

## 2. The three guardrail classes

- **AccessController** (loads `data/access_control.json`): `can_view_field` and
  `can_view_document` check role permissions; `redact_response`,
  `filter_documents`, and `redact_employee_fields` remove data a role cannot
  see; every decision is written to an audit log. Unknown sensitivity is denied
  (fail-safe).
- **RateLimiter**: a sliding 60-second window per user, set to 30 queries per
  minute.
- **CostEnforcer**: a monthly budget per role (engineer \$100, manager \$500,
  hr \$200, finance \$500, executive \$1000). I added a `role` parameter to
  `can_afford_query` so a new user with no spending record still has a budget to
  check against.

## 3. Integration with the Week 5 agent

- `query()` now takes `user_id` (for rate and cost tracking) and `user_role`
  (for access control).
- The rate limit and budget are checked first, so a blocked request never calls
  the LLM.
- Access control runs at retrieval time: the employee tool redacts sensitive
  columns and the policy tool drops documents the role cannot read, so the model
  never sees restricted data. `redact_response` on the final answer is a second
  layer. Each lookup and query also writes an audit entry.

## 4. Offline tests

`test_access_control.py` covers every guardrail without calling the LLM, and all
pass. One concrete result: an engineer can retrieve 21 of the 74 policy
documents (the Public and Internal ones), while a manager can retrieve all 74.

![](image/REPORT/1781760246264.png)

The original starter self-test also passes:

![](image/REPORT/1781760180662.png)

## 5. Live demo

`demo_access_control.py` is the only script that calls Gemini.

Same salary question, two roles: the engineer gets a redacted answer, HR gets
the figure.

![](image/REPORT/1781822029680.png)

A normal policy question still works for an engineer:

![](image/REPORT/1781822079240.png)

The rate-limit and budget blocks return an error with no LLM call, followed by
the tail of the audit log:

![](image/REPORT/1781822095986.png)
