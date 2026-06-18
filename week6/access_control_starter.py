"""
Week 6: Access Control, Rate Limiting & Cost Enforcement

Three guardrails that run BEFORE and AROUND the Week 5 agent:
1. AccessController - role-based document/field access control + audit log
2. RateLimiter - limit queries per minute per user
3. CostEnforcer - enforce monthly budget limits per role

Design notes (choices beyond the bare starter):
- redact_response keeps the field *name* visible and replaces only the value
  with [REDACTED], so a user can tell that something was withheld rather than
  silently disappearing.
- The agent also enforces access at retrieval time (field-level redaction in
  the employee tool, document-level filtering in the policy tool) so the LLM
  never even sees data the role is not allowed to read. redact_response then
  acts as a second, defense-in-depth layer on the final answer.
- CostEnforcer.can_afford_query takes an optional role, because a brand-new
  user has no spending record yet and therefore no budget to look up. This is
  the approach the starter hints at in its TODO.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TASK 1: AccessController
# ============================================================================


class AccessController:
    """Enforce role-based access control over documents and fields."""

    # Value patterns reused when redacting free-text answers.
    _MONEY = r"\$\s?\d[\d,]*(?:\.\d+)?"
    _SSN = r"\b\d{3}-\d{2}-\d{4}\b"

    def __init__(self, access_policy_path: str):
        """Load the access policy JSON and start an empty audit log."""
        try:
            with open(access_policy_path) as f:
                self.policy = json.load(f)
        except Exception as e:
            # Fail safe: with no policy loaded, every check below denies.
            logger.error(f"Failed to load access policy: {e}")
            self.policy = {}
        self.audit_log: List[Dict[str, Any]] = []

    def can_view_document(self, role: str, document: Dict[str, Any]) -> bool:
        """Check if a role may view a document given its sensitivity level."""
        sensitivity = document.get("sensitivity")
        access = self.policy.get("document_access", {})
        # Unknown / missing sensitivity is treated as not-allowed (fail safe).
        if sensitivity not in access:
            return False
        return role in access[sensitivity]

    def can_view_field(self, role: str, field_name: str) -> bool:
        """Check if a role may view a field.

        A field that is not listed in sensitive_fields is not sensitive, so it
        is visible to everyone. A listed field is visible only to the roles in
        its visibility list.
        """
        sensitive = self.policy.get("sensitive_fields", {})
        if field_name not in sensitive:
            return True
        return role in sensitive[field_name].get("visibility", [])

    def redact_response(self, role: str, response: str) -> str:
        """Redact the values of sensitive fields the role cannot view.

        The field name is kept and only the value is replaced with [REDACTED],
        so the reader can see that something was withheld.
        """
        if not response:
            return response

        sensitive = self.policy.get("sensitive_fields", {})
        redacted = response
        for field, meta in sensitive.items():
            if self.can_view_field(role, field):
                continue

            # Keywords that may introduce the field in prose.
            keywords = {field, meta.get("description", "")}
            if field == "ssn":
                keywords.add("social security")
            if field == "compensation":
                keywords.update({"bonus", "stock", "stock options", "equity"})
            if field == "performance_review":
                keywords.update({"performance review", "performance rating", "rating"})
            keywords = [re.escape(k) for k in keywords if k and len(k) > 2]

            value = self._SSN if field == "ssn" else self._MONEY

            for kw in keywords:
                # "<keyword> ... <value>"  -> keep keyword, redact the value.
                pattern = rf"(?i)({kw}[^\n]{{0,30}}?)({value})"
                redacted = re.sub(pattern, r"\1[REDACTED]", redacted)

            # SSNs are unambiguous, so scrub any that remain regardless of
            # surrounding text.
            if field == "ssn":
                redacted = re.sub(self._SSN, "[REDACTED]", redacted)

            if redacted != response:
                self.log_access(role, "response", allowed=False, field=field)

        return redacted

    def log_access(
        self, role: str, resource: str, allowed: bool, field: str = None
    ) -> None:
        """Append an access attempt to the audit log."""
        self.audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "role": role,
                "resource": resource,
                "field": field,
                "allowed": allowed,
            }
        )
        # Keep the log bounded so a long-running process does not grow forever.
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]

    def filter_documents(
        self, role: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return only the documents a role may view, logging each decision."""
        visible = []
        for doc in documents:
            allowed = self.can_view_document(role, doc)
            resource = doc.get("id") or doc.get("title") or "document"
            self.log_access(role, resource, allowed)
            if allowed:
                visible.append(doc)
        return visible

    def redact_employee_fields(
        self, role: str, employee: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Redact sensitive columns of a structured employee record.

        Used at retrieval time so the LLM never sees values the role cannot
        read. DB columns are mapped onto the policy's sensitive_fields.
        """
        # DB column -> policy sensitive_field key.
        column_to_field = {
            "salary": "salary",
            "ssn": "ssn",
            "address": "address",
            "stock_options": "compensation",
            "bonus_eligible": "compensation",
        }
        safe = dict(employee)
        for column, field in column_to_field.items():
            if column in safe and not self.can_view_field(role, field):
                safe[column] = "[REDACTED]"
                self.log_access(role, f"employee.{column}", allowed=False, field=field)
        return safe

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log entries."""
        return self.audit_log


# ============================================================================
# TASK 2: RateLimiter
# ============================================================================


class RateLimiter:
    """Rate limit queries per user using a sliding 60-second window."""

    def __init__(self, max_queries_per_minute: int = 30):
        self.max_queries_per_minute = max_queries_per_minute
        self.user_query_times: Dict[str, List[float]] = {}

    def _recent(self, user_id: str) -> List[float]:
        """Return (and store back) only this user's timestamps within 60s."""
        cutoff = time() - 60
        recent = [t for t in self.user_query_times.get(user_id, []) if t > cutoff]
        self.user_query_times[user_id] = recent
        return recent

    def is_allowed(self, user_id: str) -> bool:
        """Allow and record a query if under the limit, else deny."""
        recent = self._recent(user_id)
        if len(recent) < self.max_queries_per_minute:
            recent.append(time())
            return True
        return False

    def get_remaining_queries(self, user_id: str) -> int:
        """Queries the user can still make in the current minute."""
        recent = self._recent(user_id)
        return max(0, self.max_queries_per_minute - len(recent))


# ============================================================================
# TASK 3: CostEnforcer
# ============================================================================


class CostEnforcer:
    """Enforce monthly budget limits per role."""

    DEFAULT_BUDGETS = {
        "engineer": 100.0,
        "manager": 500.0,
        "hr": 200.0,
        "finance": 500.0,
        "executive": 1000.0,
    }

    def __init__(self, policy_path: str = None):
        self.role_budgets = dict(self.DEFAULT_BUDGETS)
        if policy_path:
            try:
                with open(policy_path) as f:
                    self.role_budgets.update(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load budget policy: {e}")
        # {user_id: {"role": str, "total": float}}
        self.user_spending: Dict[str, Dict[str, Any]] = {}

    def add_cost(self, user_id: str, role: str, cost: float) -> None:
        """Record cost against a user's running total."""
        entry = self.user_spending.setdefault(user_id, {"role": role, "total": 0.0})
        entry["role"] = role  # keep the latest role on record
        entry["total"] += cost

    def _budget_for(self, user_id: str, role: Optional[str]) -> float:
        """Resolve a user's monthly budget from their recorded or passed role."""
        recorded = self.user_spending.get(user_id, {}).get("role")
        resolved = recorded or role or "engineer"
        return self.role_budgets.get(resolved, 0.0)

    def can_afford_query(
        self, user_id: str, estimated_cost: float, role: str = None
    ) -> bool:
        """True if the estimated cost fits in the user's remaining budget.

        A brand-new user has no spending record, so their role cannot be
        looked up; pass ``role`` for them (the agent supplies user_role). If
        neither is available we fall back to the smallest budget (engineer).
        """
        budget = self._budget_for(user_id, role)
        spent = self.user_spending.get(user_id, {}).get("total", 0.0)
        return estimated_cost <= (budget - spent)

    def get_budget_remaining(self, user_id: str, role: str = None) -> float:
        """Remaining budget for a user (never negative)."""
        budget = self._budget_for(user_id, role)
        spent = self.user_spending.get(user_id, {}).get("total", 0.0)
        return max(0.0, budget - spent)


# ============================================================================
# TASK 5: Tests (run with: python3 access_control_starter.py)
# ============================================================================

if __name__ == "__main__":
    """Quick test of access control functionality."""

    # Test AccessController
    print("Testing AccessController...")
    controller = AccessController("data/access_control.json")

    assert not controller.can_view_field(
        "engineer", "salary"
    ), "Engineer should not see salary"
    assert controller.can_view_field("hr", "salary"), "HR should see salary"
    assert controller.can_view_field("manager", "salary"), "Manager should see salary"
    assert not controller.can_view_field(
        "engineer", "ssn"
    ), "Engineer should not see SSN"
    print("  can_view_field: PASSED")

    docs = [
        {"id": "doc1", "sensitivity": "Public", "content": "Mission statement"},
        {"id": "doc2", "sensitivity": "Confidential", "content": "Salary ranges"},
    ]
    visible = controller.filter_documents("engineer", docs)
    assert (
        len(visible) == 1 and visible[0]["id"] == "doc1"
    ), "Engineer should only see Public doc"
    print("  filter_documents: PASSED")

    # redact_response: value gone, field name kept (TROUBLESHOOTING example)
    redacted = controller.redact_response("engineer", "Employee salary: $100,000")
    assert "salary" in redacted.lower() and "$100,000" not in redacted, (
        f"Salary value should be redacted, got: {redacted}"
    )
    assert controller.redact_response("hr", "Employee salary: $100,000") == (
        "Employee salary: $100,000"
    ), "HR answer should not be redacted"
    print("  redact_response: PASSED")

    # Test RateLimiter
    print("\nTesting RateLimiter...")
    limiter = RateLimiter(max_queries_per_minute=3)
    assert limiter.is_allowed("user1"), "First query should be allowed"
    assert limiter.is_allowed("user1"), "Second query should be allowed"
    assert limiter.is_allowed("user1"), "Third query should be allowed"
    assert not limiter.is_allowed("user1"), "Fourth query should be blocked"
    assert limiter.get_remaining_queries("user1") == 0, "No queries should remain"
    print("  is_allowed: PASSED")

    # Test CostEnforcer
    print("\nTesting CostEnforcer...")
    enforcer = CostEnforcer()
    assert enforcer.can_afford_query(
        "user1", 50.0
    ), "Should afford $50 within $100 budget"
    enforcer.add_cost("user1", "engineer", 50.0)
    assert enforcer.can_afford_query(
        "user1", 49.0
    ), "Should afford $49 with $50 remaining"
    assert not enforcer.can_afford_query(
        "user1", 51.0
    ), "Should not afford $51 with $50 remaining"
    assert enforcer.get_budget_remaining("user1") == 50.0, "Should have $50 remaining"
    print("  can_afford_query: PASSED")

    print("\nAll tests passed!")
