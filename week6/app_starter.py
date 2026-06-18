"""
Week 5 + 6: Agent Architecture with Access-Control Guardrails

Build an AI agent that answers TechCorp questions using:
- Gemini Flash LLM (free tier via Google AI API)
- SQLite database queries
- Policy document retrieval

Design notes (changes from the starter template):
- Model is gemini-3.5-flash (free tier), following the instructor's update.
- Tool calls use the Gemini native function-calling API instead of parsing
  "TOOL: / ARGS:" text. The model returns a structured FunctionCall object,
  so there is no regex parsing that can silently break, and the model cannot
  invent tools that are not declared.
- The reasoning loop is multi-step: the model may call several tools in a
  row (up to MAX_TOOL_STEPS) before giving the final answer.

Week 6 additions (guardrails that run before and around the agent):
- RateLimiter and CostEnforcer are checked at the start of query(), before any
  LLM call, so an over-quota user is blocked without spending tokens.
- AccessController is enforced in two places (defense in depth):
    * at retrieval time - the employee tool redacts sensitive columns and the
      policy tool drops documents the role cannot see, so the LLM never reads
      data the user is not allowed to access (the READING.md "oversharing"
      case study);
    * on the final answer - redact_response scrubs any sensitive value that
      slipped through.
- query() now takes user_id (for rate/cost tracking) and user_role (for
  access control), and every access decision is written to the audit log.
"""

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any
import google.genai as genai
from google.genai import errors as genai_errors
from google.genai import types
import logging
import os

from access_control_starter import AccessController, RateLimiter, CostEnforcer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _load_env_file() -> None:
    """Load week5/.env into os.environ without extra dependencies.

    Existing environment variables are not overwritten.
    """
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Default model follows the instructor's recommendation for free-tier use
# (gemini-3.5-flash or gemini-2.5-flash). Override with the GEMINI_MODEL env
# var if needed. Note: free-tier daily quotas are counted per model.
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Safety limit for the reasoning loop so the agent cannot call tools forever.
MAX_TOOL_STEPS = 5

# Retry settings for free-tier rate limits. On a transient API error (429
# rate limit, 503 overloaded) we back off and retry instead of failing.
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 10


# TASK 1: The Tool base class


class Tool:
    """Base class for tools the agent can call."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        """Execute the tool. Each subclass must override this method."""
        raise NotImplementedError


# TASK 2: EmployeeLookupTool


class EmployeeLookupTool(Tool):
    """Look up employee information from SQLite database."""

    MAX_ROWS = 10  # cap the rows fed back to the LLM to keep the prompt small

    def __init__(self, db_path: str, access_controller: "AccessController" = None):
        super().__init__("employee_lookup", "Find employee information by name or ID")
        self.db_path = db_path
        # Week 6: enforce field-level access at retrieval time. The agent sets
        # self.role before each query so the tool knows whose permissions apply.
        self.access_controller = access_controller
        self.role = None

    def execute(self, employee_name: str = None, employee_id: str = None) -> str:
        """Look up employee by name (partial match) or ID (exact match).

        Returns:
            JSON string with employee info or error message
        """
        try:
            if employee_id is None and not employee_name:
                return "Error: provide employee_name or employee_id"

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if employee_id is not None:
                cursor.execute(
                    "SELECT * FROM employees WHERE id = ?", (employee_id,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM employees WHERE name LIKE ? LIMIT ?",
                    (f"%{employee_name}%", self.MAX_ROWS + 1),
                )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "Employee not found"

            truncated = len(rows) > self.MAX_ROWS
            results = [dict(row) for row in rows[: self.MAX_ROWS]]

            # Week 6: redact sensitive columns the caller's role cannot view
            # BEFORE the data reaches the LLM. Without a controller/role set we
            # fall back to Week 5 behaviour (return everything).
            if self.access_controller and self.role:
                results = [
                    self.access_controller.redact_employee_fields(self.role, emp)
                    for emp in results
                ]

            payload = {"employees": results}
            if truncated:
                payload["note"] = (
                    f"More than {self.MAX_ROWS} matches; showing the first "
                    f"{self.MAX_ROWS}. Ask the user to narrow the name."
                )
            return json.dumps(payload, default=str)
        except Exception as e:
            logger.error(f"Employee lookup error: {e}")
            return f"Error: {str(e)}"


# TASK 3: PolicySearchTool


class PolicySearchTool(Tool):
    """Search policy documents by keyword."""

    SNIPPET_CHARS = 500

    def __init__(self, documents_path: str = None, access_controller: "AccessController" = None):
        super().__init__("policy_search", "Search policy documents by keyword or topic")
        path = documents_path or str(DATA_DIR / "documents.json")
        try:
            with open(path) as f:
                self.documents = json.load(f)
        except Exception as e:
            # Degrade gracefully: the tool reports the problem instead of
            # crashing the whole agent at startup.
            logger.error(f"Failed to load documents: {e}")
            self.documents = []
        # Week 6: filter documents by the caller's role at retrieval time. The
        # agent sets self.role before each query.
        self.access_controller = access_controller
        self.role = None

    def execute(self, query: str, limit: int = 5) -> str:
        """Search policies by case-insensitive keyword match.

        Returns:
            Formatted string with title + snippet for the top matches
        """
        try:
            if not self.documents:
                return "Error: policy documents are not available"

            # Week 6: only search documents this role is allowed to see, so the
            # agent cannot synthesise an answer from restricted documents.
            documents = self.documents
            if self.access_controller and self.role:
                documents = self.access_controller.filter_documents(
                    self.role, self.documents
                )
                if not documents:
                    return "No policy documents are available for your role."

            # Split the query into words so that a phrase like "travel policy"
            # still matches a document titled "Travel and Expense Policy".
            words = [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]
            if not words:
                words = [query.lower()]

            # Rank by how many distinct query words a document matches, then
            # by occurrence count; a hit in the title weighs more than a hit
            # in the body so the most on-topic document ranks first.
            scored = []
            for doc in documents:
                title = doc.get("title", "")
                content = doc.get("content", "")
                title_l, content_l = title.lower(), content.lower()
                matched_words = 0
                score = 0
                for w in words:
                    hits = content_l.count(w) + 5 * title_l.count(w)
                    if hits:
                        matched_words += 1
                        score += hits
                if matched_words:
                    scored.append((matched_words, score, title, content))
            if not scored:
                return f"No policy documents found for: {query}"

            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            parts = []
            for _, _, title, content in scored[:limit]:
                snippet = content.strip()[: self.SNIPPET_CHARS]
                parts.append(f"## {title}\n{snippet}")
            return "\n\n".join(parts)
        except Exception as e:
            logger.error(f"Policy search error: {e}")
            return f"Error: {str(e)}"


# TASK 4: ExpenseQueryTool


class ExpenseQueryTool(Tool):
    """Query expense policies and approval limits."""

    def __init__(self, policies_path: str = None):
        super().__init__("expense_query", "Query expense approval limits by role")
        path = policies_path or str(DATA_DIR / "policies.json")
        try:
            with open(path) as f:
                self.policies = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            self.policies = {}

    def execute(self, role: str) -> str:
        """Query expense approval limit for a given role.

        Args:
            role: Employee role (ic1_ic2, ic3, manager, director, vp)

        Returns:
            String with approval limit for the given role
        """
        try:
            limits = self.policies.get("expense", {}).get("approval_limits", {})
            if not limits:
                return "Error: expense policies are not available"

            key = role.strip().lower()
            if key not in limits:
                valid = ", ".join(limits.keys())
                return f"Role not found: {role}. Valid roles are: {valid}"
            return f"Approval limit for {key}: ${limits[key]}"
        except Exception as e:
            logger.error(f"Expense query error: {e}")
            return f"Error: {str(e)}"


# TASK 5: The Agent class


class Agent:
    """AI agent that answers questions using Gemini LLM + tools."""

    def __init__(self, db_path: str, api_key: str = None):
        """Initialize the agent.

        Args:
            db_path: Path to SQLite database
            api_key: Google AI API key (or use GOOGLE_API_KEY env var)
        """
        self.db_path = db_path
        self.api_key = api_key or GOOGLE_API_KEY

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Get free key at: "
                "https://aistudio.google.com/app/apikey"
            )

        self.client = genai.Client(api_key=self.api_key)

        # Week 6: guardrails. Access control is enforced inside the data tools
        # (retrieval-time) and again on the final answer (response-time).
        self.access_controller = AccessController(str(DATA_DIR / "access_control.json"))
        self.rate_limiter = RateLimiter(max_queries_per_minute=30)
        self.cost_enforcer = CostEnforcer()

        self.tools = {
            "employee_lookup": EmployeeLookupTool(db_path, self.access_controller),
            "policy_search": PolicySearchTool(access_controller=self.access_controller),
            "expense_query": ExpenseQueryTool(),
        }

        self.token_count = 0
        self.total_cost = 0.0
        self.queries_run = 0

    def _build_tool_declarations(self) -> types.Tool:
        """Declare the tools in the Gemini function-calling schema.

        The model can only call functions declared here, which prevents it
        from hallucinating tool names.
        """
        declarations = [
            types.FunctionDeclaration(
                name="employee_lookup",
                description=(
                    "Find employee information (title, department, contact, "
                    "hire date, etc.) by name or by numeric employee ID."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "employee_name": types.Schema(
                            type=types.Type.STRING,
                            description="Full or partial employee name",
                        ),
                        "employee_id": types.Schema(
                            type=types.Type.STRING,
                            description="Exact numeric employee ID",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="policy_search",
                description=(
                    "Search TechCorp policy documents (HR, travel, security, "
                    "benefits, etc.) by a short keyword such as 'travel' or "
                    "'parental leave'. Returns the most relevant documents."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Short keyword or topic to search for",
                        ),
                        "limit": types.Schema(
                            type=types.Type.INTEGER,
                            description="Max number of documents to return",
                        ),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="expense_query",
                description=(
                    "Get the expense approval limit in dollars for a role. "
                    "Valid roles: ic1_ic2, ic3, manager, director, vp. Map "
                    "user wording to these roles (e.g. 'junior engineer' -> "
                    "ic1_ic2, 'senior engineer' -> ic3)."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "role": types.Schema(
                            type=types.Type.STRING,
                            description="One of: ic1_ic2, ic3, manager, director, vp",
                        ),
                    },
                    required=["role"],
                ),
            ),
        ]
        return types.Tool(function_declarations=declarations)

    def _build_system_prompt(self, user_role: str) -> str:
        """Build system prompt describing the agent and its tools."""
        tool_lines = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools.values()
        )
        return f"""You are a TechCorp internal assistant. You answer employee questions
about company data and policies.

User role: {user_role}

Available tools:
{tool_lines}

Rules:
1. Always answer from tool results. Do NOT answer company-specific questions
   from your own knowledge, and do NOT make up employees, numbers, or policy
   text. If the question is about TechCorp data or policy, call a tool first.
2. If a tool returns an error or no results, say so honestly and suggest what
   the user could try instead. Never invent a substitute answer.
3. If the question is not about TechCorp at all, politely say it is out of
   scope for this assistant.
4. Keep the final answer short and factual, and base every claim on the tool
   output you received."""

    def query(
        self, user_query: str, user_id: str, user_role: str = "engineer"
    ) -> Dict[str, Any]:
        """Answer a question using LLM + tools, behind Week 6 guardrails.

        Guardrails (checked before any LLM call):
        1. Rate limit per user_id.
        2. Budget per user_id / role.

        Then the reasoning loop:
        3. Send the question to Gemini with the system prompt and the
           declared tools (the data tools now redact/filter by role).
        4. Execute any function calls and feed results back, up to
           MAX_TOOL_STEPS, tracking tokens and cost.
        5. Redact any sensitive value left in the final answer, record the
           real cost against the user, and audit the query.

        Returns:
            Dict with keys: answer, tokens_used, cost, role (or error)
        """
        logger.info(f"Processing query from {user_id} ({user_role}): {user_query}")

        # --- Guardrail 1: rate limit (before spending any tokens) ---
        if not self.rate_limiter.is_allowed(user_id):
            self.access_controller.log_access(user_role, "rate_limit", allowed=False)
            return {
                "error": "Rate limit exceeded",
                "remaining_queries": self.rate_limiter.get_remaining_queries(user_id),
            }

        # --- Guardrail 2: budget (before spending any tokens) ---
        if not self.cost_enforcer.can_afford_query(
            user_id, estimated_cost=0.01, role=user_role
        ):
            self.access_controller.log_access(user_role, "budget", allowed=False)
            return {
                "error": "Budget exceeded",
                "budget_remaining": self.cost_enforcer.get_budget_remaining(
                    user_id, role=user_role
                ),
            }

        # Tell the access-aware tools whose permissions apply this query.
        for tool in self.tools.values():
            if hasattr(tool, "role"):
                tool.role = user_role

        config = types.GenerateContentConfig(
            system_instruction=self._build_system_prompt(user_role),
            tools=[self._build_tool_declarations()],
        )
        contents = [
            types.Content(
                role="user", parts=[types.Part.from_text(text=user_query)]
            )
        ]

        query_input_tokens = 0
        query_output_tokens = 0
        answer = None

        try:
            for step in range(MAX_TOOL_STEPS):
                response = self._generate_with_retry(contents, config)

                usage = response.usage_metadata
                query_input_tokens += usage.prompt_token_count or 0
                # Thinking tokens are billed as output, so count them too.
                query_output_tokens += (usage.candidates_token_count or 0) + (
                    usage.thoughts_token_count or 0
                )

                if response.function_calls:
                    # Keep the model's turn in the history, then execute every
                    # requested tool and feed the results back.
                    contents.append(response.candidates[0].content)
                    for fc in response.function_calls:
                        result = self._execute_tool(fc.name, dict(fc.args or {}))
                        logger.info(f"Tool {fc.name}({dict(fc.args or {})}) -> {result[:200]}")
                        contents.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_function_response(
                                        name=fc.name,
                                        response={"result": result},
                                    )
                                ],
                            )
                        )
                    continue

                answer = response.text or "(empty response from model)"
                break

            if answer is None:
                answer = (
                    "I could not finish answering within the tool-call limit. "
                    "Please try a more specific question."
                )
        except Exception as e:
            # The agent never crashes on an API failure; it reports it.
            logger.error(f"LLM call failed: {e}")
            answer = f"Sorry, I hit an internal error and cannot answer right now: {e}"

        cost = self._estimate_query_cost(query_input_tokens, query_output_tokens)
        tokens_used = query_input_tokens + query_output_tokens

        # Defense in depth: scrub any sensitive value the LLM may have echoed.
        answer = self.access_controller.redact_response(user_role, answer)

        # Charge the real cost to the user and audit the query.
        self.cost_enforcer.add_cost(user_id, user_role, cost)
        self.access_controller.log_access(user_role, "query", allowed=True)

        self.queries_run += 1
        self.token_count += tokens_used
        self.total_cost += cost

        return {
            "answer": answer,
            "tokens_used": tokens_used,
            "cost": cost,
            "role": user_role,
            "budget_remaining": self.cost_enforcer.get_budget_remaining(
                user_id, role=user_role
            ),
        }

    def _generate_with_retry(self, contents, config):
        """Call Gemini, retrying with exponential backoff on rate limits.

        The free tier allows 5 requests/minute, and one agent query needs at
        least two requests, so back-to-back queries can hit 429 errors.
        """
        delay = INITIAL_BACKOFF_SECONDS
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.models.generate_content(
                    model=MODEL_ID, contents=contents, config=config
                )
            except genai_errors.APIError as e:
                # 429 = free-tier rate limit, 503 = model overloaded.
                # Both are transient, so back off and retry.
                code = getattr(e, "code", None)
                if code in (429, 503) and attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Transient API error ({code}); retrying in {delay}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute one tool call requested by the LLM, never raising."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"Error: unknown tool '{tool_name}'"
        try:
            return tool.execute(**args)
        except TypeError as e:
            # Bad/missing arguments from the model: report instead of crashing.
            return f"Error: invalid arguments for {tool_name}: {e}"
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return f"Error: {tool_name} failed: {e}"

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on tokens.

        Rates from the assignment README:
        - Input: $0.075 per 1M tokens
        - Output: $0.3 per 1M tokens
        """
        input_cost = (input_tokens / 1_000_000) * 0.075
        output_cost = (output_tokens / 1_000_000) * 0.3
        return input_cost + output_cost

    def get_metrics(self) -> Dict[str, Any]:
        """Return running performance metrics."""
        avg_cost = self.total_cost / self.queries_run if self.queries_run else 0.0
        return {
            "total_queries": self.queries_run,
            "total_tokens": self.token_count,
            "total_cost": self.total_cost,
            "avg_cost_per_query": avg_cost,
        }


# TASK 6: Test your implementation

if __name__ == "__main__":
    """Quick test of agent functionality."""
    import sys

    try:
        # Initialize agent
        agent = Agent(str(DATA_DIR / "techcorp.db"))
        print("Agent initialized successfully")

        # Test a query (Week 6: query() now needs user_id and user_role)
        print("\nTesting query: 'What is the travel policy?'")
        result = agent.query(
            "What is the travel policy?", user_id="user1", user_role="engineer"
        )
        print(f"Answer: {result.get('answer') or result}")
        if "tokens_used" in result:
            print(f"Tokens: {result['tokens_used']}")
            print(f"Cost: ${result['cost']:.6f}")

        # Check metrics
        metrics = agent.get_metrics()
        print(f"\nMetrics: {metrics}")
        print(f"Audit log entries: {len(agent.access_controller.get_audit_log())}")

    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Error during test")
        sys.exit(1)
