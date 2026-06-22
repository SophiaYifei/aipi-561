"""
Week 7: Cost Optimization & Feedback Loop

Implements three systems on top of the Week 5/6 agent:
1. CostAnalyzer        - track per-query cost, break it down by component,
                         and flag statistically expensive queries.
2. OptimizationStrategy - cut cost via caching, retrieval top-k, complexity-aware
                          model selection, and response compression.
3. FeedbackLoop        - collect user corrections, validate them by role
                         authority, and measure their quality over time.

The three classes are self-contained: they operate on cost dicts and plain
strings, so the test block below runs fully offline with synthetic data and
spends no LLM quota. The agent in app_starter.py produces exactly these cost
dicts at runtime, so CostAnalyzer can be fed real queries unchanged.
"""

import json
import logging
import statistics
from typing import Dict, List, Any
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- tunable constants -------------------------------------------------------
COST_COMPONENTS = ["retrieval_cost", "llm_cost", "tool_cost", "error_cost"]
SPIKE_STDEV_MULTIPLIER = 2.0          # outlier threshold: mean + 2*stdev
RETRIEVAL_TOP_K = 3                   # documents kept after retrieval pruning
COMPRESSION_MAX_SENTENCES = 3         # sentences kept when compressing answers
# Models used by our agent (Week 5 settled on gemini-3.5-flash as the cheap tier).
CHEAP_MODEL = "gemini-3.5-flash"
CAPABLE_MODEL = "gemini-2.5-pro"
COMPLEX_KEYWORDS = ["analyze", "compare", "design", "evaluate", "explain why",
                    "trade-off", "tradeoff", "summarize across", "recommend"]
# Estimated savings per strategy (fraction of the cost the strategy touches).
# Caching is intentionally excluded here: its real saving is the observed cache
# hit rate, computed at runtime, not a fixed guess.
STRATEGY_SAVINGS = {
    "retrieval_pruning": 0.40,  # fewer retrieved docs -> fewer input tokens
    "model_selection": 0.50,    # cheap model on a simple query
    "response_compression": 0.15,
}


# ============================================================================
# TASK 1: CostAnalyzer
# ============================================================================


class CostAnalyzer:
    """Analyze and track query costs by component."""

    def __init__(self):
        # Each entry is a normalized cost record (see record_query).
        self.query_history: List[Dict[str, Any]] = []

    def record_query(self, query: Dict[str, Any]):
        """Record a query and its cost breakdown.

        Accepts a dict with any subset of the cost components; missing
        components default to 0.0. total_cost is recomputed from the
        components so it always stays consistent, and a timestamp is stamped.
        """
        record = {
            "query_text": query.get("query_text", ""),
            "retrieval_cost": float(query.get("retrieval_cost", 0.0)),
            "llm_cost": float(query.get("llm_cost", 0.0)),
            "tool_cost": float(query.get("tool_cost", 0.0)),
            "error_cost": float(query.get("error_cost", 0.0)),
        }
        record["total_cost"] = round(sum(record[c] for c in COST_COMPONENTS), 8)
        record["timestamp"] = query.get("timestamp") or _utc_now()
        self.query_history.append(record)
        logger.info(
            "Recorded query (total=$%.6f): %s",
            record["total_cost"], record["query_text"][:60],
        )
        return record

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get total cost broken down by component across all recorded queries."""
        breakdown = {
            "retrieval_total": round(sum(q["retrieval_cost"] for q in self.query_history), 8),
            "llm_total": round(sum(q["llm_cost"] for q in self.query_history), 8),
            "tool_total": round(sum(q["tool_cost"] for q in self.query_history), 8),
            "error_total": round(sum(q["error_cost"] for q in self.query_history), 8),
            "query_count": len(self.query_history),
        }
        breakdown["total_daily"] = round(
            breakdown["retrieval_total"] + breakdown["llm_total"]
            + breakdown["tool_total"] + breakdown["error_total"], 8,
        )
        return breakdown

    def identify_cost_spikes(self) -> List[Dict]:
        """Flag queries whose total cost is a statistical outlier.

        A query is a spike when its total_cost exceeds mean + 2*stdev of all
        recorded query costs. Needs at least two queries with non-zero spread;
        otherwise there is no meaningful distribution and we return [].
        """
        if len(self.query_history) < 2:
            return []

        costs = [q["total_cost"] for q in self.query_history]
        mean = statistics.mean(costs)
        stdev = statistics.stdev(costs)  # sample stdev
        if stdev == 0:
            return []

        threshold = mean + SPIKE_STDEV_MULTIPLIER * stdev
        spikes = []
        for q in self.query_history:
            if q["total_cost"] > threshold:
                spikes.append({
                    "query_text": q["query_text"],
                    "total_cost": q["total_cost"],
                    "threshold": round(threshold, 8),
                    "mean": round(mean, 8),
                    "stdev": round(stdev, 8),
                    "times_over_mean": round(q["total_cost"] / mean, 2) if mean else None,
                })
        if spikes:
            logger.warning("Detected %d cost spike(s) above $%.6f", len(spikes), threshold)
        return spikes


# ============================================================================
# TASK 2: OptimizationStrategy
# ============================================================================


class OptimizationStrategy:
    """Optimize agent costs through multiple strategies."""

    def __init__(self):
        self.cache: Dict[str, str] = {}        # {query: response}
        self.strategies_applied: List[str] = []  # names of strategies actually used
        self.cache_hits = 0
        self.cache_misses = 0

    def _mark(self, strategy: str):
        if strategy not in self.strategies_applied:
            self.strategies_applied.append(strategy)

    def apply_caching(self, query: str, response: str) -> tuple:
        """Return a cached answer when the exact query was seen before.

        Returns (is_cached_hit, response). On a hit the cost of recomputing
        the answer is avoided entirely; on a miss the response is stored.
        """
        if query in self.cache:
            self.cache_hits += 1
            self._mark("caching")
            logger.info("Cache HIT for query: %s", query[:60])
            return (True, self.cache[query])

        self.cache_misses += 1
        self.cache[query] = response
        logger.info("Cache MISS, stored query: %s", query[:60])
        return (False, response)

    def optimize_retrieval_count(self, num_docs: int) -> int:
        """Keep only the top-k documents instead of everything retrieved.

        Caps the count at RETRIEVAL_TOP_K (e.g. 15 -> 3) while never returning
        less than 1 when at least one document exists.
        """
        optimized = min(num_docs, RETRIEVAL_TOP_K) if num_docs > 0 else 0
        optimized = max(optimized, 1) if num_docs > 0 else 0
        if optimized < num_docs:
            self._mark("retrieval_pruning")
            logger.info("Retrieval pruned: %d -> %d docs", num_docs, optimized)
        return optimized

    def select_model_by_complexity(self, query: str) -> str:
        """Route simple queries to the cheap model, complex ones to the capable model."""
        q = query.lower()
        is_complex = any(kw in q for kw in COMPLEX_KEYWORDS) or len(query.split()) > 25
        model = CAPABLE_MODEL if is_complex else CHEAP_MODEL
        if model == CHEAP_MODEL:
            self._mark("model_selection")
        logger.info("Selected model %s for query: %s", model, query[:60])
        return model

    def enable_response_compression(self, response: str) -> str:
        """Trim a long answer down to its first few sentences.

        Splits on sentence terminators and keeps the leading
        COMPRESSION_MAX_SENTENCES sentences. Short answers pass through.
        """
        # Split on '.', '!', '?' while keeping reasonable sentence chunks.
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s.strip()]
        if len(sentences) <= COMPRESSION_MAX_SENTENCES:
            return response
        compressed = " ".join(sentences[:COMPRESSION_MAX_SENTENCES])
        self._mark("response_compression")
        logger.info("Compressed response: %d -> %d sentences", len(sentences), COMPRESSION_MAX_SENTENCES)
        return compressed

    def get_optimization_impact(self) -> Dict[str, Any]:
        """Estimate combined cost savings from the strategies actually applied.

        Per-strategy estimates are combined multiplicatively (savings stack on
        the cost that survives the previous strategy), which avoids the
        unrealistic >100% you would get by naively adding them.
        """
        breakdown = {s: STRATEGY_SAVINGS[s] for s in self.strategies_applied
                     if s in STRATEGY_SAVINGS}

        # Caching's saving is the measured hit rate, not a fixed estimate.
        total_lookups = self.cache_hits + self.cache_misses
        cache_hit_rate = round(self.cache_hits / total_lookups, 2) if total_lookups else 0.0
        if "caching" in self.strategies_applied and cache_hit_rate > 0:
            breakdown["caching"] = cache_hit_rate

        remaining = 1.0
        for saving in breakdown.values():
            remaining *= (1.0 - saving)
        total_savings_pct = round((1.0 - remaining) * 100, 1)

        impact = {
            "total_savings_pct": total_savings_pct,
            "strategies_applied": self.strategies_applied,
            "breakdown": {s: f"{v * 100:.0f}%" for s, v in breakdown.items()},
        }
        if "caching" in self.strategies_applied:
            impact["cache_hit_rate"] = cache_hit_rate
        return impact


# ============================================================================
# TASK 3: FeedbackLoop
# ============================================================================


class FeedbackLoop:
    """Collect and validate user corrections for continuous improvement."""

    def __init__(self):
        self.corrections: List[Dict[str, Any]] = []
        # Authority hierarchy for role-based validation.
        self.authority = {
            "engineer": 1,
            "hr": 2,
            "finance": 2,
            "manager": 3,
            "executive": 4,
        }
        self.rejected_count = 0  # submissions that failed the acceptance gate

    # Corrections from this level or above are trusted enough to auto-apply.
    TRUSTED_AUTHORITY_LEVEL = 3  # manager and above

    def submit_correction(
        self,
        original_query: str,
        original_answer: str,
        corrected_answer: str,
        user_role: str,
    ) -> Dict[str, Any]:
        """Accept a correction if the role is recognized and it adds detail.

        Acceptance gate (everyone recognized may contribute):
          1. user_role is a known role.
          2. corrected_answer is more detailed (longer) than the original.
        Accepted corrections are stored; rejected ones are not.
        """
        role = (user_role or "").lower()
        if role not in self.authority:
            self.rejected_count += 1
            return {"accepted": False, "reason": f"Unknown role '{user_role}'"}

        if len(corrected_answer.strip()) <= len(original_answer.strip()):
            self.rejected_count += 1
            return {
                "accepted": False,
                "reason": "Correction is not more detailed than the original answer",
            }

        entry = {
            "original_query": original_query,
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "user_role": role,
            "authority_level": self.authority[role],
            "timestamp": _utc_now(),
        }
        self.corrections.append(entry)
        logger.info("Accepted correction from %s for query: %s", role, original_query[:60])
        return {"accepted": True, "reason": "Correction accepted", "index": len(self.corrections) - 1}

    def validate_correction(self, index: int) -> bool:
        """Decide whether a stored correction is trustworthy enough to apply.

        Stricter than submission: requires manager+ authority, that the
        correction is genuinely more detailed, and that it actually differs
        from the original answer.
        """
        if index < 0 or index >= len(self.corrections):
            return False
        c = self.corrections[index]

        has_authority = c["authority_level"] >= self.TRUSTED_AUTHORITY_LEVEL
        more_detailed = len(c["corrected_answer"].strip()) > len(c["original_answer"].strip())
        makes_sense = (
            c["corrected_answer"].strip().lower() != c["original_answer"].strip().lower()
            and len(c["corrected_answer"].split()) >= 3
        )
        return bool(has_authority and more_detailed and makes_sense)

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Summarize the quality of collected feedback."""
        total = len(self.corrections)
        if total == 0:
            return {
                "total_corrections": 0,
                "validation_rate": 0.0,
                "avg_correction_length": 0.0,
                "top_error_patterns": [],
                "rejected_submissions": self.rejected_count,
            }

        valid = sum(1 for i in range(total) if self.validate_correction(i))
        avg_len = sum(len(c["corrected_answer"]) for c in self.corrections) / total

        return {
            "total_corrections": total,
            "validation_rate": round(valid / total, 2),
            "avg_correction_length": round(avg_len, 1),
            "top_error_patterns": self._top_error_patterns(),
            "rejected_submissions": self.rejected_count,
        }

    def _top_error_patterns(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Group corrected queries into coarse topics to surface recurring gaps."""
        topics = {
            "travel": ["travel", "flight", "trip", "hotel", "business class"],
            "compensation": ["salary", "bonus", "stock", "compensation", "pay"],
            "leave": ["leave", "pto", "vacation", "sick", "parental"],
            "expense": ["expense", "reimburse", "budget", "spend"],
            "security": ["security", "access", "password", "vpn", "data"],
        }
        counts: Dict[str, int] = {}
        for c in self.corrections:
            q = c["original_query"].lower()
            label = next((t for t, kws in topics.items() if any(k in q for k in kws)), "other")
            counts[label] = counts.get(label, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [{"pattern": p, "count": n} for p, n in ranked[:top_n]]


# ============================================================================
# Test block - runs offline with synthetic data (no LLM quota used).
# Run with: python3 cost_optimization_starter.py
# ============================================================================

if __name__ == "__main__":
    passed = 0
    failed = 0

    def check(name: str, condition: bool):
        global passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}")

    # ------------------------------------------------------------------
    # TEST 1: CostAnalyzer - breakdown by component + spike detection
    # ------------------------------------------------------------------
    print("\n=== Testing CostAnalyzer ===")
    analyzer = CostAnalyzer()

    # Empty state is well-defined.
    empty = analyzer.get_cost_breakdown()
    check("empty breakdown totals are zero", empty["total_daily"] == 0.0 and empty["query_count"] == 0)
    check("no spikes with no data", analyzer.identify_cost_spikes() == [])

    # A day of cheap baseline queries plus one obvious spike (a retry storm).
    # mean + 2*stdev needs enough baseline points so one outlier doesn't
    # inflate the standard deviation past its own value.
    cheap_queries = [
        {"query_text": "What is the PTO policy?", "retrieval_cost": 0.0002, "llm_cost": 0.0010, "tool_cost": 0.0001, "error_cost": 0.0},
        {"query_text": "Where is the office?", "retrieval_cost": 0.0001, "llm_cost": 0.0008, "tool_cost": 0.0001, "error_cost": 0.0},
        {"query_text": "What is the dress code?", "retrieval_cost": 0.0002, "llm_cost": 0.0009, "tool_cost": 0.0001, "error_cost": 0.0},
        {"query_text": "How many vacation days?", "retrieval_cost": 0.0001, "llm_cost": 0.0011, "tool_cost": 0.0002, "error_cost": 0.0},
        {"query_text": "What time does the office open?", "retrieval_cost": 0.0001, "llm_cost": 0.0009, "tool_cost": 0.0001, "error_cost": 0.0},
        {"query_text": "Is there a gym?", "retrieval_cost": 0.0001, "llm_cost": 0.0008, "tool_cost": 0.0001, "error_cost": 0.0},
        {"query_text": "What is the parental leave policy?", "retrieval_cost": 0.0002, "llm_cost": 0.0012, "tool_cost": 0.0001, "error_cost": 0.0},
        {"query_text": "How do I reset my password?", "retrieval_cost": 0.0001, "llm_cost": 0.0007, "tool_cost": 0.0001, "error_cost": 0.0},
        {"query_text": "Where do I submit expenses?", "retrieval_cost": 0.0002, "llm_cost": 0.0010, "tool_cost": 0.0002, "error_cost": 0.0},
        {"query_text": "What is the remote work policy?", "retrieval_cost": 0.0002, "llm_cost": 0.0011, "tool_cost": 0.0001, "error_cost": 0.0},
    ]
    for q in cheap_queries:
        analyzer.record_query(q)
    # An expensive query: many retries (error_cost) + large context.
    analyzer.record_query({
        "query_text": "Compare every benefit across all 7 departments and summarize",
        "retrieval_cost": 0.0050, "llm_cost": 0.0300, "tool_cost": 0.0020, "error_cost": 0.0150,
    })

    breakdown = analyzer.get_cost_breakdown()
    print("  breakdown:", json.dumps(breakdown, indent=2))
    check("query_count is 11", breakdown["query_count"] == 11)
    expected_total = sum(
        q.get("retrieval_cost", 0) + q.get("llm_cost", 0) + q.get("tool_cost", 0) + q.get("error_cost", 0)
        for q in cheap_queries
    ) + (0.0050 + 0.0300 + 0.0020 + 0.0150)
    check("total_daily matches manual sum", abs(breakdown["total_daily"] - expected_total) < 1e-9)
    check("components sum to total",
          abs((breakdown["retrieval_total"] + breakdown["llm_total"]
               + breakdown["tool_total"] + breakdown["error_total"]) - breakdown["total_daily"]) < 1e-9)

    spikes = analyzer.identify_cost_spikes()
    print("  spikes:", json.dumps(spikes, indent=2))
    check("exactly one spike detected", len(spikes) == 1)
    check("spike is the expensive comparison query",
          spikes and spikes[0]["query_text"].startswith("Compare every benefit"))

    # ------------------------------------------------------------------
    # TEST 2: OptimizationStrategy - caching, retrieval, model, compression
    # ------------------------------------------------------------------
    print("\n=== Testing OptimizationStrategy ===")
    optimizer = OptimizationStrategy()

    hit1, resp1 = optimizer.apply_caching("What is the PTO policy?", "PTO is 20 days.")
    hit2, resp2 = optimizer.apply_caching("What is the PTO policy?", "PTO is 20 days.")
    check("first lookup is a miss", hit1 is False)
    check("second lookup is a hit", hit2 is True and resp2 == "PTO is 20 days.")

    check("15 docs pruned to 3", optimizer.optimize_retrieval_count(15) == 3)
    check("2 docs left as 2", optimizer.optimize_retrieval_count(2) == 2)
    check("0 docs stays 0", optimizer.optimize_retrieval_count(0) == 0)

    check("simple query -> cheap model",
          optimizer.select_model_by_complexity("What is the office address?") == CHEAP_MODEL)
    check("complex query -> capable model",
          optimizer.select_model_by_complexity("Analyze and compare the travel policy across departments") == CAPABLE_MODEL)

    long_answer = ("Employees may book business class on long flights. "
                   "Approval is required from a manager. "
                   "Receipts must be submitted within 30 days. "
                   "Reimbursement takes two weeks. "
                   "Contact finance for exceptions.")
    compressed = optimizer.enable_response_compression(long_answer)
    check("compression keeps 3 sentences", compressed.count(".") == 3)
    check("short answer passes through unchanged",
          optimizer.enable_response_compression("PTO is 20 days.") == "PTO is 20 days.")

    impact = optimizer.get_optimization_impact()
    print("  impact:", json.dumps(impact, indent=2))
    check("impact lists applied strategies", len(impact["strategies_applied"]) >= 3)
    check("total savings between 0 and 100", 0 < impact["total_savings_pct"] <= 100)
    check("cache hit rate reported", impact.get("cache_hit_rate") == 0.5)

    # ------------------------------------------------------------------
    # TEST 3: FeedbackLoop - corrections, role authority, metrics
    # ------------------------------------------------------------------
    print("\n=== Testing FeedbackLoop ===")
    feedback = FeedbackLoop()

    # Engineer correction: accepted (recognized + more detailed) but NOT trusted.
    r1 = feedback.submit_correction(
        original_query="What is the travel policy for flights over 8 hours?",
        original_answer="There is no specific policy for 8+ hour flights.",
        corrected_answer="Employees can book business class for flights over 8 hours with manager approval.",
        user_role="engineer",
    )
    check("engineer correction accepted", r1["accepted"] is True)
    check("engineer correction not validated (below manager)", feedback.validate_correction(r1["index"]) is False)

    # Manager correction: accepted and trusted.
    r2 = feedback.submit_correction(
        original_query="How many vacation days do new hires get?",
        original_answer="Unsure.",
        corrected_answer="New hires accrue 15 vacation days in their first year, rising to 20 after three years.",
        user_role="manager",
    )
    check("manager correction accepted", r2["accepted"] is True)
    check("manager correction validated", feedback.validate_correction(r2["index"]) is True)

    # Unknown role: rejected.
    r3 = feedback.submit_correction("Q", "A", "A much longer corrected answer here", "intern")
    check("unknown role rejected", r3["accepted"] is False)

    # Not more detailed: rejected.
    r4 = feedback.submit_correction("Q", "This is the original answer.", "Too short.", "executive")
    check("non-detailed correction rejected", r4["accepted"] is False)

    metrics = feedback.get_feedback_metrics()
    print("  metrics:", json.dumps(metrics, indent=2))
    check("two corrections stored", metrics["total_corrections"] == 2)
    check("validation rate is 0.5", metrics["validation_rate"] == 0.5)
    check("two submissions rejected", metrics["rejected_submissions"] == 2)
    check("top error patterns present", len(metrics["top_error_patterns"]) >= 1)

    # ------------------------------------------------------------------
    print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
    if failed:
        raise SystemExit(1)
    print("All tests passed.")
