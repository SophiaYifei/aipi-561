"""Offline test of the three tools (no LLM API calls needed).

Usage:
    cd week5
    source .venv/bin/activate
    python test_tools.py
"""

from app_starter import (
    DATA_DIR,
    EmployeeLookupTool,
    ExpenseQueryTool,
    PolicySearchTool,
)


def main():
    print("Tool tests (no LLM involved)\n")

    emp = EmployeeLookupTool(str(DATA_DIR / "techcorp.db"))
    print("1. employee_lookup by id=1:")
    print(emp.execute(employee_id="1")[:250], "...\n")
    print("2. employee_lookup by name='Brian Yang':")
    print(emp.execute(employee_name="Brian Yang")[:250], "...\n")
    print("3. employee_lookup with a name that does not exist:")
    print(emp.execute(employee_name="Zebulon Quixote"), "\n")
    print("4. employee_lookup with no arguments:")
    print(emp.execute(), "\n")

    pol = PolicySearchTool()
    print("5. policy_search for 'travel policy' (top 1):")
    print(pol.execute("travel policy", limit=1)[:400], "...\n")
    print("6. policy_search for a word with no matches:")
    print(pol.execute("xyzzy"), "\n")

    exp = ExpenseQueryTool()
    print("7. expense_query for 'manager':")
    print(exp.execute("manager"), "\n")
    print("8. expense_query for 'VP' (case-insensitive):")
    print(exp.execute("VP"), "\n")
    print("9. expense_query for an unknown role:")
    print(exp.execute("intern"), "\n")

    print("All tool tests done.")


if __name__ == "__main__":
    main()
