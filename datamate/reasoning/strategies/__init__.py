"""Import để đăng ký các strategy mặc định vào registry."""
from datamate.reasoning.strategies import guard, llm_code, llm_plan, rule_plan  # noqa: F401
from datamate.agent import agent  # noqa: F401,E402  (v2: strategy 'agent')
