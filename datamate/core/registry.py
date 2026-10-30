"""Registry cho strategy: thêm chiến lược mới = viết 1 class + @register_strategy('ten'), không sửa orchestrator."""
from __future__ import annotations

from typing import Callable, Dict, Type

_STRATEGIES: Dict[str, type] = {}


def register_strategy(name: str) -> Callable[[Type], Type]:
    def deco(cls: Type) -> Type:
        cls.name = name
        _STRATEGIES[name] = cls
        return cls
    return deco


def get_strategy(name: str) -> type:
    if name not in _STRATEGIES:
        # import lazy để đăng ký các strategy mặc định
        import datamate.reasoning.strategies  # noqa: F401
    return _STRATEGIES[name]


def list_strategies() -> list[str]:
    import datamate.reasoning.strategies  # noqa: F401
    return sorted(_STRATEGIES)
