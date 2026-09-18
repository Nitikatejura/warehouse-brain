"""
Blocks World Domain for Module A Planning
Defines predicates, primitive actions (STRIPS style), and WorldState.
"""

from typing import Set, Tuple, List, Optional, Dict, Any
import copy


class Action:
    def __init__(self, name: str, pre: Set[Tuple], add: Set[Tuple], delete: Set[Tuple]):
        self.name = name
        self.pre = set(pre)
        self.add = set(add)
        self.delete = set(delete)

    def is_applicable(self, state: 'WorldState') -> bool:
        return self.pre.issubset(state.predicates)

    def apply(self, state: 'WorldState') -> 'WorldState':
        if not self.is_applicable(state):
            raise ValueError(f"Action {self.name} cannot be applied to state: missing {self.pre - state.predicates}")
        new_state = copy.deepcopy(state)
        new_state.predicates.difference_update(self.delete)
        new_state.predicates.update(self.add)
        return new_state

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Action) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class WorldState:
    def __init__(self, predicates: Optional[Set[Tuple]] = None):
        self.predicates: Set[Tuple] = set(predicates) if predicates else set()

    def add(self, pred: Tuple):
        self.predicates.add(pred)

    def remove(self, pred: Tuple):
        self.predicates.discard(pred)

    def satisfies(self, goal: Set[Tuple]) -> bool:
        return goal.issubset(self.predicates)

    def to_dict(self) -> Dict[str, Any]:
        return {"predicates": [list(p) for p in sorted(list(self.predicates))]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldState':
        preds = {tuple(p) for p in data.get("predicates", [])}
        return cls(preds)

    def __repr__(self):
        return f"WorldState({sorted(list(self.predicates))})"

    def __eq__(self, other):
        return isinstance(other, WorldState) and self.predicates == other.predicates


def get_all_actions(blocks: List[str]) -> List[Action]:
    """Generates all instantiated STRIPS actions for the given blocks."""
    actions = []
    # Pickup(x)
    for x in blocks:
        actions.append(Action(
            name=f"Pickup({x})",
            pre={("Clear", x), ("OnTable", x), ("ArmEmpty",)},
            add={("Holding", x)},
            delete={("Clear", x), ("OnTable", x), ("ArmEmpty",)}
        ))

    # Putdown(x)
    for x in blocks:
        actions.append(Action(
            name=f"Putdown({x})",
            pre={("Holding", x)},
            add={("Clear", x), ("OnTable", x), ("ArmEmpty",)},
            delete={("Holding", x)}
        ))

    # Stack(x, y)
    for x in blocks:
        for y in blocks:
            if x != y:
                actions.append(Action(
                    name=f"Stack({x}, {y})",
                    pre={("Holding", x), ("Clear", y)},
                    add={("On", x, y), ("Clear", x), ("ArmEmpty",)},
                    delete={("Holding", x), ("Clear", y)}
                ))

    # Unstack(x, y)
    for x in blocks:
        for y in blocks:
            if x != y:
                actions.append(Action(
                    name=f"Unstack({x}, {y})",
                    pre={("On", x, y), ("Clear", x), ("ArmEmpty",)},
                    add={("Holding", x), ("Clear", y)},
                    delete={("On", x, y), ("Clear", x), ("ArmEmpty",)}
                ))

    return actions
