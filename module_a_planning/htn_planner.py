"""
Hierarchical Task Network (HTN) Planner for Warehouse Pallet Stacking.
Decomposes high-level operators like LoadPallet(pallet_id, stacks)
into intermediate subtasks (e.g., ClearBlock, MoveBlock) and primitive actions.
"""

from typing import List, Tuple, Dict, Any, Optional
import copy
from .blocks_world import Action, WorldState, get_all_actions


class HTNPlanner:
    def __init__(self, blocks: List[str]):
        self.blocks = blocks
        self.all_actions = {a.name: a for a in get_all_actions(blocks)}

    def plan_load_pallet(self, initial_state: WorldState, target_stacks: List[List[str]]) -> List[Action]:
        """
        High-level operator: LoadPallet(P)
        Target stacks: e.g. [['C', 'B', 'A'], ['E', 'D']]
        where target_stacks[i] has bottom at index 0 and top at index -1.
        Expands recursively into primitive stack/unstack/pickup/putdown actions.
        """
        current_state = copy.deepcopy(initial_state)
        primitive_plan: List[Action] = []

        # High-level decomposition:
        # Step 1: Clear all blocks to table if they are currently stacked
        for stack in target_stacks:
            for block in stack:
                self._decompose_clear(block, current_state, primitive_plan)

        # Step 2: Ensure all base blocks are on table
        for stack in target_stacks:
            bottom = stack[0]
            if not self._is_on_table(current_state, bottom):
                self._decompose_clear(bottom, current_state, primitive_plan)
                self._decompose_move_to_table(bottom, current_state, primitive_plan)

        # Step 3: For each stack, build from bottom to top
        for stack in target_stacks:
            for i in range(len(stack) - 1):
                base = stack[i]
                top = stack[i + 1]
                if ("On", top, base) in current_state.predicates:
                    continue
                # Ensure top and base are clear
                self._decompose_clear(top, current_state, primitive_plan)
                self._decompose_clear(base, current_state, primitive_plan)
                # Stack top onto base
                self._decompose_stack(top, base, current_state, primitive_plan)

        return primitive_plan

    def _is_clear(self, state: WorldState, x: str) -> bool:
        return ("Clear", x) in state.predicates

    def _is_on_table(self, state: WorldState, x: str) -> bool:
        return ("OnTable", x) in state.predicates

    def _get_block_on(self, state: WorldState, x: str) -> Optional[str]:
        for pred in state.predicates:
            if pred[0] == "On" and pred[2] == x:
                return pred[1]
        return None

    def _get_support_of(self, state: WorldState, x: str) -> Optional[str]:
        for pred in state.predicates:
            if pred[0] == "On" and pred[1] == x:
                return pred[2]
        return None

    def _apply_action(self, action: Action, state: WorldState, plan: List[Action]):
        new_state = action.apply(state)
        state.predicates = new_state.predicates
        plan.append(action)

    def _decompose_clear(self, x: str, state: WorldState, plan: List[Action]):
        """Recursively clears any block sitting on top of x."""
        while not self._is_clear(state, x):
            above = self._get_block_on(state, x)
            if above:
                self._decompose_clear(above, state, plan)
                self._decompose_move_to_table(above, state, plan)
            else:
                break

    def _decompose_move_to_table(self, x: str, state: WorldState, plan: List[Action]):
        """Primitive decomposition: moves block x to table."""
        support = self._get_support_of(state, x)
        if support:
            unstack_act = self.all_actions[f"Unstack({x}, {support})"]
            self._apply_action(unstack_act, state, plan)

            putdown_act = self.all_actions[f"Putdown({x})"]
            self._apply_action(putdown_act, state, plan)
        elif ("Holding", x) in state.predicates:
            putdown_act = self.all_actions[f"Putdown({x})"]
            self._apply_action(putdown_act, state, plan)

    def _decompose_stack(self, top: str, base: str, state: WorldState, plan: List[Action]):
        """Primitive decomposition: picks up top and stacks it on base."""
        support = self._get_support_of(state, top)
        if support:
            unstack_act = self.all_actions[f"Unstack({top}, {support})"]
            self._apply_action(unstack_act, state, plan)
        else:
            pickup_act = self.all_actions[f"Pickup({top})"]
            self._apply_action(pickup_act, state, plan)

        stack_act = self.all_actions[f"Stack({top}, {base})"]
        self._apply_action(stack_act, state, plan)
