"""
Reactive Execution Layer for Warehouse Pallet Stacking.
Monitors execution step-by-step. If a physical disturbance occurs mid-execution
(e.g., box knocked over), the reactive layer repairs the violated precondition locally
instead of discarding the remaining plan and replanning from scratch.
"""

from typing import List, Tuple, Dict, Any, Callable, Optional
import copy
from .blocks_world import Action, WorldState, get_all_actions


class DisturbanceEvent:
    def __init__(self, step_trigger: int, description: str, state_mutation: Callable[[WorldState], None]):
        self.step_trigger = step_trigger
        self.description = description
        self.state_mutation = state_mutation


class ReactiveExecutionMonitor:
    def __init__(self, blocks: List[str]):
        self.blocks = blocks
        self.all_actions = {a.name: a for a in get_all_actions(blocks)}
        self.intervention_logs: List[str] = []

    def execute_with_monitoring(
        self,
        initial_state: WorldState,
        plan: List[Action],
        disturbances: Optional[List[DisturbanceEvent]] = None
    ) -> Tuple[WorldState, List[str], List[str]]:
        """
        Executes plan actions one by one.
        Returns:
            - final_state: WorldState
            - execution_trace: List of executed action descriptions
            - intervention_logs: Log messages from reactive interventions
        """
        disturbances = disturbances or []
        current_state = copy.deepcopy(initial_state)
        trace: List[str] = []
        self.intervention_logs.clear()

        i = 0
        remaining_plan = list(plan)

        step_counter = 0
        while remaining_plan:
            action = remaining_plan.pop(0)
            step_counter += 1

            # Check if a disturbance is scheduled for this step
            for dist in disturbances:
                if dist.step_trigger == step_counter:
                    log_msg = f"[DISTURBANCE @ step {step_counter}]: {dist.description}"
                    self.intervention_logs.append(log_msg)
                    trace.append(f"*** EVENT: {dist.description} ***")
                    dist.state_mutation(current_state)

            # Check if action's preconditions are satisfied
            if not action.is_applicable(current_state):
                missing_pre = action.pre - current_state.predicates
                repair_actions = self._react_to_missing_preconditions(action, missing_pre, current_state)
                
                log_entry = (
                    f"[REACTIVE INTERVENTION @ step {step_counter}]: Precondition(s) {missing_pre} "
                    f"for action '{action.name}' were invalidated. "
                    f"Local repair synthesized: {[a.name for a in repair_actions]}."
                )
                self.intervention_logs.append(log_entry)
                trace.append(f"--> REACTIVE INTERVENTION: Repairing {missing_pre} with {[a.name for a in repair_actions]}")

                # Execute repair actions
                for r_act in repair_actions:
                    current_state = r_act.apply(current_state)
                    trace.append(f"Executed Repair Action: {r_act.name} | New State: {sorted(list(current_state.predicates))}")

            # Now execute the original planned action
            if not action.is_applicable(current_state):
                raise RuntimeError(f"Reactive layer failed to re-establish preconditions for {action.name}")

            current_state = action.apply(current_state)
            trace.append(f"Executed Planned Action: {action.name} | New State: {sorted(list(current_state.predicates))}")

        return current_state, trace, self.intervention_logs

    def _react_to_missing_preconditions(
        self,
        action: Action,
        missing: set,
        state: WorldState
    ) -> List[Action]:
        """
        Local precondition repair without full replanning.
        Identifies minimal corrective actions to restore unsatisfied preconditions.
        """
        repair: List[Action] = []
        # Case 1: Arm is not empty because robot is holding something unexpected
        if ("ArmEmpty",) in missing:
            holding_block = next((p[1] for p in state.predicates if p[0] == "Holding"), None)
            if holding_block:
                putdown = self.all_actions[f"Putdown({holding_block})"]
                repair.append(putdown)
                state = putdown.apply(state)

        # Case 2: Block to pick up or stack on is not Clear (something fell onto it)
        for p in list(missing):
            if p[0] == "Clear":
                obstructing_block = p[1]
                # Find what is on top of obstructing_block
                block_on_top = next((pred[1] for pred in state.predicates if pred[0] == "On" and pred[2] == obstructing_block), None)
                if block_on_top:
                    # Unstack block_on_top to table
                    if ("ArmEmpty",) not in state.predicates:
                        h = next((pred[1] for pred in state.predicates if pred[0] == "Holding"), None)
                        if h:
                            pd = self.all_actions[f"Putdown({h})"]
                            repair.append(pd)
                            state = pd.apply(state)
                    unstack = self.all_actions[f"Unstack({block_on_top}, {obstructing_block})"]
                    repair.append(unstack)
                    state = unstack.apply(state)
                    putdown = self.all_actions[f"Putdown({block_on_top})"]
                    repair.append(putdown)
                    state = putdown.apply(state)

        # Case 3: Block needed on table was placed or fell onto another block
        for p in list(missing):
            if p[0] == "OnTable":
                block = p[1]
                current_support = next((pred[2] for pred in state.predicates if pred[0] == "On" and pred[1] == block), None)
                if current_support:
                    # Clear top of block if needed
                    top_b = next((pred[1] for pred in state.predicates if pred[0] == "On" and pred[2] == block), None)
                    if top_b:
                        u = self.all_actions[f"Unstack({top_b}, {block})"]
                        repair.append(u)
                        state = u.apply(state)
                        pd = self.all_actions[f"Putdown({top_b})"]
                        repair.append(pd)
                        state = pd.apply(state)
                    u2 = self.all_actions[f"Unstack({block}, {current_support})"]
                    repair.append(u2)
                    state = u2.apply(state)
                    pd2 = self.all_actions[f"Putdown({block})"]
                    repair.append(pd2)
                    state = pd2.apply(state)

        return repair
