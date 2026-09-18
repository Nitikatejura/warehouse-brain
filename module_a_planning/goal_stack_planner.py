"""
Goal Stack Planning (GSP) implementation for Blocks World.
Classical linear planning method using a stack of goals, subgoals, and operators.
"""

from typing import List, Set, Tuple, Optional, Dict
import copy
from .blocks_world import Action, WorldState, get_all_actions


class GoalStackPlanner:
    def __init__(self, blocks: List[str], max_depth: int = 200):
        self.blocks = blocks
        self.actions = get_all_actions(blocks)
        self.max_depth = max_depth

    def plan(self, initial_state: WorldState, goal_predicates: Set[Tuple]) -> List[Action]:
        """
        Generates a sequence of actions from initial_state to goal_predicates.
        Supports goal reordering if the initial attempt gets stuck (classic Sussman anomaly handling).
        """
        # We can try different orderings of the top-level goal predicates if needed
        goal_list = list(goal_predicates)
        
        # Priority order heuristic for Blocks World goals:
        # Build from the bottom up (e.g. OnTable or lower On before higher On)
        def goal_priority(pred):
            if pred[0] == "OnTable":
                return 0
            elif pred[0] == "On":
                return 1
            elif pred[0] == "Clear":
                return 2
            return 3

        sorted_goals = sorted(goal_list, key=goal_priority)

        plan = self._solve_stack(initial_state, sorted_goals)
        if plan is not None:
            return plan

        # Try reverse ordering if needed
        plan = self._solve_stack(initial_state, list(reversed(sorted_goals)))
        if plan is not None:
            return plan

        raise RuntimeError(f"Goal Stack Planner failed to find a plan within {self.max_depth} steps.")

    def _solve_stack(self, initial_state: WorldState, top_goals: List[Tuple]) -> Optional[List[Action]]:
        current_state = copy.deepcopy(initial_state)
        # Stack items can be:
        # - Action: primitive action to execute
        # - Tuple (predicate): single subgoal
        # - List[Tuple]: compound goal (all subgoals must hold simultaneously)
        stack = []
        stack.append(list(top_goals))
        for g in reversed(top_goals):
            stack.append(g)

        plan: List[Action] = []
        steps = 0

        while stack and steps < self.max_depth:
            steps += 1
            top = stack.pop()

            if isinstance(top, Action):
                # An action is popped only when its preconditions are met
                if top.is_applicable(current_state):
                    current_state = top.apply(current_state)
                    plan.append(top)
                else:
                    # Precondition violated due to subsequent actions, push back
                    return None

            elif isinstance(top, list):
                # Compound goal
                all_satisfied = all(p in current_state.predicates for p in top)
                if not all_satisfied:
                    # Repush compound goal and then individual unsatisfied goals
                    stack.append(top)
                    for p in reversed(top):
                        if p not in current_state.predicates:
                            stack.append(p)

            elif isinstance(top, tuple):
                # Atomic predicate
                if top in current_state.predicates:
                    continue
                else:
                    # Find an action that adds 'top'
                    candidate_actions = [a for a in self.actions if top in a.add]
                    if not candidate_actions:
                        return None
                    
                    # Choose candidate that has minimal conflicts
                    chosen_action = candidate_actions[0]
                    for cand in candidate_actions:
                        if cand.is_applicable(current_state):
                            chosen_action = cand
                            break

                    stack.append(chosen_action)
                    pre_list = sorted(list(chosen_action.pre), key=lambda x: 0 if x[0] == "Clear" else 1)
                    stack.append(pre_list)
                    for pre in reversed(pre_list):
                        stack.append(pre)

        # Verify final state
        if all(g in current_state.predicates for g in top_goals):
            return plan
        return None
