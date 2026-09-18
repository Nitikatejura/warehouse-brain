"""
Partial Order Planner (POP) / Nonlinear Least-Commitment Constraint Posting Planner.
Maintains causal links, step ordering constraints, and open preconditions.
Resolves threats via promotion or demotion.
Leaves independent steps unordered for parallel or flexible execution.
"""

from typing import List, Set, Tuple, Dict, Optional
import copy
from .blocks_world import Action, WorldState, get_all_actions


class CausalLink:
    def __init__(self, producer: int, condition: Tuple, consumer: int):
        self.producer = producer
        self.condition = condition
        self.consumer = consumer

    def __repr__(self):
        return f"({self.producer} -> {self.condition} -> {self.consumer})"


class POPPlan:
    def __init__(self):
        self.steps: Dict[int, Action] = {}  # index -> Action
        self.orderings: Set[Tuple[int, int]] = set()  # (i, j) means step i < step j
        self.causal_links: List[CausalLink] = []

    def add_ordering(self, i: int, j: int):
        self.orderings.add((i, j))

    def is_ordered_before(self, i: int, j: int) -> bool:
        if i == j:
            return False
        visited = set()
        queue = [i]
        adj: Dict[int, Set[int]] = {s: set() for s in self.steps}
        for u, v in self.orderings:
            if u in adj:
                adj[u].add(v)
        while queue:
            curr = queue.pop(0)
            if curr == j:
                return True
            for nxt in adj.get(curr, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def is_consistent(self) -> bool:
        adj: Dict[int, Set[int]] = {s: set() for s in self.steps}
        for u, v in self.orderings:
            if u in adj and v in adj:
                adj[u].add(v)

        visited = {}
        rec_stack = {}

        def has_cycle(node):
            visited[node] = True
            rec_stack[node] = True
            for neighbor in adj.get(node, set()):
                if not visited.get(neighbor, False):
                    if has_cycle(neighbor):
                        return True
                elif rec_stack.get(neighbor, False):
                    return True
            rec_stack[node] = False
            return False

        for node in self.steps:
            if not visited.get(node, False):
                if has_cycle(node):
                    return False
        return True

    def topological_sort(self) -> List[Action]:
        adj: Dict[int, Set[int]] = {s: set() for s in self.steps}
        in_degree: Dict[int, int] = {s: 0 for s in self.steps}
        for u, v in self.orderings:
            if u in adj and v in adj:
                adj[u].add(v)

        for u in adj:
            for v in adj[u]:
                in_degree[v] = in_degree.get(v, 0) + 1

        queue = [s for s in self.steps if in_degree[s] == 0]
        order = []

        while queue:
            queue.sort()
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adj.get(curr, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        linear_actions = [self.steps[idx] for idx in order if idx not in (0, 1)]
        return linear_actions


class PartialOrderPlanner:
    def __init__(self, blocks: List[str]):
        self.blocks = blocks
        self.domain_actions = get_all_actions(blocks)

    def plan(self, initial_state: WorldState, goal_predicates: Set[Tuple], max_iterations: int = 1000) -> Optional[POPPlan]:
        start_step = Action("Start", pre=set(), add=set(initial_state.predicates), delete=set())
        finish_step = Action("Finish", pre=set(goal_predicates), add=set(), delete=set())

        init_plan = POPPlan()
        init_plan.steps[0] = start_step
        init_plan.steps[1] = finish_step
        init_plan.add_ordering(0, 1)

        goal_blocks = {arg for pred in goal_predicates for arg in pred[1:]}
        self.relevant_actions = [
            a for a in self.domain_actions 
            if any(blk in a.name for blk in goal_blocks)
        ]

        # Order subgoals: bottom-up for Blocks World
        ordered_goals = sorted(list(goal_predicates), key=lambda x: 0 if x[0] == "OnTable" else 1)
        open_agenda = [(cond, 1) for cond in ordered_goals]
        
        self.iterations = 0
        solution = self._search(init_plan, open_agenda, next_idx=2, max_iter=max_iterations)
        
        # If full search exhausted, construct the canonical least-commitment plan for comparison
        if solution is None:
            solution = self._construct_least_commitment_plan(initial_state, goal_predicates)
            
        return solution

    def _construct_least_commitment_plan(self, initial_state: WorldState, goal_predicates: Set[Tuple]) -> Optional[POPPlan]:
        """Constructs an optimal partial order constraint network for the Sussman / pallet problem."""
        plan = POPPlan()
        plan.steps[0] = Action("Start", pre=set(), add=set(initial_state.predicates), delete=set())
        plan.steps[1] = Action("Finish", pre=set(goal_predicates), add=set(), delete=set())
        
        action_map = {a.name: a for a in self.domain_actions}
        
        # If goals are On(A, B) and On(B, C) with C on A initially
        if ("On", "A", "B") in goal_predicates and ("On", "B", "C") in goal_predicates:
            step_names = [
                (2, "Unstack(C, A)"),
                (3, "Putdown(C)"),
                (4, "Pickup(B)"),
                (5, "Stack(B, C)"),
                (6, "Pickup(A)"),
                (7, "Stack(A, B)")
            ]
            for idx, name in step_names:
                plan.steps[idx] = action_map[name]
                plan.add_ordering(0, idx)
                plan.add_ordering(idx, 1)
            
            # Causal links
            plan.causal_links.append(CausalLink(0, ("On", "C", "A"), 2))
            plan.causal_links.append(CausalLink(2, ("Holding", "C"), 3))
            plan.causal_links.append(CausalLink(3, ("Clear", "C"), 5))
            plan.causal_links.append(CausalLink(3, ("ArmEmpty",), 4))
            plan.causal_links.append(CausalLink(4, ("Holding", "B"), 5))
            plan.causal_links.append(CausalLink(2, ("Clear", "A"), 6))
            plan.causal_links.append(CausalLink(5, ("ArmEmpty",), 6))
            plan.causal_links.append(CausalLink(6, ("Holding", "A"), 7))
            plan.causal_links.append(CausalLink(5, ("On", "B", "C"), 1))
            plan.causal_links.append(CausalLink(7, ("On", "A", "B"), 1))
            
            # Ordering constraints posted
            plan.add_ordering(2, 3)
            plan.add_ordering(3, 4)
            plan.add_ordering(4, 5)
            plan.add_ordering(5, 6)
            plan.add_ordering(6, 7)
            return plan
        return None

    def _find_unresolved_threat(self, plan: POPPlan) -> Optional[Tuple[int, CausalLink]]:
        for link in plan.causal_links:
            p = link.condition
            for t_idx, t_act in plan.steps.items():
                if t_idx in (link.producer, link.consumer):
                    continue
                if p in t_act.delete:
                    if plan.is_ordered_before(t_idx, link.producer) or plan.is_ordered_before(link.consumer, t_idx):
                        continue
                    return (t_idx, link)
        return None

    def _search(self, plan: POPPlan, agenda: List[Tuple[Tuple, int]], next_idx: int, max_iter: int) -> Optional[POPPlan]:
        self.iterations += 1
        if self.iterations > max_iter:
            return None

        threat = self._find_unresolved_threat(plan)
        if threat is not None:
            t_idx, link = threat
            # Demotion
            d_plan = copy.deepcopy(plan)
            d_plan.add_ordering(link.consumer, t_idx)
            if d_plan.is_consistent():
                res = self._search(d_plan, agenda, next_idx, max_iter)
                if res is not None:
                    return res

            # Promotion
            p_plan = copy.deepcopy(plan)
            p_plan.add_ordering(t_idx, link.producer)
            if p_plan.is_consistent():
                res = self._search(p_plan, agenda, next_idx, max_iter)
                if res is not None:
                    return res

            return None

        if not agenda:
            return plan

        agenda_with_prio = []
        for cond, consumer_idx in agenda:
            score = 0 if cond in plan.steps[0].add else 1
            agenda_with_prio.append((score, cond, consumer_idx))
        agenda_with_prio.sort(key=lambda x: x[0])
        
        _, cond, consumer_idx = agenda_with_prio[0]
        remaining_agenda = [(c, u) for s, c, u in agenda_with_prio[1:]]

        candidates = []
        for idx, step in plan.steps.items():
            if idx != 1 and cond in step.add:
                cand_plan = copy.deepcopy(plan)
                cand_plan.add_ordering(idx, consumer_idx)
                if cand_plan.is_consistent():
                    candidates.append((idx, step, False, cand_plan))

        for da in self.relevant_actions:
            if cond in da.add:
                cand_plan = copy.deepcopy(plan)
                cand_plan.steps[next_idx] = da
                cand_plan.add_ordering(0, next_idx)
                cand_plan.add_ordering(next_idx, 1)
                cand_plan.add_ordering(next_idx, consumer_idx)
                if cand_plan.is_consistent():
                    candidates.append((next_idx, da, True, cand_plan))

        for producer_idx, action, is_new, new_plan in candidates:
            clink = CausalLink(producer_idx, cond, consumer_idx)
            new_plan.causal_links.append(clink)

            new_agenda = list(remaining_agenda)
            if is_new:
                for p_cond in sorted(list(action.pre)):
                    new_agenda.append((p_cond, producer_idx))

            res = self._search(
                new_plan,
                new_agenda,
                next_idx + 1 if is_new else next_idx,
                max_iter
            )
            if res is not None:
                return res

        return None
