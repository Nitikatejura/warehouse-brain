"""
Module D: Comparative Analysis — Hopfield Network vs. RNN vs. Symbolic AI.
Practical bridge connecting Connectionist Models to Symbolic AI.

========================================================================================================================
COMPARISON TABLE: HOPFIELD NETWORK vs. RECURRENT NEURAL NETWORK (RNN/LSTM) vs. SYMBOLIC SYSTEMS
========================================================================================================================
Dimension                  | Hopfield Network (From Scratch)       | Recurrent Neural Network (PyTorch LSTM) | Classical Symbolic AI (Modules A & E)
---------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------
1. Core Task Paradigm      | Auto-associative pattern completion   | Sequence classification / forecasting   | Deductive logic, search, constraint sat
                           | & static noise filtering.             | across temporal trajectories.           | over discrete predicate states.
---------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------
2. Temporal Dynamics       | Static: relaxes to fixed-point energy | Dynamic: maintains hidden state h_t     | Discrete transitions over state graph
                           | minimum attractor (no concept of time)| parameterized by recurrent gates.       | governed by action preconditions/effects.
---------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------
3. Data Representation     | Distributed bipolar vectors           | Distributed continuous embedding        | Explicit propositional / first-order
                           | x in {-1, +1}^N (e.g. 64-bit labels). | vectors h in R^d over continuous inputs.| predicates: On(A, B), Clear(x).
---------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------
4. Storage / Memory Limit  | Theoretical capacity limit ~0.14 * N  | Parametric memory distributed across    | Unbounded factual memory (KB size),
                           | (spurious local minima if exceeded).  | weight tensors; handles long sequences. | limited only by search tree complexity.
---------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------
5. What Hopfield CAN do    | - Instant one-shot associative recall | - Generalizes across unseen sequential  | - Guarantees 100% sound deduction.
   that others CANNOT      |   using Hebbian outer-product rule.   |   feature variations.                   | - Generates explicit causal explanations
                           | - Proven convergence via Lyapunov     | - Detects temporal drift & trends       |   and "WHY" traces.
                           |   energy function E = -0.5 s^T W s.   |   prior to physical failure.            | - Allows hot-swapping domain rules.
---------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------
6. What Hopfield CANNOT do | - Cannot process temporal ordering    | - Can predict imminent anomalies from   | - Cannot naturally handle noisy pixels
   that RNN CAN do         |   (e.g., vibration increasing vs dec).|   temporal slope & cross-channel trends.|   or continuous sensor fuzziness without
                           | - Cannot handle continuous features.  | - Can process continuous sensor metrics.|   discretization.
---------------------------+---------------------------------------+-----------------------------------------+-----------------------------------------
7. Explainability /        | Black-box / energy basin geometry;    | Latent state activations;               | Glass-box: complete proof trees and
   Interpretability        | cannot explain "why" a bit flipped.   | difficult to inspect directly.          | rule chains (Rule R1 fired because P).
========================================================================================================================
"""

def print_connectionist_symbolic_bridge():
    """Prints the connectionist vs symbolic comparison table to terminal."""
    print(__doc__)
