/-
  Kernel-checked development for "Modular WMC for NeSy over large ontologies".

  No `mathlib`, no `native_decide`: every theorem is closed by `decide`, i.e. reduced in
  the Lean kernel. To keep rational arithmetic cheap in the kernel we use a custom rational
  encoding `Q = Int x Int` (numerator, denominator) whose equality is the cross-multiplied
  integer test `a.num * b.den = b.num * a.den` (no fraction normalization). Weighted model
  counts are explicit sums over enumerated Boolean worlds, so the exact WMC marginal, the
  soft-OR (belief-propagation) update, and their (in)equality are all decidable.

  Verified facts (the concrete instances behind Theorem 1 and the reconvergence claim):
    * single_edge_exact      : on one edge, the soft-OR update equals the exact WMC marginal.
    * star2_exact, star3_exact: on a k-child star (a tree), soft-OR = exact WMC marginal.
    * diamond_exact_5_6      : the diamond's reconvergent node has exact WMC marginal 5/6.
    * diamond_softor_9_10    : the soft-OR update (paths treated independently) gives 9/10.
    * diamond_softor_ne_exact: hence soft-OR != exact at the reconvergence (5/6 != 9/10).
-/

namespace Wmc

/-- A rational as an unnormalized numerator/denominator pair. -/
structure Q where
  num : Int
  den : Int
deriving Repr

namespace Q

def mk' (n d : Int) : Q := ⟨n, d⟩

/-- Cross-multiplied equality as a decidable Bool; no fraction normalization. All theorems
    compare with this (kept inside the kernel via `decide`). -/
def beq (a b : Q) : Bool := decide (a.num * b.den = b.num * a.den)

def add (a b : Q) : Q := ⟨a.num * b.den + b.num * a.den, a.den * b.den⟩
def mul (a b : Q) : Q := ⟨a.num * b.num, a.den * b.den⟩
/-- 1 - a -/
def compl (a : Q) : Q := ⟨a.den - a.num, a.den⟩

instance : Add Q := ⟨add⟩
instance : Mul Q := ⟨mul⟩

def zero : Q := ⟨0, 1⟩
def one : Q := ⟨1, 1⟩

end Q

open Q

/-- A world is a fixed-length list of Booleans (truth of each atom). -/
abbrev World := List Bool

/-- All worlds over `n` atoms. -/
def worlds : Nat → List World
  | 0 => [[]]
  | n+1 => (worlds n).flatMap (fun w => [false :: w, true :: w])

/-- Weight of a world: product over atoms of (p_i if true else 1 - p_i). -/
def weight (priors : List Q) (w : World) : Q :=
  match priors, w with
  | [], [] => Q.one
  | p :: ps, b :: bs => (if b then p else Q.compl p) * weight ps bs
  | _, _ => Q.one

/-- A clause is a list of (atom-index, expected-polarity); satisfied if any literal matches. -/
abbrev Clause := List (Nat × Bool)

def litSat (w : World) (lit : Nat × Bool) : Bool :=
  w.getD lit.fst false == lit.snd

def clauseSat (w : World) (c : Clause) : Bool := c.any (litSat w)

def theorySat (w : World) (T : List Clause) : Bool := T.all (clauseSat w)

/-- Sum of weights of worlds satisfying `T` and the extra predicate `p`. -/
def wmcSum (n : Nat) (priors : List Q) (T : List Clause) (p : World → Bool) : Q :=
  (worlds n).foldl (fun acc w => if theorySat w T && p w then acc + weight priors w else acc) Q.zero

/-- Exact WMC marginal numerator (atom `i` true) and partition function. -/
def margNum (n : Nat) (priors : List Q) (T : List Clause) (i : Nat) : Q :=
  wmcSum n priors T (fun w => w.getD i false)
def partZ (n : Nat) (priors : List Q) (T : List Clause) : Q :=
  wmcSum n priors T (fun _ => true)

/-- soft-OR update for a parent with given prior and a list of child marginals:
    q / (q + (1-q) * prod (1 - child)). Returned as a `Q`. -/
def softOR (qp : Q) (children : List Q) : Q :=
  let prod := children.foldl (fun acc c => acc * Q.compl c) Q.one
  let denom := qp + Q.compl qp * prod
  ⟨qp.num * denom.den, qp.den * denom.num⟩   -- qp / denom

/-- Marginal as a single `Q` ratio: margNum / partZ. -/
def marginal (n : Nat) (priors : List Q) (T : List Clause) (i : Nat) : Q :=
  let a := margNum n priors T i
  let z := partZ n priors T
  ⟨a.num * z.den, a.den * z.num⟩

/- ===================== single edge: child(0) -> parent(1) ===================== -/
/-  clause (¬child ∨ parent) = [(0,false),(1,true)] ; priors both 1/2 -/
def half : Q := ⟨1, 2⟩

theorem single_edge_exact :
    Q.beq (marginal 2 [half, half] [[(0, false), (1, true)]] 1)
          (softOR half [half]) = true := by decide

/- ===================== star: parent(0), k children ===================== -/
/-  child j -> parent : clause (¬child_j ∨ parent) = [(j,false),(0,true)] -/
def starT (k : Nat) : List Clause := (List.range k).map (fun j => [(j+1, false), (0, true)])

theorem star2_exact :
    Q.beq (marginal 3 [half, half, half] (starT 2) 0) (softOR half [half, half]) = true := by
  decide

theorem star3_exact :
    Q.beq (marginal 4 [half, half, half, half] (starT 3) 0)
          (softOR half [half, half, half]) = true := by decide

/- ===================== diamond: d=3 -> b=1,c=2 ; b,c -> a=0 ===================== -/
/-  clauses: (¬d∨b),(¬d∨c),(¬b∨a),(¬c∨a) ; atoms a,b,c,d = 0,1,2,3 ; uniform 1/2 -/
def diamondT : List Clause :=
  [[(3, false), (1, true)], [(3, false), (2, true)], [(1, false), (0, true)], [(2, false), (0, true)]]

def diamondP : List Q := [half, half, half, half]

theorem diamond_exact_5_6 :
    Q.beq (marginal 4 diamondP diamondT 0) ⟨5, 6⟩ = true := by decide

/-- soft-OR applied at the reconvergence: children b,c each have soft-OR value 2/3
    (their own star over d), then a = softOR(1/2, [2/3, 2/3]) = 9/10. -/
theorem diamond_softor_9_10 :
    Q.beq (softOR half [softOR half [half], softOR half [half]]) ⟨9, 10⟩ = true := by decide

theorem diamond_softor_ne_exact :
    Q.beq (marginal 4 diamondP diamondT 0)
          (softOR half [softOR half [half], softOR half [half]]) = false := by decide

end Wmc

#print axioms Wmc.diamond_exact_5_6
#print axioms Wmc.star3_exact
#print axioms Wmc.single_edge_exact
#print axioms Wmc.diamond_softor_ne_exact
