import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

open Finset BigOperators

/-!
Parametric proof of Theorem 1 (tree-exactness, all `k` and all real priors).

A star: a parent with `k` children, each child forcing the parent true. Worlds are
`(parent, children) : Bool × (Fin k → Bool)`, weighted by the product distribution with
parent prior `qp` and child priors `q i`. The exact WMC marginal of the parent equals the
soft-OR update  qp / (qp + (1-qp) * ∏ i, (1 - q i))  for every `k` and all real priors.
-/

namespace WmcStar

variable {k : ℕ}

/-- Weight of a world `(par, c)`. -/
def wt (qp : ℝ) (q : Fin k → ℝ) (par : Bool) (c : Fin k → Bool) : ℝ :=
  (if par then qp else 1 - qp) * ∏ i, (if c i then q i else 1 - q i)

/-- Valid iff every true child forces the parent true. -/
def valid (par : Bool) (c : Fin k → Bool) : Prop := ∀ i, c i = true → par = true

instance (par : Bool) (c : Fin k → Bool) : Decidable (valid par c) := by
  unfold valid; infer_instance

/-- Each child factor sums to 1, so the product over children sums to 1. -/
lemma sum_children_one (q : Fin k → ℝ) :
    ∑ c : Fin k → Bool, ∏ i, (if c i then q i else 1 - q i) = 1 := by
  have h := Finset.prod_univ_sum (fun (_ : Fin k) => (Finset.univ : Finset Bool))
              (fun i b => if b then q i else 1 - q i)
  rw [Fintype.piFinset_univ] at h
  rw [← h]
  refine Finset.prod_eq_one ?_
  intro i _
  simp [Fintype.sum_bool]

/-- Parent-true marginal numerator (every parent-true world is valid). -/
def num (qp : ℝ) (q : Fin k → ℝ) : ℝ :=
  ∑ c : Fin k → Bool, (if valid true c then wt qp q true c else 0)

/-- Partition function over all valid worlds. -/
def Z (qp : ℝ) (q : Fin k → ℝ) : ℝ :=
  ∑ par : Bool, ∑ c : Fin k → Bool, (if valid par c then wt qp q par c else 0)

lemma num_eq (qp : ℝ) (q : Fin k → ℝ) : num qp q = qp := by
  have hv : ∀ c : Fin k → Bool, valid true c := fun c i _ => rfl
  unfold num
  rw [Finset.sum_congr rfl (fun c _ => if_pos (hv c))]
  simp only [wt, if_true]
  rw [← Finset.mul_sum, sum_children_one, mul_one]

/-- The all-false child assignment is the only valid one when the parent is false. -/
lemma valid_false_iff (c : Fin k → Bool) : valid false c ↔ c = (fun _ => false) := by
  unfold valid
  constructor
  · intro h; funext i
    by_contra hne
    have hc : c i = true := by cases c i <;> simp_all
    exact absurd (h i hc) (by decide)
  · intro h i hi; rw [h] at hi; simp at hi

lemma Z_eq (qp : ℝ) (q : Fin k → ℝ) : Z qp q = qp + (1 - qp) * ∏ i, (1 - q i) := by
  unfold Z
  simp only [Fintype.sum_bool]
  -- f false + f true ; rewrite the parent-true sum as num = qp
  have htrue : (∑ c : Fin k → Bool, (if valid true c then wt qp q true c else 0)) = qp := num_eq qp q
  -- the parent-false sum collapses to the all-false world
  have hfalse : (∑ c : Fin k → Bool, (if valid false c then wt qp q false c else 0))
      = (1 - qp) * ∏ i, (1 - q i) := by
    rw [Finset.sum_eq_single (fun _ => false)]
    · rw [if_pos ((valid_false_iff _).mpr rfl)]
      simp [wt]
    · intro c _ hc
      rw [if_neg ?_]
      intro hv
      exact hc ((valid_false_iff c).mp hv)
    · intro h; exact absurd (Finset.mem_univ _) h
  rw [hfalse, htrue]

/-- Theorem 1: exact WMC marginal = soft-OR update, for all `k` and all priors. -/
theorem star_marginal_eq_softOR (qp : ℝ) (q : Fin k → ℝ) :
    num qp q / Z qp q = qp / (qp + (1 - qp) * ∏ i, (1 - q i)) := by
  rw [num_eq, Z_eq]

end WmcStar

#print axioms WmcStar.star_marginal_eq_softOR
#print axioms WmcStar.num_eq
#print axioms WmcStar.sum_children_one
