-- Five known theorems for testing lean_eval on a 5-record fixture.
-- Each theorem uses a single tactic that the fine-tuned model should predict.
-- Run with: lean eval/lean_harness/Fixture.lean

theorem fixture_01 (n : Nat) : n + 0 = n := by simp
theorem fixture_02 (n : Nat) : 0 + n = n := by simp
theorem fixture_03 : True := by trivial
theorem fixture_04 : 1 + 1 = 2 := by rfl
theorem fixture_05 (a b : Nat) (h : a = b) : b = a := by exact h.symm
