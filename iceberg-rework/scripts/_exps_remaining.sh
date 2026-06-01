# _exps_remaining.sh: shared list of (exp_id, training_timestamp) pairs for
# the seven distinct Phase A backbones not covered by re_phase_b_with_a{0,1,7b}.sh.
# Sourced by re_phase_b_all_remaining.sh and re_phase_b_tophat_all_remaining.sh.
#
# Empirical 1x3 collapse documented in T1 / T1b: A6 == A5, A8 == A9 == A7,
# A6a == A5a, A8a == A9a == A7a, A8b == A9b == A7b. Only the seven distinct
# backbones run.

EXPS_REMAINING=(
    "exp_A2_our_lt65|20260428_094654"
    "exp_A3_our_lt65_plus_nulls|20260429_234518"
    "exp_A4_our_lt65_plus_nulls_aug|20260429_234835"
    "exp_A5_our_lt65_plus_nulls_aug_2pos|20260430_001810"
    "exp_A7_our_lt65_plus_nulls_aug_size|20260430_002545"
    "exp_A5a_a1_2pos|20260505_205201"
    "exp_A7a_a1_size|20260505_210018"
)
