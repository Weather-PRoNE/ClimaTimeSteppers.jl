# ARS343 constants (16 significant digits) and tolerances shared by the
# first-gate tests; included by first_gate_ars343.jl and first_gate_newton.jl.
const γ = 0.4358665215084590
const b1 = 1.208496649176010
const b2 = -0.644363170684469
const a41 = -0.1058582960718796
const α = 0.5529291480359398  # a42 = a43

# Coefficient and stage-time matching: provisional rounding-count rationale,
# not a certified bound (D006). About 9 eps(1) for O(1) entries that src forms
# with a few roundings.
const coeff_atol = 2e-15
# Stage times of a run with state type FT: the same rationale with eps(FT) in
# place of eps(Float64) (identical to coeff_atol for Float64).
stage_atol(FT) = coeff_atol * (eps(FT) / eps(Float64))

# Results of one O(1) step with state type FT: provisional rounding-count
# rationale, not a certified bound (D006). 64 eps(FT) relative to the scale
# covers the few dozen roundings of one ARS343 step with margin.
provisional_tol(x, FT = Float64) = 64 * eps(FT) * abs(x)
