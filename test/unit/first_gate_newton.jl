# First-gate check of the finite-Newton defect at the second ARS343 stage of
# f_exp = 0, f_imp(u) = -u² / (1 U s), u0 = 1 U, h = dt γ = 1/4 s, after one
# undamped Newton iteration from the baseline B = 1 U. Expected values are exact
# rationals typed in here; the converged root -2 + 2√2 U is not the expectation.
# White-box: reads the stored stage tendencies `integrator.cache.T_imp`
# (`IMEXARKCache`), set by `evaluate_stage_tendencies!` to (U - temp) / dtγ.
#
# One iteration from the baseline with the exact Jacobian follows from the
# `NewtonsMethod` defaults in `solve_newton!`: `max_iters = 1`, no
# `convergence_checker` (no early exit), no `line_search` (undamped step), and
# `update_j = UpdateEvery(NewNewtonIteration)` (`Wfact` at the iterate).
# `solve_stage_implicit!` starts from the stage value of `compute_stage_value!`,
# which the default no-op `initialize_imp!`, `dss!`, `constrain_state!` and
# `cache_imp!` leave unchanged. The residual is U₀ + dtγ T(U′) - U′, so `Wfact`
# returns W = dtγ J - I with J = ∂f_imp/∂u = -2u / (1 U s).
using ClimaTimeSteppers, Test
import ClimaTimeSteppers as CTS

# ARS343 constants (16 significant digits)
const γ = 0.4358665215084590
const b1 = 1.208496649176010
const b2 = -0.644363170684469
const b_imp = (0.0, b1, b2, γ)
const coeff_atol = 2e-15

# Problem in numerical unit coordinates (U = 1, s = 1)
f_imp(u) = -u^2
const u0 = 1.0
const dt = 1 / (4γ)

# Exact expectations: one Newton step U = B - R(B) / W(B) with R(B) = h f_imp(B),
# W(B) = -(1 + 2hB); defect D = U - B - h f_imp(U)
const B = 1 // 1
const h = 1 // 4
const U_expected = 5 // 6
const T_physical_expected = -25 // 36
const D_expected = 1 // 144
const T_eff_expected = -2 // 3
const U_converged = -2 + 2√2

# Tolerances. States are O(1 U): the Newton
# step, dt = 1/(4γ) and dtγ each round within a few ulps of their O(1) values,
# so 64 eps of the state scale |B| = 1 U bounds the state error with margin.
# Tendencies are state differences divided by h = 1/4 s, which multiplies the
# absolute error by 1/h = 4. Both are below 1/144 by about 11 orders.
const tol_state = 64 * eps(Float64) * abs(B)
const tol_tend = tol_state / h

# One ARS343 step with a recording explicit tendency f_exp = 0. Returns the
# recorded (t, u) stage calls, the stored implicit stage tendencies and the
# endpoint.
function one_step()
    calls = Tuple{Float64, Float64}[]
    T_exp! = (du, u, p, t) -> (push!(calls, (t, u[1])); du .= 0)
    # W = dtγ J - I with J = f_imp'(u) = -2u, evaluated at the current iterate
    Wfact!(W, u, p, dtγ, t) = (W[1, 1] = dtγ * (-2 * u[1]) - 1)
    T_imp = CTS.ODEFunction(
        (du, u, p, t) -> (du .= f_imp.(u));
        jac_prototype = zeros(1, 1),
        Wfact = Wfact!,
    )
    f = ClimaODEFunction(; T_exp!, T_imp! = T_imp)
    prob = CTS.ODEProblem(f, [u0], (0.0, dt), nothing)
    alg = IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 1))
    integrator = CTS.init(prob, alg; dt, saveat = (dt,))
    CTS.solve!(integrator)
    T_imp_stored = [integrator.cache.T_imp[j][1] for j in 2:4]
    return (; calls, T_imp_stored, u1 = integrator.u[1])
end

@testset "First-gate finite-Newton defect" begin
    (; calls, T_imp_stored, u1) = one_step()

    # Stage 2 runs at explicit time dt c_exp[2] = dt γ = h; its value U is
    # seen by the explicit tendency after the Newton solve.
    U_obs = [u for (t, u) in calls if abs(t - h) <= coeff_atol]
    T_eff_obs = T_imp_stored[1]

    @testset "Stage-2 state and stored tendency" begin
        @test length(U_obs) == 1
        U = only(U_obs)
        T_physical = f_imp(U)  # independent evaluation, not the solver's
        D = U - B - h * T_physical

        @test abs(U - U_expected) <= tol_state
        @test abs(T_physical - T_physical_expected) <= tol_tend
        @test abs(D - D_expected) <= tol_state
        @test abs(T_eff_obs - T_eff_expected) <= tol_tend
        @test abs((T_eff_obs - T_physical) - D_expected / h) <= tol_tend
        @test abs((T_eff_obs - T_physical) - D / h) <= tol_tend
        # The defect is explicitly nonzero, and U is not the converged root.
        @test abs(D) > tol_state
        @test abs(U - U_converged) > tol_state
    end

    # Negative controls: each faulty requirement fails by more than the
    # tolerance, by the predicted amount.
    @testset "Negative controls" begin
        U = only(U_obs)
        T_physical = f_imp(U)
        D = U - B - h * T_physical

        # Requiring zero defect fails by D = 1/144 U.
        @test !(abs(D - 0) <= tol_state)
        @test abs(abs(D - 0) - D_expected) <= tol_state

        # Storage replaced by the physical tendency at stage level misses the
        # stored value by D/h = 1/36 U/s.
        @test !(abs(T_physical - T_eff_obs) <= tol_tend)
        @test abs((T_eff_obs - T_physical) - 1 // 36) <= tol_tend

        # Assembly: with f_exp = 0 the endpoint is u0 + dt Σ b_imp[j] T_imp[j]
        # (assembly in `step_u!`); b_imp[1] = 0. The stored tendencies
        # reproduce the solver endpoint ...
        tol_u1 = 64 * eps(Float64) * (abs(u0) + dt * sum(abs, b_imp[2:4] .* T_imp_stored))
        u1_assembled = u0 + dt * sum(b_imp[2:4] .* T_imp_stored)
        @test abs(u1_assembled - u1) <= tol_u1
        # ... while T_physical in place of T_eff at stage 2 shifts it by
        # dt b1 (T_physical - T_eff) = -dt b1 D / h = -b1 D / γ = -b1 / (144 γ).
        T_swapped = [T_physical; T_imp_stored[2:3]]
        u1_fault = u0 + dt * sum(b_imp[2:4] .* T_swapped)
        shift_expected = -b1 / (144γ)
        @test abs((u1_fault - u1) - shift_expected) <= tol_u1
        @test !(abs(u1_fault - u1) <= tol_u1)
    end
end
