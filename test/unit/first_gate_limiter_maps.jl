#=
First-gate checks for the applied limited channel and the final maps (B001 T4).

Expected values are the frozen T0 expectations (B001 `t0-sections.md`,
Section 2 "applied limited channel" and Section 5 "final maps and endpoint
ordering"). They are typed in here, never taken from the solver. The targets
are exact integers in exact arithmetic; see `tol` for the Float64 bound.
=#
using ClimaTimeSteppers, Test
import ClimaTimeSteppers as CTS

# Section 2 setup: u0 = 2 U, dt = 1 s, T_lim = 4, T_exp = -2, T_imp = 0,
# L(x) = min(x, 3). Constant tendencies, so stage values cannot change them.
const u0 = 2.0
const u_raw = 6.0       # u0 + Σ b_exp T_lim dt = 2 + 4 (Σ b_exp = 1)
const u_limited = 3.0   # L(u_raw)
const Y_asm = 1.0       # u_limited + Σ b_exp T_exp dt = 3 - 2
L(x) = min(x, 3.0)

# Section 5 maps, applied after assembly as M_K ∘ M_C ∘ M_D.
M_D(x) = 2x + 1
M_C(x) = min(x, 2.0)
M_K(x) = x + 4
const Y_next = 6.0      # 1 -> 3 -> 2 -> 6

# The oracle is exact, but the solver's binary ARS343 weights need not sum to
# exactly 1, so each weighted sum carries a few roundings of size eps(|x|)
# with |x| <= 10 U (largest value in any run, cache map applied twice). A
# bound of 64 eps(10) ≈ 1.1e-13 U covers that and is 1e13 times smaller than
# every fault shift below (>= 1 U). Provisional, uncalibrated (D006).
const tol = 64 * eps(10.0)
near(x, y) = abs(x - y) <= tol

# Map hook that records every call and mutates only the integrator state, so
# the stage buffers and the init-time `cache!(u0)` call stay untouched; the
# recorded calls on the integrator state are the end-of-step final maps.
function map_hook(name, M, trace, target)
    return (u, p, t) -> begin
        u === target[] || return nothing
        before = u[1]
        u .= M.(u)
        push!(trace, (name, t, before, u[1]))
        return nothing
    end
end

# One ARS343 step of the Section 2 split problem. `lim` is the limiter body,
# `maps` the scalar maps run by (dss!, constrain_state!, cache!).
function limiter_maps_step(;
    lim = L,
    T_lim = 4.0,
    T_exp = -2.0,
    maps = (identity, identity, identity),
)
    trace = Tuple{Symbol, Float64, Float64, Float64}[]
    lim_calls = Tuple{Float64, Float64, Float64, Float64, Bool}[]
    target = Ref{Any}(nothing)
    temp = Ref{Any}(nothing)
    lim! = (u, p, t, u_ref) -> begin
        before = u[1]
        u .= lim.(u)
        push!(lim_calls, (t, before, u[1], u_ref[1], u === temp[]))
        return nothing
    end
    # T_imp = 0 with W = dtγ * 0 - I
    T_imp = CTS.ODEFunction(
        (du, u, p, t) -> (du .= 0);
        jac_prototype = zeros(1, 1),
        Wfact = (W, u, p, dtγ, t) -> (W[1, 1] = -1),
    )
    f = ClimaODEFunction(;
        T_exp! = (du, u, p, t) -> (du .= T_exp),
        T_lim! = (du, u, p, t) -> (du .= T_lim),
        T_imp! = T_imp,
        lim!,
        dss! = map_hook(:dss!, maps[1], trace, target),
        constrain_state! = map_hook(:constrain_state!, maps[2], trace, target),
        cache! = map_hook(:cache!, maps[3], trace, target),
    )
    prob = CTS.ODEProblem(f, [u0], (0.0, 1.0), nothing)
    alg = IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2))
    integrator = CTS.init(prob, alg; dt = 1.0, saveat = (1.0,))
    target[] = integrator.u
    temp[] = integrator.cache.temp
    CTS.solve!(integrator)
    return (; u1 = integrator.u[1], trace, lim_calls)
end

@testset "First-gate limiter and final maps" begin

    # t0-sections.md Section 2 (lines 179-207)
    @testset "Applied limited channel" begin
        (; u1, trace, lim_calls) = limiter_maps_step()
        @test near(u1, Y_asm)
        # Identity maps: each end-of-step map runs once and changes nothing.
        @test first.(trace) == [:dss!, :constrain_state!, :cache!]
        @test all(((_, _, b, a),) -> b == a, trace)

        # The endpoint limiter runs once (one solver-side application), on the
        # raw intermediate in the scratch buffer, relative to u0: 6 -> 3.
        endpoint_calls = filter(c -> c[5], lim_calls)
        @test length(endpoint_calls) == 1
        (t, before, after, u_ref, _) = only(endpoint_calls)
        @test t == 1.0 && u_ref == u0
        @test near(before, u_raw)
        @test after == u_limited             # L clips to exactly 3
        @test near(after - u_ref, 1.0)       # ΔY_lim,applied
        @test near(u1 - after, -2.0)         # unlimited explicit contribution

        @testset "Fault: limiter skipped" begin
            # Raw contribution 4 instead of applied 1: endpoint 4 U, residual -3 U.
            u1_fault = limiter_maps_step(; lim = identity).u1
            @test near(u1_fault - Y_asm, 3.0)
            @test abs(u1_fault - Y_asm) > tol
        end

        @testset "Fault: limit after unlimited increments" begin
            # Unlimited tendency moved into the limited channel:
            # min(2 + 4 - 2, 3) = 3 U instead of 1 U.
            u1_fault = limiter_maps_step(; T_lim = 4.0 - 2.0, T_exp = 0.0).u1
            @test near(u1_fault - Y_asm, 2.0)
            @test abs(u1_fault - Y_asm) > tol
        end
    end

    # t0-sections.md Section 5 (lines 294-328), starting from Y_asm = 1 U
    @testset "Final maps and ordering" begin
        golden_maps = (M_D, M_C, M_K)
        (; u1, trace) = limiter_maps_step(; maps = golden_maps)
        @test near(u1, Y_next)
        @test near(u1 - Y_asm, 5.0)  # total final-map change

        # Each map acts once, in the order dss!, constrain_state!, cache!, at
        # t_final, with the before/after trace of Section 5.
        @test [(n, t) for (n, t, _, _) in trace] ==
              [(:dss!, 1.0), (:constrain_state!, 1.0), (:cache!, 1.0)]
        befores = [b for (_, _, b, _) in trace]
        afters = [a for (_, _, _, a) in trace]
        @test all(near.(befores, [1.0, 3.0, 2.0]))
        @test all(near.(afters, [3.0, 2.0, 6.0]))
        @test all(near.(afters .- befores, [2.0, -1.0, 4.0]))

        @testset "Fault: first two maps swapped" begin
            # 1 -> 1 -> 3 -> 7 U
            u1_fault = limiter_maps_step(; maps = (M_C, M_D, M_K)).u1
            @test near(u1_fault - Y_next, 1.0)
            @test abs(u1_fault - Y_next) > tol
        end

        @testset "Fault: cache map omitted" begin
            # Endpoint 2 U, residual 4 U
            u1_fault = limiter_maps_step(; maps = (M_D, M_C, identity)).u1
            @test near(Y_next - u1_fault, 4.0)
            @test abs(u1_fault - Y_next) > tol
        end

        @testset "Fault: cache map applied twice" begin
            # Hand value from the Section 5 maps: 1 -> 3 -> 2 -> 6 -> 10 U.
            # (M_C twice is idempotent and M_D twice is clipped by M_C, so the
            # cache map is the one whose double application is visible.)
            u1_fault = limiter_maps_step(; maps = (M_D, M_C, M_K ∘ M_K)).u1
            @test near(u1_fault - Y_next, 4.0)
            @test abs(u1_fault - Y_next) > tol
        end
    end
end
