# First-gate checks of the ARS343 coefficients and one-step assembly. Expected
# values are typed in or derived inline, never taken from the solver.
# White-box: reads tableau entries through the internal `CTS.SparseCoeffs` field
# `coeffs`.
using ClimaTimeSteppers, Test, LinearAlgebra
using StaticArrays
import ClimaTimeSteppers as CTS

# Helper to extract dense Float64 arrays from SparseCoeffs (as in tableaus.jl)
dense(sc::CTS.SparseCoeffs) = Array(Float64.(sc.coeffs))

include("first_gate_constants.jl")

# Manufactured linear problem: f_exp(u, t) = 1, f_imp(u, t) = u / (2γ), u0 = 0,
# one step with dt = 1. Stage by stage, U4 = 2H and u1 = 2H with
# H = (a41 + 2α) + b1 + b2 / γ.
const H = (a41 + 2α) + b1 + b2 / γ
const u1_expected = 2H

# FT is the state type (state, time, step and Jacobian); the tableau stays
# Float64 (default `cast_tableau_to_state_eltype = false`).
function manufactured_endpoint(alg; T_exp! = (du, u, p, t) -> (du .= 1), FT = Float64)
    # W = dtγ * J - I with J = 1 / (2γ)
    Wfact!(W, u, p, dtγ, t) = (W[1, 1] = dtγ / (2γ) - 1)
    T_imp = CTS.ODEFunction(
        (du, u, p, t) -> (du .= u ./ (2γ));
        jac_prototype = zeros(FT, 1, 1),
        Wfact = Wfact!,
    )
    f = ClimaODEFunction(; T_exp!, T_imp! = T_imp)
    prob = CTS.ODEProblem(f, [zero(FT)], (zero(FT), one(FT)), nothing)
    sol = CTS.solve(prob, alg; dt = one(FT), saveat = (one(FT),))
    return sol.u[end][1]
end

# Stage recorder: the explicit tendency is called with each stage value and its
# explicit stage time t + dt * c_exp[i], so the recorded (t, u) pairs expose the
# stage values. Returns the record and a T_exp! computing du = f_exp(t).
function stage_recorder(f_exp)
    calls = Tuple{Float64, Float64}[]
    T_exp! = (du, u, p, t) -> (push!(calls, (t, u[1])); du .= f_exp(t))
    return calls, T_exp!
end

# Stage times come from row sums, so match them to c within coefficient
# tolerance rather than exactly.
stage_values(calls, c; atol = coeff_atol) = [u for (t, u) in calls if abs(t - c) <= atol]

# Harness copy of the ARS343 tableau, built through the public keyword
# constructor; `modify!` may change entries of the copied coefficient arrays.
# c_exp and c_imp default to the row sums of the (modified) matrices.
function ars343_copy(modify! = Returns(nothing))
    tab = IMEXTableau(ARS343())
    coeffs = map(dense, (; tab.a_exp, tab.b_exp, tab.a_imp, tab.b_imp))
    modify!(coeffs)
    return IMEXTableau(;
        a_exp = SMatrix{4, 4}(coeffs.a_exp),
        b_exp = SVector{4}(coeffs.b_exp),
        a_imp = SMatrix{4, 4}(coeffs.a_imp),
        b_imp = SVector{4}(coeffs.b_imp),
    )
end

@testset "First-gate ARS343" begin

    @testset "Coefficients" begin
        tab = IMEXTableau(ARS343())
        a_exp = dense(tab.a_exp)
        b_exp = dense(tab.b_exp)
        a_imp = dense(tab.a_imp)
        b_imp = dense(tab.b_imp)

        matches(x, expected) =
            isapprox(x, expected; atol = coeff_atol) && sign(x) == sign(expected)

        @test matches(a_exp[2, 1], γ)
        @test matches(a_exp[4, 1], a41)
        @test matches(a_exp[4, 2], α)
        @test matches(a_exp[4, 3], α)

        @test all(matches.(b_exp, [0, b1, b2, γ]))

        @test all(matches.(a_imp[1, :], [0, 0, 0, 0]))
        @test all(matches.(a_imp[2, :], [0, γ, 0, 0]))
        @test all(matches.(a_imp[3, :], [0, (1 - γ) / 2, γ, 0]))
        @test all(matches.(a_imp[4, :], [0, b1, b2, γ]))

        # Frozen assumption: b_imp is the last row of a_imp
        @test b_imp == a_imp[4, :]
        @test all(matches.(b_imp, [0, b1, b2, γ]))

        @test b1 + b2 + γ ≈ 1 atol = coeff_atol
        @test b_exp[2] + b_exp[3] + b_exp[4] ≈ 1 atol = coeff_atol
    end

    # γ is the middle root of 6γ³ - 18γ² + 9γ - 1 = 0 (roots ≈ 0.159, 0.436,
    # 2.405), checked independently of the 16-digit literal it is compared to
    # in src.
    @testset "γ is the middle root of the ARS343 cubic" begin
        p(x) = ((6x - 18) * x + 9) * x - 1
        function bisect(lo, hi)
            for _ in 1:200
                mid = (lo + hi) / 2
                (mid == lo || mid == hi) && break
                sign(p(mid)) == sign(p(lo)) ? (lo = mid) : (hi = mid)
            end
            return (lo + hi) / 2
        end
        # Sign changes on three brackets give all three real roots.
        brackets = ((0.0, 0.3), (0.3, 1.0), (1.0, 3.0))
        @test all(((lo, hi),) -> sign(p(lo)) != sign(p(hi)), brackets)
        r1, r2, r3 = map(b -> bisect(b...), brackets)
        # k·eps rationale: Horner takes 5 rounded operations with intermediates
        # below 16 in magnitude (6γ - 18 ≈ -15.4), so each error is at most
        # eps(8) / 2 = 2 eps(4) / 2 before later factors |γ| < 1 damp it; the
        # 16-digit literal adds <= 5e-17 |p'(γ)| ≈ 1.7e-16 < eps(4). The sum is
        # below 6 eps(4); 16 eps(4) ≈ 1.4e-14 covers it with margin.
        tol_cubic = 16 * eps(4.0)
        @test abs(p(γ)) <= tol_cubic
        # |p'(γ)| ≈ 3.3 > 1, so the root error is below the residual bound.
        @test abs(γ - r2) <= tol_cubic
        @test r1 < γ < r3
        @test abs(γ - r1) > 0.2 && abs(γ - r3) > 1.9
    end

    # Float32 state with the Float64 tableau, and Float64 throughout; the
    # Float64 expected value is converted to FT before comparison.
    @testset "Manufactured linear implicit problem ($FT)" for FT in (Float32, Float64)
        tol = provisional_tol(u1_expected, FT)
        alg(n) = IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = n))
        u1 = manufactured_endpoint(alg(2); FT)
        @test u1 isa FT
        @test abs(u1 - FT(u1_expected)) <= tol

        # The problem is linear, so a single Newton iteration is exact.
        u1_one_iter = manufactured_endpoint(alg(1); FT)
        @test abs(u1_one_iter - FT(u1_expected)) <= tol
        @test abs(u1_one_iter - u1) <= tol
    end

    # Negative controls: one flipped sign in a harness copy of the tableau shifts the endpoint by a predicted amount.
    @testset "Negative controls" begin
        newtons_method = NewtonsMethod(; max_iters = 2)
        endpoint(tab; kwargs...) =
            manufactured_endpoint(IMEXAlgorithm(tab, newtons_method); kwargs...)
        flip_a41!(c) = (c.a_exp[4, 1] *= -1)

        # The unmodified harness copy reproduces the golden endpoint.
        @test abs(endpoint(ars343_copy()) - u1_expected) <= provisional_tol(u1_expected)

        controls = (
            ("b_imp[3] flipped", c -> (c.b_imp[3] *= -1), -2b2 / γ),
            ("b_exp[3] flipped", c -> (c.b_exp[3] *= -1), -2b2),
            ("a_exp[4, 1] flipped", flip_a41!, -2a41),
        )
        @testset "$name" for (name, flip!, shift) in controls
            u1_faulty = endpoint(ars343_copy(flip!))
            # Endpoint moves by the predicted shift ...
            @test abs((u1_faulty - u1_expected) - shift) <=
                  provisional_tol(u1_expected + shift)
            # ... and therefore fails the golden comparison.
            @test abs(u1_faulty - u1_expected) > provisional_tol(u1_expected)
        end

        # Stage level: the a41 flip moves the explicit stage time
        # c4 = a41 + 2α = 1 to 1 - 2a41 and the stage value U4 = 2H by -4a41,
        # so the golden-stage comparison fails as well.
        @testset "a_exp[4, 1] flipped: stage U4" begin
            tol4 = provisional_tol(2H)
            calls_golden, T_exp_golden! = stage_recorder(Returns(1.0))
            calls_fault, T_exp_fault! = stage_recorder(Returns(1.0))
            endpoint(ars343_copy(); T_exp! = T_exp_golden!)
            endpoint(ars343_copy(flip_a41!); T_exp! = T_exp_fault!)
            U4_golden = stage_values(calls_golden, 1)
            U4_fault = stage_values(calls_fault, 1 - 2a41)
            @test !isempty(U4_golden) && all(u -> abs(u - 2H) <= tol4, U4_golden)
            @test !isempty(U4_fault) &&
                  all(u -> abs((u - 2H) - (-4a41)) <= tol4, U4_fault)
            @test !isempty(U4_fault) &&
                  all(u -> !isapprox(u, 2H; atol = tol4), U4_fault)
        end
    end

    # Time-dependent explicit problem: f_exp(u, t) = t, f_imp(u, t) = u / (2γ), u0 = 0,
    # one step with dt = 1. Unlike f_exp ≡ 1, the endpoint depends on a31 and
    # a32 individually, not only on their sum c3.
    @testset "C2 time-dependent explicit problem" begin
        # Independent expression for a32 (hand-derived, not taken from src)
        p = α * (-15 / 4 + 15γ - (21 / 4) * γ^2) + 4 - (25 / 2) * γ + (9 / 2) * γ^2
        c3 = (1 + γ) / 2
        J = b1 * γ + b2 * c3 + γ  # explicit final contribution
        u1_c2 = J + α * (γ + c3) + 2b2 * p
        U3_c2 = 2γ * p
        U4_c2 = 2 * (α * (γ + c3) + b2 * p)
        swap_shift = 2b2 * (c3 - 2p)
        tol = provisional_tol(u1_c2)

        newtons_method = NewtonsMethod(; max_iters = 2)
        endpoint(tab; kwargs...) =
            manufactured_endpoint(IMEXAlgorithm(tab, newtons_method); kwargs...)
        swap!(c) = (c.a_exp[3, 1], c.a_exp[3, 2]) = (c.a_exp[3, 2], c.a_exp[3, 1])

        @testset "a31, a32 against independent expression" begin
            a_exp = dense(IMEXTableau(ARS343()).a_exp)
            @test abs(a_exp[3, 2] - p) <= coeff_atol
            @test abs(a_exp[3, 1] - (c3 - p)) <= coeff_atol
        end

        # Float32 state with the Float64 tableau, and Float64 throughout; the
        # Float64 expected values are converted to FT before comparison.
        @testset "Endpoint ($FT)" for FT in (Float32, Float64)
            # Stage values observed through the explicit tendency
            calls, T_exp_rec! = stage_recorder(t -> t)
            u1 = endpoint(ARS343(); T_exp! = T_exp_rec!, FT)
            @test u1 isa FT
            @test abs(u1 - FT(u1_c2)) <= provisional_tol(u1_c2, FT)
            if FT == Float64
                # Display-precision guard against the 8-decimal hand value; the
                # bound is the printed precision (rounding error <= 5e-9).
                @test abs(u1 - 0.62679059) <= 1e-8
            end

            # Recorded stage values are FT values widened to Float64.
            U3_obs = stage_values(calls, c3; atol = stage_atol(FT))
            U4_obs = stage_values(calls, 1; atol = stage_atol(FT))
            @test !isempty(U3_obs) &&
                  all(u -> abs(u - FT(U3_c2)) <= provisional_tol(U3_c2, FT), U3_obs)
            @test !isempty(U4_obs) &&
                  all(u -> abs(u - FT(U4_c2)) <= provisional_tol(U4_c2, FT), U4_obs)
        end

        @testset "Swap control: a_exp[3, 1] <-> a_exp[3, 2]" begin
            calls_golden, T_exp_golden! = stage_recorder(t -> t)
            calls_swap, T_exp_swap! = stage_recorder(t -> t)
            u1_golden = endpoint(ars343_copy(); T_exp! = T_exp_golden!)
            u1_swap = endpoint(ars343_copy(swap!); T_exp! = T_exp_swap!)
            @test abs(u1_golden - u1_c2) <= tol
            @test abs((u1_swap - u1_golden) - swap_shift) <= tol
            @test !isapprox(u1_swap, u1_c2; atol = tol)

            # Stage level: the swap keeps c3 = a31 + a32, so U3 is still
            # recorded at c3, moves by 2γ(c3 - 2p) and fails the golden U3.
            tol3 = provisional_tol(U3_c2)
            U3_golden = stage_values(calls_golden, c3)
            U3_swap = stage_values(calls_swap, c3)
            @test !isempty(U3_golden) && all(u -> abs(u - U3_c2) <= tol3, U3_golden)
            @test !isempty(U3_swap) &&
                  all(u -> abs((u - U3_c2) - 2γ * (c3 - 2p)) <= tol3, U3_swap)
            @test !isempty(U3_swap) &&
                  all(u -> !isapprox(u, U3_c2; atol = tol3), U3_swap)

            # The same swap leaves the f_exp ≡ 1 endpoint unchanged.
            u1_const = endpoint(ars343_copy())
            u1_const_swap = endpoint(ars343_copy(swap!))
            @test abs(u1_const_swap - u1_const) < provisional_tol(u1_expected)
        end
    end
end
