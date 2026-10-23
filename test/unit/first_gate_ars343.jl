#=
First-gate checks for ARS343 coefficients and one-step assembly (B001 T3).

Expected values are the frozen T0 expectations (B001 `t0-expectations.md`,
Section 1). They are typed in here or derived from the frozen constants, never
taken from the solver.
=#
using ClimaTimeSteppers, Test, LinearAlgebra
using StaticArrays
import ClimaTimeSteppers as CTS

# Helper to extract dense Float64 arrays from SparseCoeffs (as in tableaus.jl)
dense(sc::CTS.SparseCoeffs) = Array(Float64.(sc.coeffs))

# Frozen constants (t0-expectations.md Section 1; src/solvers/imex_tableaus.jl:310–338)
const γ = 0.4358665215084590
const b1 = 1.208496649176010
const b2 = -0.644363170684469
const a41 = -0.1058582960718796
const α = 0.5529291480359398  # a42 = a43
const coeff_atol = 2e-15

# Manufactured linear problem: f_exp(u, t) = 1, f_imp(u, t) = u / (2γ), u0 = 0,
# one step with dt = 1. Stage by stage, U4 = 2H and u1 = 2H with
# H = (a41 + 2α) + b1 + b2 / γ.
const H = (a41 + 2α) + b1 + b2 / γ
const u1_expected = 2H
provisional_tol(x) = 64 * eps(Float64) * abs(x)  # provisional, uncalibrated (D006)

function manufactured_endpoint(alg; T_exp! = (du, u, p, t) -> (du .= 1))
    # W = dtγ * J - I with J = 1 / (2γ)
    Wfact!(W, u, p, dtγ, t) = (W[1, 1] = dtγ / (2γ) - 1)
    T_imp = CTS.ODEFunction(
        (du, u, p, t) -> (du .= u ./ (2γ));
        jac_prototype = zeros(1, 1),
        Wfact = Wfact!,
    )
    f = ClimaODEFunction(; T_exp!, T_imp! = T_imp)
    prob = CTS.ODEProblem(f, [0.0], (0.0, 1.0), nothing)
    sol = CTS.solve(prob, alg; dt = 1.0, saveat = (1.0,))
    return sol.u[end][1]
end

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

    # t0-expectations.md Section 1, coefficients
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

    # t0-expectations.md Section 1, manufactured linear implicit problem
    @testset "Manufactured linear implicit problem" begin
        tol = provisional_tol(u1_expected)
        u1 = manufactured_endpoint(IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2)))
        @test abs(u1 - u1_expected) <= tol

        # The problem is linear, so a single Newton iteration is exact.
        u1_one_iter =
            manufactured_endpoint(IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 1)))
        @test abs(u1_one_iter - u1_expected) <= tol
        @test abs(u1_one_iter - u1) <= tol
    end

    # t0-expectations.md Section 1, negative controls: one flipped sign in a
    # harness copy of the tableau shifts the endpoint by a predicted amount.
    @testset "Negative controls" begin
        newtons_method = NewtonsMethod(; max_iters = 2)
        endpoint(tab) = manufactured_endpoint(IMEXAlgorithm(tab, newtons_method))

        # The unmodified harness copy reproduces the golden endpoint.
        @test abs(endpoint(ars343_copy()) - u1_expected) <= provisional_tol(u1_expected)

        controls = (
            ("b_imp[3] flipped", c -> (c.b_imp[3] *= -1), -2b2 / γ),
            ("b_exp[3] flipped", c -> (c.b_exp[3] *= -1), -2b2),
            ("a_exp[4, 1] flipped", c -> (c.a_exp[4, 1] *= -1), -2a41),
        )
        @testset "$name" for (name, flip!, shift) in controls
            u1_faulty = endpoint(ars343_copy(flip!))
            # Endpoint moves by the predicted shift ...
            @test abs((u1_faulty - u1_expected) - shift) <=
                  provisional_tol(u1_expected + shift)
            # ... and therefore fails the golden comparison.
            @test abs(u1_faulty - u1_expected) > provisional_tol(u1_expected)
        end
    end

    # B001 C2 case (test 3): f_exp(u, t) = t, f_imp(u, t) = u / (2γ), u0 = 0,
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
        tol = provisional_tol(u1_c2)  # provisional, uncalibrated (D006)

        newtons_method = NewtonsMethod(; max_iters = 2)
        T_exp_t! = (du, u, p, t) -> (du .= t)
        endpoint(tab; kwargs...) =
            manufactured_endpoint(IMEXAlgorithm(tab, newtons_method); kwargs...)
        swap!(c) = (c.a_exp[3, 1], c.a_exp[3, 2]) = (c.a_exp[3, 2], c.a_exp[3, 1])

        @testset "a31, a32 against independent expression" begin
            a_exp = dense(IMEXTableau(ARS343()).a_exp)
            @test abs(a_exp[3, 2] - p) <= coeff_atol
            @test abs(a_exp[3, 1] - (c3 - p)) <= coeff_atol
        end

        @testset "Endpoint" begin
            # Stage values observed through the explicit tendency, which is
            # called with each stage value and stage time.
            calls = Tuple{Float64, Float64}[]
            T_exp_rec! = (du, u, p, t) -> (push!(calls, (t, u[1])); du .= t)
            u1 = endpoint(ARS343(); T_exp! = T_exp_rec!)
            @test abs(u1 - u1_c2) <= tol  # provisional (D006)
            # Display-precision guard against the 8-digit hand value
            @test abs(u1 - 0.62679059) <= 1e-8

            # Stage times come from row sums, so match them to c3 and 1 within
            # coefficient tolerance rather than exactly.
            U3_obs = [u for (t, u) in calls if abs(t - c3) <= coeff_atol]
            U4_obs = [u for (t, u) in calls if abs(t - 1) <= coeff_atol]
            @test !isempty(U3_obs) && all(u -> abs(u - U3_c2) <= provisional_tol(U3_c2), U3_obs)
            @test !isempty(U4_obs) && all(u -> abs(u - U4_c2) <= provisional_tol(U4_c2), U4_obs)
        end

        @testset "Swap control: a_exp[3, 1] <-> a_exp[3, 2]" begin
            u1_golden = endpoint(ars343_copy(); T_exp! = T_exp_t!)
            u1_swap = endpoint(ars343_copy(swap!); T_exp! = T_exp_t!)
            @test abs(u1_golden - u1_c2) <= tol
            @test abs((u1_swap - u1_golden) - swap_shift) <= tol
            @test !isapprox(u1_swap, u1_c2; atol = tol)

            # The same swap leaves the f_exp ≡ 1 endpoint unchanged.
            u1_const = endpoint(ars343_copy())
            u1_const_swap = endpoint(ars343_copy(swap!))
            @test abs(u1_const_swap - u1_const) < provisional_tol(u1_expected)
        end
    end
end
