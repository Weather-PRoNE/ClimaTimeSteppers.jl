#=
Labelled solver events in the IMEX ARK stage loop. An observer passed as
`init(...; solver_observer)` receives `StepStart`, `StageBaseline`,
`EffectiveStorage`, `AssemblyInputs` and `Operation` events with borrowed
payloads; without an observer the stepping path is unchanged.

Manufactured problem: one ARS343 step of `du/dt = 1 + u / (2γ)`, `u0 = 0`,
`dt = 1`, explicit part `1`, implicit part `u / (2γ)`. The stage values follow
from hand algebra on the tableau constants:
  stage 2: temp = γ, U = 2γ, T_imp[2] = 1
  stage 3: temp = 1, U = 2,  T_imp[3] = 1/γ
  stage 4: temp = H, U = 2H, T_imp[4] = H/γ,  H = 1 + b1 + b2/γ
  endpoint u1 = 2H.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: ODEProblem, ODEFunction

# ARS343 constants (src/solvers/imex_tableaus.jl)
const γ = 0.4358665215084590
const b1 = 1.208496649176010
const b2 = -0.644363170684469
const H = 1 + b1 + b2 / γ
# Provisional tolerance for values that pass through Newton.
close_to(x, e) = abs(x - e) <= 64 * eps(Float64) * abs(e)
# Tableau coefficients are computed expressions (b2 differs from its decimal
# literal in the last bit): within 2e-15 absolute, sign exact.
coeffs_match(c, e) =
    length(c) == length(e) &&
    all(abs(x - y) <= 2e-15 && sign(x) == sign(y) for (x, y) in zip(c, e))

function make_prob(; u0 = [0.0], kwargs...)
    FT = eltype(u0)
    g = FT(γ)
    ODEProblem(
        ClimaODEFunction(;
            T_exp! = (du, u, p, t) -> (du .= 1),
            T_imp! = ODEFunction(
                (du, u, p, t) -> (du .= u ./ (2g));
                jac_prototype = zeros(FT, 1, 1),
                # Newton expects W = dtγ J - I with J = 1 / (2γ).
                Wfact = (W, u, p, dtγ, t) -> (W .= dtγ / (2g) - 1),
            ),
            kwargs...,
        ),
        u0,
        (zero(FT), FT(10)),
        nothing,
    )
end
make_alg(; max_iters = 1, kwargs...) =
    IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters); kwargs...)

function one_step(; prob = make_prob(), alg = make_alg(), kwargs...)
    integrator = CTS.init(prob, alg; dt = one(eltype(prob.u0)), kwargs...)
    CTS.step!(integrator)
    return integrator
end

# Test observer: copies every payload so the record survives buffer reuse.
struct Recorder
    log::Vector{Any}
end
Recorder() = Recorder(Any[])
function (r::Recorder)(e)
    c = e.ctx
    data = Dict{Symbol, Any}(
        f => deepcopy(getfield(e, f)) for f in fieldnames(typeof(e)) if f !== :ctx
    )
    push!(r.log, (; kind = nameof(typeof(e)), c.step, c.t, c.dt, c.stage, c.dtγ, data))
    return nothing
end
tag(r) = r.kind === :Operation ? (r.data[:site], r.data[:phase]) : r.kind
events(log, kind) = filter(r -> r.kind === kind, log)

# Test observer: snapshots the mutable payload leaves, calls `inner`, and
# records the event kind if any leaf changed bitwise.
struct CheckedObserver{O}
    inner::O
    violations::Vector{Symbol}
end
CheckedObserver(inner) = CheckedObserver(inner, Symbol[])
function mutable_leaves(e)
    leaves = AbstractArray[]
    for f in fieldnames(typeof(e))
        x = getfield(e, f)
        x isa AbstractArray && push!(leaves, x)
        x isa CTS.SparseContainer && append!(leaves, parent(x))
    end
    return leaves
end
same_bits(a, b) = size(a) == size(b) && all(x === y for (x, y) in zip(a, b))
function (c::CheckedObserver)(e)
    leaves = mutable_leaves(e)
    snapshots = map(copy, leaves)
    c.inner(e)
    all(map(same_bits, leaves, snapshots)) || push!(c.violations, nameof(typeof(e)))
    return nothing
end

const ROLES = Dict(
    :stage_lim => :stage_construction,
    :stage_dss_pre => :stage_construction,
    :stage_constrain_pre => :stage_construction,
    :stage_cache_no_implicit => :stage_construction,
    :init_imp => :stage_processing,
    :stage_dss_post_init => :stage_processing,
    :stage_constrain_post_init => :stage_processing,
    :cache_imp_pre_newton => :subordinate_stage_write,
    :cache_imp_newton => :subordinate_stage_write,
    :cache_imp_post_newton => :subordinate_stage_write,
    :post_imp_correction => :subordinate_stage_write,
    :stage_dss_post => :stage_processing,
    :stage_constrain_post => :stage_processing,
    :stage_cache_post => :stage_processing,
    :stage_cache_fallback => :subordinate_stage_write,
    :final_lim => :final_limiter,
    :final_dss => :final_map,
    :final_constrain => :final_map,
    :final_cache => :final_map,
)

# Event sequence of one ARS343 step, derived from src/solvers/imex_ark.jl.
# Stage 1 has a zero implicit diagonal and is skipped by every event. The
# defaults are `initialize_imp! = Returns(nothing)` (wired),
# `update_cache = UpdateEvery(EndOfStage)` and
# `update_constrain_state = UpdateEvery(EndOfStep)`; ARS343 is not FSAL.
function expected_trace(;
    newton_calls,
    init_imp = true,
    cache_post = true,
    constrain_stages = false,
    lim = false,
    post_imp = false,
)
    seq = Any[:StepStart]
    op(site) = push!(seq, (site, :entry), (site, :exit))
    for _ in 2:4
        lim && op(:stage_lim)
        op(:stage_dss_pre)
        constrain_stages && op(:stage_constrain_pre)
        push!(seq, :StageBaseline)
        if init_imp
            op(:init_imp)
            op(:stage_dss_post_init)
            constrain_stages && op(:stage_constrain_post_init)
        end
        op(:cache_imp_pre_newton)
        foreach(_ -> op(:cache_imp_newton), 1:newton_calls)
        if post_imp
            op(:cache_imp_post_newton)
            op(:post_imp_correction)
        end
        op(:stage_dss_post)
        constrain_stages && op(:stage_constrain_post)
        op(cache_post ? :stage_cache_post : :stage_cache_fallback)
        push!(seq, :EffectiveStorage)
    end
    push!(seq, :AssemblyInputs)
    lim && op(:final_lim)
    op(:final_dss)
    op(:final_constrain)
    op(:final_cache)
    return seq
end

newton_calls_per_stage(log) = [
    count(r -> tag(r) == (:cache_imp_newton, :entry) && r.stage == i, log) for i in 2:4
]

@testset "trace of one ARS343 step" begin
    rec = Recorder()
    one_step(; solver_observer = rec)
    log = rec.log
    @test length(events(log, :StepStart)) == 1
    @test [r.stage for r in events(log, :StageBaseline)] == [2, 3, 4]
    @test [r.data[:i] for r in events(log, :EffectiveStorage)] == [2, 3, 4]
    @test length(events(log, :AssemblyInputs)) == 1
    @test newton_calls_per_stage(log) == [0, 0, 0]
    @test map(tag, log) == expected_trace(; newton_calls = 0)
    for r in events(log, :Operation)
        @test r.data[:role] === ROLES[r.data[:site]]
    end
    @test all(r -> r.step == 1 && r.t == 0 && r.dt == 1, log)
    @test all(r -> r.stage == 0 && r.dtγ === nothing, filter(r -> r.stage == 0, log))
    @test all(r -> r.dtγ == γ, filter(r -> r.stage != 0, log))
    @test all(r -> r.stage != 0, events(log, :StageBaseline))
end

@testset "payload values against hand algebra" begin
    rec = Recorder()
    integrator = one_step(; solver_observer = rec)
    log = rec.log

    baselines = events(log, :StageBaseline)
    @test all(map(close_to, [r.data[:temp][1] for r in baselines], [γ, 1, H]))

    effective = events(log, :EffectiveStorage)
    @test all(map(close_to, [r.data[:U][1] for r in effective], [2γ, 2, 2H]))
    @test all(map(close_to, [r.data[:T_imp_i][1] for r in effective], [1, 1 / γ, H / γ]))
    @test all(map(close_to, [r.data[:temp][1] for r in effective], [γ, 1, H]))
    @test all(r -> r.data[:dtγ] == γ, effective)

    assembly = only(events(log, :AssemblyInputs))
    d = assembly.data
    T_exp, T_imp, b_exp, b_imp, dt = d[:T_exp], d[:T_imp], d[:b_exp], d[:b_imp], d[:dt]
    @test dt == 1
    @test all(j -> T_exp[j][1] == 1, 1:4)
    @test coeffs_match(b_exp.coeffs, [0, b1, b2, γ])
    @test coeffs_match(b_imp.coeffs, [0, b1, b2, γ])
    @test all(map(close_to, [T_imp[j][1] for j in 2:4], [1, 1 / γ, H / γ]))
    @test close_to(sum(j -> b_exp.coeffs[j] * T_exp[j][1], 1:4), 1.0)
    @test close_to(sum(j -> b_imp.coeffs[j] * T_imp[j][1], 2:4), 2H - 1)

    @test close_to(integrator.u[1], 2H)
    @test isapprox(2H, 1.4602937636; atol = 1e-10)
end

@testset "payloads are borrowed" begin
    integrator_ref = Ref{Any}()
    identities = Bool[]
    rec = Recorder()
    observer = e -> begin
        if e isa StageBaseline || e isa EffectiveStorage
            cache = integrator_ref[].cache
            push!(identities, e.temp === cache.temp && e.U === cache.U)
        end
        rec(e)
    end
    prob = make_prob()
    integrator = CTS.init(prob, make_alg(); dt = 1.0, solver_observer = observer)
    integrator_ref[] = integrator
    CTS.step!(integrator)
    @test length(identities) == 6 && all(identities)
    # The copy taken at stage 2 keeps γ; the live buffer now holds stage 4.
    stage2 = first(events(rec.log, :StageBaseline))
    @test stage2.data[:temp][1] == γ
    @test close_to(integrator.cache.temp[1], H)

    # The checked observer passes a read-only observer ...
    checked = CheckedObserver(Recorder())
    integrator = one_step(; solver_observer = checked)
    @test isempty(checked.violations)
    @test close_to(integrator.u[1], 2H)

    # ... and detects one that writes into a borrowed payload.
    vandal = e -> (e isa StageBaseline && (e.temp[1] = 0); nothing)
    checked = CheckedObserver(vandal)
    one_step(; solver_observer = checked)
    @test checked.violations == [:StageBaseline, :StageBaseline, :StageBaseline]
end

newton_arrays(nmc::Nothing) = AbstractArray[]
newton_arrays(nmc) = AbstractArray[getfield(nmc, k) for k in propertynames(nmc) if getfield(nmc, k) isa AbstractArray]
cache_arrays(c) = [
    c.U, c.temp, parent(c.T_exp)..., parent(c.T_lim)..., parent(c.T_imp)...,
    newton_arrays(c.newtons_method_cache)...,
]
function warm_step_allocations(; kwargs...)
    integrator = CTS.init(make_prob(), make_alg(); dt = 1.0, kwargs...)
    CTS.step!(integrator)
    CTS.step!(integrator)
    return @allocated CTS.step!(integrator)
end

@testset "no observer, nothing, no-op and recording agree bitwise" begin
    runs = (
        one_step(),
        one_step(; solver_observer = nothing),
        one_step(; solver_observer = Returns(nothing)),
        one_step(; solver_observer = Recorder()),
    )
    reference = first(runs)
    for integrator in runs
        @test same_bits(integrator.u, reference.u)
        @test all(map(same_bits, cache_arrays(integrator.cache), cache_arrays(reference.cache)))
    end
    @test length(newton_arrays(reference.cache.newtons_method_cache)) == 3  # Δx, f, j
    perturbed = one_step()
    perturbed.cache.newtons_method_cache.Δx[1] += 1
    @test !all(map(same_bits, cache_arrays(perturbed.cache), cache_arrays(reference.cache)))
    @test reference.cache.solver_observer === nothing
    @test runs[2].cache.solver_observer === nothing

    a = warm_step_allocations()
    b = warm_step_allocations(; solver_observer = nothing)
    c = warm_step_allocations(; solver_observer = Returns(nothing))
    d = warm_step_allocations(; solver_observer = Recorder())
    @test a == b
    @info "warm step! allocations (bytes)" no_keyword = a nothing_observer = b noop_observer =
        c recording_observer = d noop_delta = c - a
    @test c == a
end

@testset "branch coverage" begin
    base = one_step()

    # T_post_imp! wired with a zero correction.
    rec = Recorder()
    integrator = one_step(;
        prob = make_prob(; T_post_imp! = (du, u, p, t) -> (du .= 0)),
        solver_observer = rec,
    )
    @test map(tag, rec.log) == expected_trace(; newton_calls = 0, post_imp = true)
    @test integrator.u == base.u

    # initialize_imp! wired explicitly (same as the default) and absent.
    rec = Recorder()
    one_step(;
        prob = make_prob(; initialize_imp! = (U, p, dtγ) -> nothing),
        solver_observer = rec,
    )
    @test map(tag, rec.log) == expected_trace(; newton_calls = 0)
    rec = Recorder()
    integrator = one_step(; prob = make_prob(; initialize_imp! = nothing), solver_observer = rec)
    @test map(tag, rec.log) == expected_trace(; newton_calls = 0, init_imp = false)
    @test integrator.u == base.u

    # update_cache: EndOfStage (the default) vs EndOfStep (fallback to cache_imp!).
    rec = Recorder()
    one_step(;
        prob = make_prob(; update_cache = UpdateEvery(EndOfStage)),
        solver_observer = rec,
    )
    @test map(tag, rec.log) == expected_trace(; newton_calls = 0)
    rec = Recorder()
    one_step(; prob = make_prob(; update_cache = UpdateEvery(EndOfStep)), solver_observer = rec)
    @test map(tag, rec.log) == expected_trace(; newton_calls = 0, cache_post = false)

    # update_constrain_state = UpdateEvery(WithDSS) fires at every stage site.
    rec = Recorder()
    one_step(;
        prob = make_prob(; update_constrain_state = UpdateEvery(WithDSS)),
        solver_observer = rec,
    )
    @test map(tag, rec.log) == expected_trace(; newton_calls = 0, constrain_stages = true)

    # A limited tendency enables the stage and final limiter sites.
    rec = Recorder()
    one_step(; prob = make_prob(; T_lim! = (du, u, p, t) -> (du .= 0)), solver_observer = rec)
    @test map(tag, rec.log) == expected_trace(; newton_calls = 0, lim = true)
    lim_ops = filter(r -> r.kind === :Operation && r.data[:site] === :final_lim, rec.log)
    @test all(r -> r.data[:extra] isa AbstractArray, lim_ops)

    # Two Newton iterations call the pre-residual cache_imp! hook.
    rec = Recorder()
    one_step(; alg = make_alg(; max_iters = 2), solver_observer = rec)
    calls = newton_calls_per_stage(rec.log)
    @info "cache_imp_newton calls per stage (stages 2:4), max_iters = 2" calls
    @test calls == [1, 1, 1]
    @test map(tag, rec.log) == expected_trace(; newton_calls = 1)

    # Cast tableau on a Float32 state: payload coefficients follow cache.tableau.
    rec = Recorder()
    alg = make_alg(; cast_tableau_to_state_eltype = true)
    integrator = one_step(; prob = make_prob(; u0 = Float32[0]), alg, solver_observer = rec)
    tableau = only(events(rec.log, :StepStart)).data[:tableau]
    assembly = only(events(rec.log, :AssemblyInputs))
    @test eltype(tableau.b_exp.coeffs) == Float32
    @test eltype(tableau.b_exp.coeffs) == eltype(integrator.cache.tableau.b_exp.coeffs)
    @test eltype(alg.tableau.b_exp.coeffs) == Float64
    @test eltype(assembly.data[:b_imp].coeffs) == Float32
end

@testset "unsupported algorithms reject an observer" begin
    prob = make_prob()
    ssp = IMEXAlgorithm(SSP333(), NewtonsMethod(; max_iters = 1))
    err = try
        CTS.init(prob, ssp; dt = 1.0, solver_observer = Recorder())
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("IMEX ARK only", sprint(showerror, err))
    @test_throws ArgumentError CTS.init(
        prob, ExplicitAlgorithm(SSP33ShuOsher()); dt = 1.0, solver_observer = Recorder(),
    )

    for kwargs in ((;), (; solver_observer = nothing))
        integrator = CTS.init(prob, ssp; dt = 1.0, kwargs...)
        CTS.step!(integrator)
        @test isfinite(integrator.u[1]) && integrator.t == 1
    end
end

@testset "explicit-only ARK (RK4) is supported and traced" begin
    prob = ODEProblem(ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= 1)), [0.0], (0.0, 10.0), nothing)
    rec = Recorder()
    integrator = CTS.init(prob, ExplicitAlgorithm(RK4()); dt = 1.0, solver_observer = rec)
    CTS.step!(integrator)
    @test close_to(integrator.u[1], 1.0)
    log = rec.log
    @test length(events(log, :StepStart)) == 1
    @test length(events(log, :AssemblyInputs)) == 1
    @test isempty(events(log, :StageBaseline))
    @test isempty(events(log, :EffectiveStorage))
    sites = [r.data[:site] for r in log if r.kind === :Operation && r.data[:phase] === :entry]
    # no implicit stage: per stage i ≠ 1 the pre-DSS and, with the default
    # UpdateEvery(EndOfStage) cache policy, the no-implicit cache site; then the final maps
    @test count(==(:stage_dss_pre), sites) == 3
    @test count(==(:stage_cache_no_implicit), sites) == 3
    @test count(==(:final_dss), sites) == 1
    @test count(==(:final_constrain), sites) == 1
    @test count(==(:final_cache), sites) == 1
    @test isempty(filter(s -> s in (:init_imp, :cache_imp_pre_newton, :cache_imp_newton, :cache_imp_post_newton, :post_imp_correction, :stage_dss_post, :stage_cache_fallback, :stage_lim, :final_lim), sites))
end
