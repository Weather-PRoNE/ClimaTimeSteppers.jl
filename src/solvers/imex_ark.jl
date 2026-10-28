import NVTX

has_jac(T_imp!) =
    hasfield(typeof(T_imp!), :Wfact) &&
    hasfield(typeof(T_imp!), :jac_prototype) &&
    !isnothing(T_imp!.Wfact) &&
    !isnothing(T_imp!.jac_prototype)

"""
    EventContext{O, T, DT, DTG, TE, TI}

Position of a solver event in the IMEX ARK step, passed as the `ctx` field of
every event sent to the `solver_observer` given to [`init`](@ref).

# Fields
- `observer`: the observer; called as `observer(event)`.
- `step::Int`: the integrator step counter.
- `t`: the time at the start of the step.
- `dt`: the actual timestep of this step (`integrator.dt`).
- `stage::Int`: the stage index, or `0` for step-level events.
- `dtγ`: the stage value `dt * a_imp[i, i]`, or `nothing` at step level.
- `t_exp`, `t_imp`: the explicit and implicit stage times, or `nothing`.

# Observer contract
Observers never mutate state or caches. Event payloads are borrowed: arrays
are the solver's live buffers, valid only until the observer returns, and
consumers that keep values must copy them. `temp` and `U` are scratch reused
across stages and by the final assembly.

Solver events are implemented for IMEX ARK (`IMEXAlgorithm{Unconstrained}`)
only. With `solver_observer = nothing` (the default) no context or event is
constructed.
"""
struct EventContext{O, T, DT, DTG, TE, TI}
    observer::O
    step::Int
    t::T
    dt::DT
    stage::Int
    dtγ::DTG
    t_exp::TE
    t_imp::TI
end

"""
    StepStart(ctx, u, tableau)

Emitted at the start of an IMEX ARK step, before the first stage; `u` is the
step-start state and `tableau` the cache tableau (after an optional eltype cast).
"""
struct StepStart{C, U, TAB}
    ctx::C
    u::U
    tableau::TAB
end

"""
    StageBaseline(ctx, temp, U)

Emitted right after the implicit-stage baseline `temp = U` is stored, only on
stages with an implicit solve. `temp` and `U` are reused scratch buffers.
"""
struct StageBaseline{C, A, B}
    ctx::C
    temp::A
    U::B
end

"""
    EffectiveStorage(ctx, i, T_imp_i, U, temp, dtγ)

Emitted right after the effective implicit tendency
`T_imp[i] = (U - temp) / dtγ` is stored for stage `i`.
"""
struct EffectiveStorage{C, A, B, D, G}
    ctx::C
    i::Int
    T_imp_i::A
    U::B
    temp::D
    dtγ::G
end

"""
    AssemblyInputs(ctx, u, dt, T_exp, T_lim, T_imp, b_exp, b_imp, tableau)

Emitted after the last stage and before the final assembly
`u ← u + dt Σ b_exp T_exp + dt Σ b_imp T_imp` (limited when `T_lim` is used).
"""
struct AssemblyInputs{C, U, DT, TE, TL, TI, BE, BI, TAB}
    ctx::C
    u::U
    dt::DT
    T_exp::TE
    T_lim::TL
    T_imp::TI
    b_exp::BE
    b_imp::BI
    tableau::TAB
end

"""
    Operation(ctx, site, role, phase, state, extra)

Emitted with `phase = :entry` before and `phase = :exit` after a solver call of
a user hook (`lim!`, `dss!`, `constrain_state!`, `cache!`, `cache_imp!`,
`initialize_imp!`, or the `T_post_imp!` correction including its application),
only when that call executes. `site` names the call site and `role` its part in
the step (`:stage_construction`, `:stage_processing`,
`:subordinate_stage_write`, `:final_limiter`, `:final_map`). `state` is the
array the hook acts on; `extra` is the limiter reference `u` for `lim!`, the
scratch `T_imp[i]` for the `T_post_imp!` correction, and `nothing` otherwise.

| `site`                       | `role`                     | hook               | `state` |
|:-----------------------------|:---------------------------|:-------------------|:--------|
| `:stage_lim`                 | `:stage_construction`      | `lim!`             | `U`     |
| `:stage_dss_pre`             | `:stage_construction`      | `dss!`             | `U`     |
| `:stage_constrain_pre`       | `:stage_construction`      | `constrain_state!` | `U`     |
| `:stage_cache_no_implicit`   | `:stage_construction`      | `cache!`           | `U`     |
| `:init_imp`                  | `:stage_processing`        | `initialize_imp!`  | `U`     |
| `:stage_dss_post_init`       | `:stage_processing`        | `dss!`             | `U`     |
| `:stage_constrain_post_init` | `:stage_processing`        | `constrain_state!` | `U`     |
| `:cache_imp_pre_newton`      | `:subordinate_stage_write` | `cache_imp!`       | `U`     |
| `:cache_imp_newton`          | `:subordinate_stage_write` | `cache_imp!`       | iterate |
| `:cache_imp_post_newton`     | `:subordinate_stage_write` | `cache_imp!`       | `U`     |
| `:post_imp_correction`       | `:subordinate_stage_write` | `T_post_imp!`      | `U`     |
| `:stage_dss_post`            | `:stage_processing`        | `dss!`             | `U`     |
| `:stage_constrain_post`      | `:stage_processing`        | `constrain_state!` | `U`     |
| `:stage_cache_post`          | `:stage_processing`        | `cache!`           | `U`     |
| `:stage_cache_fallback`      | `:subordinate_stage_write` | `cache_imp!`       | `U`     |
| `:final_lim`                 | `:final_limiter`           | `lim!`             | `temp`  |
| `:final_dss`                 | `:final_map`               | `dss!`             | `u`     |
| `:final_constrain`           | `:final_map`               | `constrain_state!` | `u`     |
| `:final_cache`               | `:final_map`               | `cache!`           | `u`     |

`U` is the stage state, `u` the step's state, `temp` the final-limiter scratch, and
"iterate" the Newton iterate passed to `cache_imp!` before each residual. `extra` is
`u` at `:stage_lim` and `:final_lim`, `T_imp[i]` at `:post_imp_correction` (whose
exit follows `U .+= dtγ .* T_imp[i]`), and `nothing` elsewhere. For
`:post_imp_correction`, `extra = T_imp[i]` still holds an earlier value at
`phase = :entry`; it holds the correction only at `phase = :exit`.
"""
struct Operation{C, S, X}
    ctx::C
    site::Symbol
    role::Symbol
    phase::Symbol
    state::S
    extra::X
end

@inline event_context(::Nothing, args...) = nothing
@inline event_context(observer, step, t, dt, stage, dtγ, t_exp, t_imp) =
    EventContext(observer, step, t, dt, stage, dtγ, t_exp, t_imp)

@inline emit(::Nothing, ::Type{E}, args...) where {E} = nothing
@inline emit(ctx::EventContext, ::Type{E}, args...) where {E} =
    (ctx.observer(E(ctx, args...)); nothing)

# Call `hook(args...)`, bracketed by `Operation` events when observed.
@inline op!(::Nothing, site, role, state, extra, hook, args...) = hook(args...)
@inline function op!(ctx::EventContext, site, role, state, extra, hook, args...)
    emit(ctx, Operation, site, role, :entry, state, extra)
    result = hook(args...)
    emit(ctx, Operation, site, role, :exit, state, extra)
    return result
end

supports_solver_events(::IMEXAlgorithm{Unconstrained}) = true

"""
    IMEXARKCache{SCU, SCE, SCI, T, Γ, NMC, TAB, O}

Pre-allocated workspace for an unconstrained IMEX ARK timestep.

# Fields
- `U`: stage state (sparse container of length `s`).
- `T_lim`: limited explicit tendency at each stage.
- `T_exp`: explicit tendency at each stage.
- `T_imp`: implicit tendency at each stage.
- `temp`: scratch array for the implicit solve.
- `γ`: common SDIRK diagonal coefficient (or `nothing` for non-SDIRK).
- `newtons_method_cache`: cache for [`NewtonsMethod`](@ref).
- `tableau`: the [`IMEXTableau`](@ref) with optional eltype cast.
- `solver_observer`: observer of solver events, or `nothing` (see
  [`EventContext`](@ref)).
"""
struct IMEXARKCache{SCU, SCE, SCI, T, Γ, NMC, TAB, O}
    U::SCU
    T_lim::SCE
    T_exp::SCE
    T_imp::SCI
    temp::T
    γ::Γ
    newtons_method_cache::NMC
    tableau::TAB
    solver_observer::O
end

function init_cache(
    prob,
    alg::IMEXAlgorithm{Unconstrained};
    solver_observer = nothing,
    kwargs...,
)
    (; u0, f) = prob
    (; T_imp!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp) = tableau
    s = length(b_exp)
    inds = ntuple(i -> i, s)
    inds_T_exp = filter(i -> !all(iszero, a_exp[:, i]) || !iszero(b_exp[i]), inds)
    inds_T_imp = filter(i -> !all(iszero, a_imp[:, i]) || !iszero(b_imp[i]), inds)
    U = zero(u0)
    T_lim = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_exp = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_imp = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_imp))), inds_T_imp)
    temp = zero(u0)
    γs = unique(filter(!iszero, diag(a_imp)))
    γ = length(γs) == 1 ? γs[1] : nothing # TODO: This could just be a constant.
    check_sdirk_compat(alg, γ)
    jac_prototype = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
    newtons_method_cache =
        !has_T_imp(f) || isnothing(newtons_method) ? nothing :
        allocate_cache(newtons_method, u0, jac_prototype)

    # Cast tableau coefficients to `eltype(u0)` if requested via
    # `cast_tableau_to_state_eltype`; otherwise preserve the tableau's eltype.
    opt_tb =
        alg.options.cast_tableau_to_state_eltype ? downcast_tableau(eltype(u0), tableau) :
        tableau

    return IMEXARKCache(
        U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache, opt_tb, solver_observer,
    )
end

# generic fallback
function step_u!(integrator, cache::IMEXARKCache)
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; cache!, T_imp!, lim!, dss!, constrain_state!) = f
    (; b_exp, a_imp, b_imp) = cache.tableau
    (; T_lim, T_exp, T_imp, temp) = cache
    v_s = get_val_S(b_imp)
    ctx = event_context(
        cache.solver_observer, integrator.step, t, dt, 0, nothing, nothing, nothing,
    )

    emit(ctx, StepStart, u, cache.tableau)
    step_stages!(integrator, cache, v_s)
    emit(ctx, AssemblyInputs, u, dt, T_exp, T_lim, T_imp, b_exp, b_imp, cache.tableau)

    t_final = t + dt

    inc_exp = has_T_exp(f) ? fused_raw_increment(dt, b_exp, T_exp, v_s) : NullBroadcasted()
    inc_imp =
        !has_T_imp(f) ? NullBroadcasted() : fused_raw_increment(dt, b_imp, T_imp, v_s)

    if has_T_lim(f)
        assign_fused_increment!(temp, u, dt, b_exp, T_lim, v_s)
        op!(ctx, :final_lim, :final_limiter, temp, u, lim!, temp, p, t_final, u)
        assign_with_increments!(u, temp, inc_exp, inc_imp)
    else
        assign_with_increments!(u, u, inc_exp, inc_imp)
    end

    # End of step is a genuine state-ready moment; fire `EndOfStepSignal`
    # unconditionally. `EndOfStep <: EndOfStage <: WithDSS`, so all three
    # handler families see it. For FSAL tableaux the redundant post-Newton
    # firing at end of stage `s` is skipped inside `solve_stage_implicit!`
    # (since `u ≡ U_s`), so no double firing occurs.
    op!(ctx, :final_dss, :final_map, u, nothing, dss!, u, p, t_final)
    needs_update!(f.update_constrain_state, EndOfStepSignal()) &&
        op!(ctx, :final_constrain, :final_map, u, nothing, constrain_state!, u, p, t_final)
    needs_update!(f.update_cache, EndOfStepSignal()) &&
        op!(ctx, :final_cache, :final_map, u, nothing, cache!, u, p, t_final)

    return u
end


@inline update_stage!(integrator, cache, ::Tuple{}) = nothing
@inline update_stage!(integrator, cache, is::Tuple{Any}) =
    update_stage!(integrator, cache, first(is))
@inline function update_stage!(integrator, cache, is::Tuple)
    update_stage!(integrator, cache, first(is))
    update_stage!(integrator, cache, Base.tail(is))
end
@inline function update_stage!(integrator, cache::IMEXARKCache, v_i::Val{i}) where {i}
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = cache.tableau
    (; U, T_lim, T_exp, T_imp, temp) = cache

    t_exp = t + dt * c_exp[i]
    t_imp = t + dt * c_imp[i]
    dtγ = float(dt) * a_imp[i, i]
    ctx = event_context(cache.solver_observer, integrator.step, t, dt, i, dtγ, t_exp, t_imp)

    compute_stage_value!(U, u, dt, a_exp, a_imp, T_lim, T_exp, T_imp, f, p, t_exp, v_i, ctx)
    solve_stage_implicit!(U, temp, p, t_exp, t_imp, dtγ, i, f, alg, cache, ctx)
    evaluate_stage_tendencies!(
        T_exp, T_lim, T_imp, U, temp,
        p, t_exp, t_imp, dtγ, v_i, f, a_exp, b_exp, a_imp, b_imp, ctx,
    )

    return nothing
end

"""
    compute_stage_value!(U, u, dt, a_exp, a_imp, T_lim, T_exp, T_imp, f, p, t_exp, v_i::Val{i}, [ctx])

Compute the stage value `U` by accumulating the linear combination dictated by the
Butcher tableau from previous explicit (limited + unlimited) and implicit tendencies.
`ctx` is an [`EventContext`](@ref) or `nothing` (no events).
"""
@inline function compute_stage_value!(
    U,
    u,
    dt,
    a_exp,
    a_imp,
    T_lim,
    T_exp,
    T_imp,
    f,
    p,
    t_exp,
    v_i::Val{i},
    ctx = nothing,
) where {i}
    inc_exp = has_T_exp(f) ? fused_raw_increment(dt, a_exp, T_exp, v_i) : NullBroadcasted()
    inc_imp =
        !has_T_imp(f) ? NullBroadcasted() : fused_raw_increment(dt, a_imp, T_imp, v_i)

    if has_T_lim(f)
        assign_fused_increment!(U, u, dt, a_exp, T_lim, v_i)
        i ≠ 1 && op!(ctx, :stage_lim, :stage_construction, U, u, f.lim!, U, p, t_exp, u)
        assign_with_increments!(U, U, inc_exp, inc_imp)
    else
        assign_with_increments!(U, u, inc_exp, inc_imp)
    end
end

"""
    solve_stage_implicit!(U, temp, p, t_exp, t_imp, dtγ, i, f, alg, cache, [ctx])

Apply DSS to the stage value, set up the implicit solve (calling
`initialize_imp!` and refreshing the implicit cache), and run the implicit
solver. The pre-solve DSS is skipped on stage 1 (the state is already
continuous from the end of the previous timestep), but `initialize_imp!` and
the implicit-cache refresh still run. Skips the implicit solve when γ == 0.
Shared by the IMEX-ARK and IMEX-SSPRK stage loops; `ctx` is an
[`EventContext`](@ref) or `nothing` (no events, the IMEX-SSPRK case).
"""
@inline function solve_stage_implicit!(
    U,
    temp,
    p,
    t_exp,
    t_imp,
    dtγ,
    i,
    f,
    alg,
    cache,
    ctx = nothing,
)
    (; T_imp!, dss!, constrain_state!, cache!, cache_imp!, initialize_imp!) = f
    (; update_constrain_state, update_cache) = f
    (; newtons_method_cache) = cache
    (; newtons_method) = alg

    # No-implicit stage → state is ready for tendency eval right after DSS
    # (fire `EndOfStageSignal`, which is `<: WithDSS`, so both handler
    # families see it). Otherwise fire only `WithDSSSignal` (pre-implicit
    # DSS — the state isn't state-ready until after the Newton solve).
    no_implicit_stage = !has_T_imp(f) || iszero(dtγ)
    stage_top_sig = no_implicit_stage ? EndOfStageSignal() : WithDSSSignal()

    if i ≠ 1
        op!(ctx, :stage_dss_pre, :stage_construction, U, nothing, dss!, U, p, t_exp)
        needs_update!(update_constrain_state, stage_top_sig) && op!(
            ctx, :stage_constrain_pre, :stage_construction, U, nothing,
            constrain_state!, U, p, t_exp,
        )
        no_implicit_stage &&
            needs_update!(update_cache, EndOfStageSignal()) && op!(
                ctx, :stage_cache_no_implicit, :stage_construction, U, nothing,
                cache!, U, p, t_exp,
            )
    end

    no_implicit_stage && return

    @assert !isnothing(newtons_method)
    @. temp = U
    emit(ctx, StageBaseline, temp, U)

    if !isnothing(initialize_imp!)
        op!(ctx, :init_imp, :stage_processing, U, nothing, initialize_imp!, U, p, dtγ)
        op!(ctx, :stage_dss_post_init, :stage_processing, U, nothing, dss!, U, p, t_exp)
        needs_update!(update_constrain_state, WithDSSSignal()) && op!(
            ctx, :stage_constrain_post_init, :stage_processing, U, nothing,
            constrain_state!, U, p, t_exp,
        )
        op!(
            ctx, :cache_imp_pre_newton, :subordinate_stage_write, U, nothing,
            cache_imp!, U, p, t_imp,
        )
    else
        i ≠ 1 && op!(
            ctx, :cache_imp_pre_newton, :subordinate_stage_write, U, nothing,
            cache_imp!, U, p, t_imp,
        )
    end

    solve_implicit_equation!(
        U, temp, p, t_imp, dtγ,
        T_imp!, newtons_method, newtons_method_cache, cache_imp!, ctx,
    )

    # Post-Newton correction (optional): evaluate `T_post_imp!` on the
    # Newton-solved `U` and apply `U ← U + dtγ · dY_post`. The correction
    # is folded into `T_imp[i]` by the downstream `(U − temp) / dtγ`
    # computation, and inherits the same `b_imp` weighting, so the ARK
    # scheme structure is preserved. Use `cache.T_imp[i]` as scratch.
    if has_T_post_imp(f)
        op!(
            ctx, :cache_imp_post_newton, :subordinate_stage_write, U, nothing,
            cache_imp!, U, p, t_imp,
        )
        T_post = cache.T_imp[i]
        emit(ctx, Operation, :post_imp_correction, :subordinate_stage_write, :entry, U, T_post)
        f.T_post_imp!(T_post, U, p, t_imp)
        @. U += dtγ * T_post
        emit(ctx, Operation, :post_imp_correction, :subordinate_stage_write, :exit, U, T_post)
    end

    # Post-Newton: DSS the solved stage value. Fire `EndOfStageSignal` for
    # `cache!` / `constrain_state!`, EXCEPT at the last stage of an FSAL
    # tableau (where `u ≡ U_s` and end-of-step handles it). When `cache!`
    # is skipped by policy, fall back to `cache_imp!` so the implicit
    # cache still tracks the post-DSS solved state.
    op!(ctx, :stage_dss_post, :stage_processing, U, nothing, dss!, U, p, t_imp)
    s = val_to_int(get_val_S(cache.tableau.b_imp))
    if !(i == s && is_fsal(alg))
        needs_update!(update_constrain_state, EndOfStageSignal()) && op!(
            ctx, :stage_constrain_post, :stage_processing, U, nothing,
            constrain_state!, U, p, t_imp,
        )
        if needs_update!(update_cache, EndOfStageSignal())
            op!(ctx, :stage_cache_post, :stage_processing, U, nothing, cache!, U, p, t_imp)
        else
            op!(
                ctx, :stage_cache_fallback, :subordinate_stage_write, U, nothing,
                cache_imp!, U, p, t_imp,
            )
        end
    end
    return nothing
end

"""
    evaluate_stage_tendencies!(T_exp, T_lim, T_imp, U, temp,
                               p, t_exp, t_imp, dtγ, v_i, f,
                               a_exp, b_exp, a_imp, b_imp, [ctx])

Evaluate explicit and implicit tendencies at the current stage value for use
in later stages and the final update.
"""
@inline function evaluate_stage_tendencies!(
    T_exp, T_lim, T_imp, U, temp,
    p, t_exp, t_imp, dtγ, v_i::Val{i}, f, a_exp, b_exp, a_imp, b_imp, ctx = nothing,
) where {i}
    has_exp_terms = !zero_column(typeof(a_exp), v_i) || !zero_coeff(typeof(b_exp), i)
    if has_exp_terms
        has_T_exp_T_lim(f) && f.T_exp_T_lim!(T_exp[i], T_lim[i], U, p, t_exp)
    end

    # Implicit tendencies
    has_imp_terms = !zero_column(typeof(a_imp), v_i) || !zero_coeff(typeof(b_imp), i)
    if has_imp_terms
        if iszero(dtγ)
            # γ == 0: T_imp is treated explicitly
            has_T_imp(f) && f.T_imp!(T_imp[i], U, p, t_imp)
        else
            # γ ≠ 0: T_imp satisfies the implicit equation
            if has_T_imp(f)
                @. T_imp[i] = (U - temp) / dtγ
                emit(ctx, EffectiveStorage, i, T_imp[i], U, temp, dtγ)
            end
        end
    end
end

"""
    solve_implicit_equation!(U, U₀, p, t_imp, dtγ, T!, newtons_method,
                             newtons_method_cache, cache_imp!, [ctx])

Solve the implicit stage equation ``U = U₀ + Δtγ\\, T(U)`` via Newton's
method. Post-Newton bookkeeping (`dss!`, `constrain_state!`, `cache!`) is
handled by the caller so that stage-index-dependent gating (e.g. skipping
the final-stage updates for FSAL tableaux) can be applied.
"""
@inline function solve_implicit_equation!(
    U,
    U₀,
    p,
    t_imp,
    dtγ,
    T!,
    newtons_method,
    newtons_method_cache,
    cache_imp!,
    ctx = nothing,
)
    implicit_equation_residual! =
        (residual, U′) -> begin
            T!(residual, U′, p, t_imp)
            @. residual = U₀ + dtγ * residual - U′
        end
    solve_newton!(
        newtons_method,
        newtons_method_cache,
        U,
        implicit_equation_residual!,
        (jacobian, U′) -> T!.Wfact(jacobian, U′, p, dtγ, t_imp),
        U′ -> op!(
            ctx, :cache_imp_newton, :subordinate_stage_write, U′, nothing,
            cache_imp!, U′, p, t_imp,
        ),
    )
    return nothing
end
