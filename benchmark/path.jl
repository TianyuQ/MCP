""" Generate a random large (convex) quadratic problem of the form
                               min_x 0.5 xᵀ M x - ϕᵀ x
                               s.t.  Ax - b ≥ 0.

NOTE: the problem may not be feasible!
"""
function generate_test_problem(; num_primals, num_inequalities)
    G(x, y; θ) =
        let
            (; M, A, ϕ) = unpack_parameters(θ; num_primals, num_inequalities)
            M * x - ϕ - A' * y
        end

    H(x, y; θ) =
        let
            (; A, b) = unpack_parameters(θ; num_primals, num_inequalities)
            A * x - b
        end

    K(z, θ) =
        let
            x = z[1:num_primals]
            y = z[(num_primals + 1):end]

            [G(x, y; θ); H(x, y; θ)]
        end

    (; G, H, K)
end

"Generate a random parameter vector Θ corresponding to a convex QP."
function generate_random_parameter(rng; num_primals, num_inequalities, sparsity_rate)
    bernoulli = Distributions.Bernoulli(1 - sparsity_rate)

    M = let
        P =
            randn(rng, num_primals, num_primals) .*
            rand(rng, bernoulli, num_primals, num_primals)
        P' * P
    end

    A =
        randn(rng, num_inequalities, num_primals) .*
        rand(rng, bernoulli, num_inequalities, num_primals)
    b = randn(rng, num_inequalities)
    ϕ = randn(rng, num_primals)

    [reshape(M, length(M)); reshape(A, length(A)); b; ϕ]
end

"Unpack a parameter vector θ into the components of a convex QP."
function unpack_parameters(θ; num_primals, num_inequalities)
    M = reshape(θ[1:(num_primals^2)], num_primals, num_primals)
    A = reshape(
        θ[(num_primals^2 + 1):(num_primals^2 + num_inequalities * num_primals)],
        num_inequalities,
        num_primals,
    )

    b =
        θ[(num_primals^2 + num_inequalities * num_primals + 1):(num_primals^2 + num_inequalities * (num_primals + 1))]
    ϕ = θ[(num_primals^2 + num_inequalities * (num_primals + 1) + 1):end]

    (; M, A, b, ϕ)
end

"Benchmark interior point solver against PATH on a bunch of random QPs."
function benchmark(;
    num_samples = 1000,
    num_primals = 100,
    num_inequalities = 100,
    sparsity_rate = 0.9,
    ip_mcp = nothing,
    path_mcp = nothing,
    ip_kwargs = (; tol = 1e-6),
)
    rng = Random.MersenneTwister(1)

    # Generate problem and random parameters.
    @info "Generating random problems..."
    problem = generate_test_problem(; num_primals, num_inequalities)

    θs = map(1:num_samples) do _
        generate_random_parameter(rng; num_primals, num_inequalities, sparsity_rate)
    end

    # Generate corresponding MCPs.
    @info "Generating IP MCP..."
    parameter_dimension = length(first(θs))
    ip_mcp =
        !isnothing(ip_mcp) ? ip_mcp :
        MixedComplementarityProblems.PrimalDualMCP(
            problem.G,
            problem.H;
            unconstrained_dimension = num_primals,
            constrained_dimension = num_inequalities,
            parameter_dimension,
        )

    @info "Generating PATH MCP..."
    lower_bounds = [fill(-Inf, num_primals); fill(0, num_inequalities)]
    upper_bounds = fill(Inf, num_primals + num_inequalities)
    path_mcp =
        !isnothing(path_mcp) ? path_mcp :
        ParametricMCPs.ParametricMCP(
            problem.K,
            lower_bounds,
            upper_bounds,
            parameter_dimension,
        )

    # Warm up the solvers.
    @info "Warming up IP solver..."
    MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        ip_mcp,
        first(θs);
        ip_kwargs...,
    )

    @info "Warming up PATH solver..."
    ParametricMCPs.solve(path_mcp, first(θs); warn_on_convergence_failure = false)

    # Solve and time.
    ip_data = @showprogress desc = "Solving IP MCPs..." map(θs) do θ
        elapsed_time = @elapsed sol = MixedComplementarityProblems.solve(
            MixedComplementarityProblems.InteriorPoint(),
            ip_mcp,
            θ;
            ip_kwargs...
        )

        (; elapsed_time, success = sol.status == :solved)
    end

    path_data = @showprogress desc = "Solving PATH MCPs..." map(θs) do θ
        # Solve and time.
        elapsed_time = @elapsed sol =
            ParametricMCPs.solve(path_mcp, θ; warn_on_convergence_failure = false)

        (; elapsed_time, success = sol.status == PATHSolver.MCP_Solved)
    end

    (; ip_mcp, path_mcp, ip_data, path_data)
end

"Compute summary statistics from solver benchmark data."
function summary_statistics(data)
    accumulate_stats(solver_data) = begin
        (; success_rate = fraction_solved(solver_data), runtime_stats(solver_data)...)
    end

    stats = (; ip = accumulate_stats(data.ip_data), path = accumulate_stats(data.path_data))
    @info "IP runtime is $(100(stats.ip.μ / stats.path.μ)) % that of PATH."

    stats
end

"Estimate mean and standard deviation of runtimes for all problems."
function runtime_stats(solver_data)
    filtered_times =
        map(datum -> datum.elapsed_time, filter(datum -> datum.success, solver_data))
    μ = Statistics.mean(filtered_times)
    σ = Statistics.stdm(filtered_times, μ)

    (; μ, σ)
end

"Compute fraction of problems solved."
function fraction_solved(solver_data)
    Statistics.mean(datum -> datum.success, solver_data)
end
