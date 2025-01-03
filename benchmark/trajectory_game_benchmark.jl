module TrajectoryGameBenchmarkUtils

using LazySets: LazySets
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    PolygonEnvironment,
    ProductDynamics,
    TimeSeparableTrajectoryGameCost,
    TrajectoryGame,
    GeneralSumCostStructure,
    num_players,
    time_invariant_linear_dynamics,
    unstack_trajectory,
    stack_trajectories,
    state_dim,
    control_dim,
    state_bounds,
    control_bounds,
    OpenLoopStrategy,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout
using TrajectoryGamesExamples: planar_double_integrator, animate_sim_steps
using BlockArrays: mortar, blocks, BlockArray, Block
using LinearAlgebra: norm_sqr, norm
using ProgressMeter: ProgressMeter

include("../examples/utils.jl")
include("../examples/lane_change.jl")

end # module TrajectoryGameBenchmarkUtils

"Generate a random trajectory game, based on the `LaneChange` problem in `examples/`."
function generate_test_problem(::TrajectoryGameBenchmark; horizon = 10)
    (;
        G,
        H,
        K,
        unconstrained_dimension,
        constrained_dimension,
        lower_bounds,
        upper_bounds,
    )
end

"Generate a random parameter vector Θ corresponding to a convex QP."
function generate_random_parameter(
    ::QuadraticProgramBenchmark;
    rng,
    num_primals,
    num_inequalities,
    sparsity_rate,
)
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
function unpack_parameters(::QuadraticProgramBenchmark, θ; num_primals, num_inequalities)
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
