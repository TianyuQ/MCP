module PlayerSelectionTraining

using MixedComplementarityProblems: MixedComplementarityProblems

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
# using BlockArrays: mortar, blocks, BlockArray, Block
using GLMakie: GLMakie
using Makie: Makie
using LinearAlgebra: norm_sqr, norm
# using ProgressMeter: ProgressMeter

###############################################################################
# Import Required Packages
###############################################################################
using BlockArrays
using Flux
using Flux.Losses: mse
using Optimisers
using Zygote
using ProgressMeter
using JSON
using Statistics
using Random
using LinearAlgebra
using Glob
# using CUDA
using BSON
using CSV
using DataFrames
using ForwardDiff
using Distributions

# include("utils.jl")
include("train_and_test_utils.jl")
include("parametric_masked_game_solver.jl")
include("train_new.jl")
# include("game_with_masks.jl")
# include("gradient_test.jl")
# include("lane_change.jl")

end # module PlayerSelectionTraining
