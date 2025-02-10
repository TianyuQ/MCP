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
using BlockArrays: mortar, blocks, BlockArray, Block
using GLMakie: GLMakie
using Makie: Makie
using LinearAlgebra: norm_sqr, norm
# using ProgressMeter: ProgressMeter

###############################################################################
# Import Required Packages
###############################################################################
using Flux
using Flux.Losses: mse
using Optimisers
using Zygote
using ProgressMeter
using JSON
# using Statistics
using Random
using LinearAlgebra
using Glob
using CUDA
using BSON
using CSV
using DataFrames

include("utils.jl")
include("masked_game_solver.jl")
include("train_and_test_utils.jl")
# include("train_mask_only.jl")
include("test.jl")

end # module PlayerSelectionTraining
