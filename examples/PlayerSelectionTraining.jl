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
# using LinearAlgebra: norm_sqr, norm
using ProgressMeter: ProgressMeter

###############################################################################
# Import Required Packages
###############################################################################
using Flux
using Flux.Losses: mse
using JSON
using Statistics
using Random
using LinearAlgebra
using Glob  # For dynamically finding all JSON files
using BSON  # For saving and loading models

include("utils.jl")
include("masked_game_solver.jl")
include("train_mask_only.jl")
include("test_mask_only.jl")

end # module PlayerSelectionTraining
