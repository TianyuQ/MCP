






function jacobian()
end

stage_costs = map(1:N) do ii
    (x, u, t, θi) -> let
    goal = θi[end-(N+1):end-N]
    mask = θi[end-(N-1):end]
        norm_sqr(x[Block(ii)][1:2] - goal) + norm_sqr(x[Block(ii)][3:4]) + 0.1 * norm_sqr(u[Block(ii)]) + 2 * sum((mask[ii] * mask[jj]) / norm_sqr(x[Block(ii)][1:2] - x[Block(jj)][1:2]) for jj in 1:N if jj != ii)
    end
end