using Flux
using Zygote

function build_model()
    model = Chain(
        Dense(480, 256, relu),
        Dense(256, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 16, relu),
        Dense(16, 8, relu),
        Dense(8, 4, sigmoid)  # Sigmoid outputs probabilities for Bernoulli sampling
    )
    return model
end

model = build_model()

context = randn(480)

output = model(context)

jacobian = Flux.jacobian(x -> model(x), context)
println(size(jacobian[1]))

random_gradient = randn(4)

true_gradient = transpose(random_gradient) * jacobian[1]
println(true_gradient)
println(size(true_gradient))