
using LinearAlgebra
using Test
using ZigZagBoomerang
using SparseArrays
using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)

d = 20
x = randn(d)
∇x = randn(d)
const σa = 5.0
const σb = 0.6
# ϕ(x) = 0.5sum(-(x[i]/σa)^2  - ((2.0 - x[i] + x[mod1(i-1, length(x))])/σb)^2 for i in 1:length(x))

ϕ(x) = 0.5sum(-(x[i]/σa)^2 for i in 1:length(x)) + 0.5*sum( -((0.0 - x[i] + x[i+1])/σb)^2 for i in 1:length(x)-1)

R = [i == j ? 1 : (j - i == -1 ? -1 : 0) for i in 2:d, j in 1:d]
@test [(0.0 - x[i] + x[i+1]) for i in 1:length(x)-1] ≈ R*x
c1 = [(i == 1 || i == d-1) ? -1 : -1 for i in 1:d-1]
@test R*x .+ ones(d-1)  ≈ [(1.0 - x[i] + x[i+1]) for i in 1:length(x)-1]

@test x'*R'*R*x/σb^2  ≈ sum(((0.0 - x[i] + x[i+1])/σb)^2 for i in 1:length(x)-1)
# this is good
Γ = R'*R/σb^2 + I/σa^2
@test  exp(ϕ(x)) ≈ exp((-0.5*(x)'*Γ*(x)))  

#ok
ci = 2.0
ϕ(x) = 0.5sum(-(x[i]/σa)^2 for i in 1:length(x)) + 0.5*sum( -((ci - x[i] + x[i+1])/σb)^2 for i in 1:length(x)-1)
#only for $\sigmaa = Inf$
μ = -inv(Γ)*R'/σb^2*(ones(d-1).*ci)
x = rand(d)
c = ϕ(x) + 0.5*(x - μ)'*Γ*(x - μ)
x = rand(d)
@test (ϕ(x)) ≈ (-0.5*(x - μ)'*Γ*(x - μ)) + c



sΓ = sparse(Γ)
T = 10000.0
Z = ZigZag(sΓ, μ) 
∇ϕ(x, i, Γ, μ) = ZigZagBoomerang.idot(Γ, i, x) -  ZigZagBoomerang.idot(Γ, i, μ)
# prior w = 0.5
wi = 0.2
ki = 1/(sqrt(2*π*σa^2))/(1/wi - 1)
println("k equal to $(ki)")
x0 = rand(d)
κ = ki*ones(length(x0))
c = fill(0.001, d)
θ0 = rand([-0.1,0.1], d)
t0 = 0.0

run_zigzag = true
if run_zigzag
    println("run once to trigger precompilation")
    ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, Γ, μ)

    println("sticky Zig-Zag")
    # timer inside sspdmp2
    x0 = fill(5.0, d)
    trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, Γ, μ)
    ts, xs = ZigZagBoomerang.sep(collect(trace))
    tsh, xsh = ZigZagBoomerang.sep(collect(discretize(trace, T/1000)))
    traceh = [xsh[i][j] for i in 1:length(xsh), j in 1:d]
end



using GLMakie
produce_heatmap = true
if produce_heatmap
    burn = 50
    fig1 = Figure()
    ax = [Axis(fig1[1, j]) for j in 1:3]
    heatmap!(ax[1], traceh[burn:end,:], colorrange=(-8,8))
    i = 2d÷5
    lines!(ax[2], getindex.(xs, i)[burn:end],getindex.(xs, i+1)[burn:end], linewidth=2.0, color=(:blue, 0.1))

    lines!(ax[3], ts[burn:end], getindex.(xs, i)[burn:end])
    lines!(ax[3], ts[burn:end], getindex.(xs, i+1)[burn:end])
    fig1
end


run_gibbs = true
if run_gibbs
    include("./gibbs_gauss2.jl")
    x = copy(x0)
    w = wi
    N = 20000
    Z = [abs(d÷2 - i) > 2 for i in eachindex(x)]
    # Γℓ = R'*R/σb^2
    # μℓ = -Γℓ \ (R'*(ones(d-1).*ci)./σb^2)
    ββ, ZZ = @time gibbs_gauss(Γ, μ, w, N, x, Z, σa, 10)
    trace2 = [ββ[i].*ZZ[i] for i in 1:length(ZZ)]
    trace2b = [ββ[i][j].*ZZ[i][j] for i in 1:length(ZZ), j in 1:length(ZZ[1])] 
end

produce_heatmap = true
if produce_heatmap
    using GLMakie, Colors
    #fig2 = Figure()
    #ax = [Axis(fig2[1, j]) for j in 1:1]
    ax = [Axis(fig1[2, j]) for j in 1:3]
    jj = axes(trace2b, 1)[burn:end]
    heatmap!(ax[1], trace2b[jj,:],  colorrange=(-8,8))
    i = 2d÷5
    lines!(ax[2], trace2b[jj, i], trace2b[jj, i+1], linewidth=2.0, color=(:blue, 0.1))

    scatter!(ax[3], jj, trace2b[jj, i], markersize=3)
    scatter!(ax[3], jj, trace2b[jj, i+1], markersize=3)
  
end
fig1