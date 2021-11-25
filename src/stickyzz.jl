#=
# Inventory

Boundaries, Reference #Q

State space
# u = (x, v, f)

Flow # 

Gradient of negative log-density ϕ    

Graph

Skeleton 
=#
using Test

struct StickyBarriers{Tx,Trule,Tκ}
    x::Tx # Intervals
    rule::Trule # set velocity to 0 and change label from free to frozen
    κ::Tκ 
end    

struct StickyFlow{T}
    old::T
end

struct StickyUpperBounds{TG,TG2,TΓ,Tstrong,Tc,Tfact}
    G1::TG
    G2::TG2
    Γ::TΓ
    strong::Tstrong
    adapt::Bool
    c::Tc
    factor::Tfact
end

struct StructuredTarget{TG,T∇ϕ}
    G::TG
    ∇ϕ::T∇ϕ
    #selfmoving ? 
end


StructuredTarget(Γ::SparseMatrixCSC, ∇ϕ) = StructuredTarget([i => rowvals(Γ)[nzrange(Γ, i)] for i in axes(Γ, 1)], ∇ϕ)

function ab(su::StickyUpperBounds, flow, i, u)
    (t, x, v) = u
    ab(su.G1, i, x, v, su.c, flow.old) #TODO
end

struct AcceptanceDiagnostics
    acc::Int
    num::Int
end

function stickystate(x0)
    d = length(x0)
    v0 = rand((-1.0, 1.0), d)
    t0 = zeros(d)
    u0 = (t0, x0, v0) 
end

struct EndTime
    T::Float64
end
finished(end_time::EndTime, t) = t < end_time.T

dir(ui) = dir = ui[3] > 0 ? 1 : 2
geti(u, i) = (u[1][i], u[2][i], u[3][i]) 

function freezing_time(barrier, ui)
    t,x,v = ui
    di = dir(ui)
    if  v*(x-barrier.x[di]) >= 0 # sic!
        return Inf
    else
        return -(x - barrier.x[di])/v
    end
end


function stickyzz(u0, target::StructuredTarget, flow::StickyFlow, upper_bounds::StickyUpperBounds, barriers::Vector{<:StickyBarriers}, end_condition)
    # Initialize
    (t0, x0, v0) = u0
    d = length(v0)
    t′ = maximum(t0)
    # priority queue
    Q = SPriorityQueue{Int,Float64}()
    # Skeleton
    Ξ = Trace(t′, x0, v0, flow.old) # TODO use trace
    # Diagnostics
    acc = AcceptanceDiagnostics(0, 0)
    ## create bounds ab
    b = [ab(upper_bounds, flow, i, u0) for i in eachindex(v0)]
    f = zeros(Bool, d)
    # fill priorityqueue
    for i in eachindex(v0)
        trefl = poisson_time(b[i], rand()) #TODO
        tfreez = freezing_time(barriers[i], geti(u0, i)) #TODO
        if trefl > tfreez
            f[i] = true
            enqueue!(Q, i => t0[i] + tfreez)
        else
            f[i] = false
            enqueue!(Q, i => t0[i] + trefl)
        end
    end
    rng = Random.GLOBAL_RNG
    println("Run main, run total")

    Ξ = @time @inferred sticky_main(rng, Q, Ξ, t′, u0, b, f, target, flow, upper_bounds, barriers, end_condition, acc)

    return Ξ
end

function sticky_main(rng, Q::SPriorityQueue, Ξ, t′, u, b, f, target, flow, upper_bounds, barriers, end_condition, acc)
    (t, x, v) = u
    u_old = (copy(t), x, copy(v))
    while finished(end_condition, t′) 
        t, x, v = u
        i, t′ = stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, f, target, flow, upper_bounds, barriers, acc)
        t, x, v = u
        push!(Ξ, event(i, t, x, v, flow.old))
    end
    return Ξ
end
function stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, f, target, flow, upper_bounds, barriers, acc)
    while true
        t, x, v = u
        t_old, _, v_old = u_old
        i, t′ = peek(Q)
        G = target.G
        G1 =  upper_bounds.G1
        G2 =  upper_bounds.G2
        if f[i] # case 1) to be frozen
            x[i] = -0*v[i] # a bit dangarous
            t_old[i] = t[i] = t′
            v_old[i], v[i] = v[i], 0.0 # stop and save speed
            f[i] = false
            di = dir(geti(u, i))
            Q[i] = t′ - log(rand(rng))/barriers[i].κ[di]
            if upper_bounds.strong == false
                t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
                t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)
                for j in neighbours(G1, i)
                    if v[j] != 0 # only non-frozen, especially not i
                        b[j] = ab(upper_bounds, flow, j, u)
                        t_old[j] = t[j]
                        Q = queue_time!(Q, u..., j, b, f, flow.old)
                    end
                end
            end
            push!(Ξ, event(i, t, x, v, flow.old))
            return i, t′
        elseif x[i] == 0 && v[i] == 0 # case 2) was frozen
            t_old[i] = t[i] = t′ # particle does not move, only time
            v[i], v_old[i] = v_old[i], 0.0 # unfreeze, restore speed
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)
            for j in neighbours(G1, i)
                if v[j] != 0 # only non-frozen, including i # check!
                    b[j] = ab(upper_bounds, flow, j, u)
                    t_old[j] = t[j]
                    Q = queue_time!(Q, u..., j, b, f, flow.old)
                end
            end
            push!(Ξ, event(i, t, x, v, flow.old))
            return i, t′ 
        else    # was either a reflection 
                #time or an event time from the upper bound  
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            # ∇ϕi = ∇ϕ_(∇ϕ, t, x, θ, i, t′, F, args...)
            ∇ϕi = target.∇ϕ(x, i) # To change, why Γ is on upperbounds?            
            # l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
            l, lb = sλ(∇ϕi, i, x, v, flow.old), sλ̄(b[i], t[i] - t_old[i])
            # acc.num +=1        
            if rand()*lb < l # was a reflection time
                # acc.acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                    acc = num = 0
                    adapt!(c, i, factor)
                end
                v = reflect!(i, ∇ϕi, x, v, flow.old) # reflect!
                t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)  # neighbours of neightbours \ neighbours
                for j in neighbours(G1, i)
                    if v[j] != 0
                        b[j] = ab(upper_bounds, flow, j, u)
                        t_old[j] = t[j]
                        queue_time!(Q, u..., j, b, f, flow.old)
                    end
                end
                push!(Ξ, event(i, t, x, v, flow.old))
                return i, t′ 
            else # was an event time from upperbound -> nothing happens
                b[i] = ab(upper_bounds, flow, i, u)
                t_old[i] = t[i]
                queue_time!(Q, u..., i, b, f, flow.old)
                # don't save
                continue
            end               
        end
    end
end