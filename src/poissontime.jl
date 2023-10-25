
"""
    poisson_time(a, b, u)

Obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = (a + b*t)^+, `a`,`b` ∈ R, `u` uniform random variable
"""
function poisson_time(a, b, u)
    if b > 0
        if a < 0
            return sqrt(-log(u)*2.0/b) - a/b
        else # a[i]>0
            return sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        end
    elseif b == 0
        if a > 0
            return -log(u)/a
        else # a[i] <= 0
            return Inf
        end
    else # b[i] < 0
        if a <= 0
            return Inf
        elseif -log(u) <= -a^2/b + a^2/(2*b)
            return -sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        else
            return Inf
        end
    end
end

"""
    poisson_time((a, b, c), u)

Obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = c + (a + b*t)^+, 
where `c`> 0 ,`a, b` ∈ R, `u` uniform random variable
"""
function poisson_time((a,b,c)::NTuple{3}, u)
    if b > 0
        if a < 0
            if -c*a/b + log(u) < 0.0
                return sqrt(-2*b*log(u) + c^2 + 2*a*c)/b - (a+c)/b
            else
                return -log(u)/c
            end
        else # a >0
            return sqrt(-log(u)*2.0*b + (a+c)^2)/b - (a+c)/b
        end
    elseif b == 0
        if a > 0
            return -log(u)/(a + c)
        else # a <= 0
            return -log(u)/c
        end
    else # b < 0
        if a <= 0.0 
            return -log(u)/c
        elseif  - c*a/b - a^2/(2*b)  + log(u) > 0.0
            return +sqrt((a+c)^2 - 2.0*log(u)*b)/b - (a+c)/b
        else
            return (-log(u)+ a^2/(2*b))/c
        end
    end
end

"""
    poisson_time(a[, u])

Obtaining waiting time for homogeneous Poisson Process
with rate of the form λ(t) = a, `a` ≥ 0, `u` uniform random variable
"""
function poisson_time(a::Number, u::Number)
    -log(u)/a
end

function poisson_time(rng, a)
    randexp(rng)/a
end
function poisson_time(a)
    randexp()/a
end

poisson_time((a, b)::Tuple, u=randn()) = poisson_time(a, b, u)

function poisson_time(t, b::Tuple, r)
    @assert t <= b[end] # check bound validity 
    Δt = t - b[1]
    a = b[2] + Δt*b[3]
    b = b[3]
    t + poisson_time((a, b, 0.01), r) # guarantee minimum rate
end

"""
Returns the real roots of the real cubic polynomial
    p0 + p1 x + p2 x^2 + p3 x^3
"""
function solve_cubic_eq(p0, p1, p2, p3)
    a, b, c, d = p0, p1, p2, p3
    # assert d != 0.0
    # x3 + a2x2 + a1x + a0
    a0 = a / d
    a1 = b / d
    a2 = c / d

    q = a1 / 3 - a2^2 / 9
    r = (a1 * a2 - 3 * a0) / 6 - a2^3 / 27

    if r^2 + q^3 > 0 # Numerical Recipes, only one real solution
        A = (abs(r) + sqrt(r^2 + q^3))^(1 / 3)
        t1 = A - q / A
        if r < 0
            t1 = -t1
        end
        x1 = t1 - a2 / 3
        return [x1]
    else  # Viete, three real
        theta = 0.0
        if q != 0
            theta = acos(r / (-q)^(3 / 2))
        end
        phi1 = theta / 3
        phi2 = theta / 3 - 2 * pi / 3
        phi3 = theta / 3 + 2 * pi / 3

        x1 = 2 * sqrt(-q) * cos(phi1) - a2 / 3
        x2 = 2 * sqrt(-q) * cos(phi2) - a2 / 3
        x3 = 2 * sqrt(-q) * cos(phi3) - a2 / 3

        return [x3, x2, x1]
    end
end


function solve_quadratic_integral_equation(y, a, b, c)
    """
    Solves t = argmin { y = \\int_0^t (a + bx + cx^2)+ dx }
    """
    if y < 0.0
        return Inf
    end

    P(x) = a * x + (b / 2) * x^2 + (c / 3) * x^3

    """
    Solve argmin x P(x) = y
    """
    function solve_P(y; min_root=0.0)
        roots = solve_cubic_eq(-y, a, b / 2, c / 3)
        return minimum(x -> !isnan(x) && x >= min_root ? x : Inf, roots)
    end

    # c ≃ 0: ab_poisson_time
    if abs(c) < 1e-10
        return poisson_time(a, b, exp(-y))  # ._.
    elseif c > 0
        # find zeros of p(x):
        # a/c + b/c x + x^2 = 0
        # (x + b/2c)^2 = -a/c + (b/2c)^2
        q = -a / c + (b / (2 * c))^2
        # if q <= 0: always non-negative
        if q <= 0
            return solve_P(y)
        elseif q > 0
            r0 = max(0.0, -b / (2 * c) - np.sqrt(q))
            r1 = max(0.0, -b / (2 * c) + np.sqrt(q))
            if P(r0) > y
                return solve_P(y)
            else
                return solve_P(y + P(r1) - P(r0); min_root=r1)
            end
        end
    else  # c < 0
        # find zeros of p(x):
        q = -a / c + (b / (2 * c))^2
        if q <= 0
            return Inf  # no solution
        else
            r0 = max(0.0, -b / (2 * c) - np.sqrt(q))
            r1 = max(0.0, -b / (2 * c) + np.sqrt(q))
            s0 = P(r1) - P(r0)  # total mass in [(r0)+, (r1)+]
            if s0 > y
                return solve_P(y + P(r0); min_root=r0)
            else
                return Inf
            end
        end
    end
end

function poisson_time(a, b, c, u) # Name conflict with poisson_time((a,b,c), u)
    return solve_quadratic_integral_equation(-log(u), a, b, c)
end