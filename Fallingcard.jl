using DifferentialEquations
using Plots
using LinearAlgebra
using Random

const CT  = 1.2
const CR  = pi
const A   = 1.4
const B   = 1.0
const mu1 = 0.2
const mu2 = 0.2

function falling_card!(du, u, p, t)
    vx, vy, theta, omega, x, y = u
    Istar, mu1_, mu2_ = p         
    speed2 = vx^2 + vy^2
    speed  = speed2 > 1e-14 ? sqrt(speed2) : 0.0

    Gamma = speed > 1e-8 ?
        (2/pi) * (-CT * vx * vy / speed + CR * omega) :
        (2/pi) * CR * omega

    Fnu_coef = speed2 > 1e-14 ?
        (1/pi) * (A - B * (vx^2 - vy^2) / speed2) * speed :
        0.0
    Fx_nu = Fnu_coef * vx
    Fy_nu = Fnu_coef * vy

    tau_nu = (mu1_ + mu2_ * abs(omega)) * omega   

    vx_dot    = ((Istar + 1) * omega * vy - Gamma * vy - sin(theta) - Fx_nu) / Istar
    vy_dot    = (-Istar * omega * vx + Gamma * vx - cos(theta) - Fy_nu) / (Istar + 1)
    omega_dot = (-vx * vy - tau_nu) / (0.25 * (Istar + 0.5))

    xdot = vx * cos(theta) - vy * sin(theta)
    ydot = vx * sin(theta) + vy * cos(theta)

    du[1] = vx_dot; du[2] = vy_dot; du[3] = omega
    du[4] = omega_dot; du[5] = xdot; du[6] = ydot
end

function simulate(Istar; mu1v=mu1, mu2v=mu2, t_max=150.0, u0=nothing)  
    if u0 === nothing
        W = sqrt(pi / (A + B))
        u0 = [0.05, -0.98 * W, 0.05, 0.05, 0.0, 0.0]
    end
    tspan = (0.0, t_max)
    prob = ODEProblem(falling_card!, u0, tspan, (Istar, mu1v, mu2v)) 
    sol = solve(prob, Tsit5(); reltol=1e-10, abstol=1e-12, maxiters=1e7, dtmax=0.05)
    return sol
end

function lyapunov_exponent(Istar; mu1v=mu1, mu2v=mu2, d0=1e-8, dt_renorm=0.5,
                           n_renorm=4000, transient=200.0, seed=1)
    Random.seed!(seed)
    p = (Istar, mu1v, mu2v)

    W = sqrt(pi / (A + B))
    u0 = [0.05, -0.98 * W, 0.05, 0.05, 0.0, 0.0]
    sol0 = solve(ODEProblem(falling_card!, u0, (0.0, transient), p), Tsit5();
                 reltol=1e-11, abstol=1e-13, dtmax=0.05, save_everystep=false)
    u_ref = sol0.u[end]

    dir = randn(4); dir ./= norm(dir)
    u_pert = copy(u_ref); u_pert[1:4] .+= d0 .* dir

    logsum = 0.0
    for _ in 1:n_renorm
        sol_ref  = solve(ODEProblem(falling_card!, u_ref,  (0.0, dt_renorm), p), Tsit5();
                          reltol=1e-11, abstol=1e-13, dtmax=0.05, save_everystep=false)
        sol_pert = solve(ODEProblem(falling_card!, u_pert, (0.0, dt_renorm), p), Tsit5();
                          reltol=1e-11, abstol=1e-13, dtmax=0.05, save_everystep=false)
        u_ref_new, u_pert_new = sol_ref.u[end], sol_pert.u[end]

        sep = u_pert_new[1:4] .- u_ref_new[1:4]
        d1  = norm(sep)
        logsum += log(d1 / d0)

        u_ref  = u_ref_new
        u_pert = copy(u_ref_new); u_pert[1:4] .+= (d0 / d1) .* sep
    end
    return logsum / (n_renorm * dt_renorm)
end

function check_lyapunov(; Istar=2.2)
    lambda = lyapunov_exponent(Istar)
    println("Max Lyapunov at I* = $Istar: $(round(lambda, digits=3))  (paper: 0.13 ± 0.01)")
    return lambda
end

function classify(Istar, mu1v; t_max=400.0, transient=200.0, chaos_tol=0.02)
    sol = simulate(Istar; mu1v=mu1v, mu2v=mu1v, t_max=t_max)
    idx = searchsortedfirst(sol.t, transient)
    omega_series = [u[4] for u in sol.u[idx:end]]
    theta_series = [u[3] for u in sol.u[idx:end]]

    maximum(abs.(omega_series)) < 1e-3 && return :steady

    lambda = lyapunov_exponent(Istar; mu1v=mu1v, mu2v=mu1v,
                               n_renorm=200, dt_renorm=0.5, transient=transient)
    lambda > chaos_tol && return :chaotic

    net_rotation = abs(theta_series[end] - theta_series[1])
    return net_rotation > 2pi ? :tumbling : :fluttering
end

function make_fig4b(; Istar_range=1.0:0.1:4.0, mu1_range=1.0:0.1:4.0,
                     outfile="fig4b.png")
    style = Dict(:steady => :black, :fluttering => :blue,
                 :tumbling => :red, :chaotic => :green)
    pts = Dict(k => (Float64[], Float64[]) for k in keys(style))

    for Istar in Istar_range, mu1v in mu1_range
        regime = classify(Istar, mu1v)
        push!(pts[regime][1], Istar); push!(pts[regime][2], mu1v)
    end

    fig = plot(xlabel="I*", ylabel="mu1", title="Phase diagram (Fig. 4b)", legend=:topright)
    for (regime, color) in style
        xs, ys = pts[regime]
        scatter!(fig, xs, ys, label=String(regime), color=color, markersize=4)
    end
    savefig(fig, outfile)
    return fig
end

function edge_on_transit_time(Istar; mu1v=mu1, mu2v=mu2, nudge=1e-6,
                               departure_tol=0.3, t_max=400.0)
    u0 = [0.0, 0.0, pi/2, nudge, 0.0, 0.0]   
    p  = (Istar, mu1v, mu2v)

    condition(u, t, integrator) = abs(u[4]) - departure_tol
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    sol = solve(ODEProblem(falling_card!, u0, (0.0, t_max), p), Tsit5();
                reltol=1e-12, abstol=1e-14, dtmax=0.02, callback=cb)
    return sol.t[end]
end

function make_fig5d(; Ic=1.2191, outfile="fig5d.png")
    deltas = [10.0^(-k) for k in 1:0.5:5]
    below, above = Ic .- deltas, Ic .+ deltas
    T_below = [edge_on_transit_time(I) for I in below]
    T_above = [edge_on_transit_time(I) for I in above]

    # println("delta = ", deltas)
    # println("T_below = ", T_below)
    # println("T_above = ", T_above)

    fig = plot(yscale=:log10, xlabel="T", ylabel="|I* - I*_c|",
               title="Log divergence near I*_c (Fig. 5d)")
    scatter!(fig, T_below, deltas, label="I* < I*_c", marker=:circle)
    scatter!(fig, T_above, deltas, label="I* > I*_c", marker=:star5)

    #T ~ T0 + slope*log(1/delta); slope = 2/lambda_u
    xs = log.(1 ./ deltas); allx = vcat(xs, xs); allT = vcat(T_below, T_above)
    slope = (length(allx)*sum(allx.*allT) - sum(allx)*sum(allT)) /
            (length(allx)*sum(allx.^2) - sum(allx)^2)
    println("lambda_u = $(round(2/slope, digits=4))  (paper: 0.3813)")

    savefig(fig, outfile)
    return fig
end
 
function make_fig3(; t_max=150.0, outfile="fig3.png")
    cases  = [1.1, 1.4, 1.45, 1.6, 2.2, 3.0]
    labels = ["(a) periodic fluttering", "(b) period-one tumbling", "(c) period-two tumbling",
              "(d) flutter/tumble mix", "(e) chaotic", "(f) small-amp broadside flutter"]
 
    plots_ = Plots.Plot[]
    for (Istar, lab) in zip(cases, labels)
        sol = simulate(Istar; t_max=t_max)
        x = [u[5] for u in sol.u]
        y = [u[6] for u in sol.u]
        p = plot(x, -y,                       
                 lw=0.8, legend=false, yflip=true,
                 title="I* = $(Istar)\n$(lab)", titlefontsize=9,
                 xlabel="x", ylabel="y", aspect_ratio=:equal)
        push!(plots_, p)
    end
 
    fig = plot(plots_..., layout=(2, 3), size=(1500, 900))
    savefig(fig, outfile)
    return fig
end
 
function make_theta_diagnostics(; t_max=150.0, outfile="theta.png")
    cases = [1.1, 1.4, 1.45, 1.6, 2.2, 3.0]
    plots_ = Plots.Plot[]
    for Istar in cases
        sol = simulate(Istar; t_max=t_max)
        t = sol.t
        theta = [u[3] for u in sol.u]
        p = plot(t, theta ./ pi, lw=0.8, legend=false,
                 title="I* = $(Istar)", xlabel="t", ylabel="theta/pi")
        push!(plots_, p)
    end
    fig = plot(plots_..., layout=(2, 3), size=(1500, 700))
    savefig(fig, outfile)
    return fig
end

function run_all(; t_max=150.0)
    make_fig3(t_max=t_max)
    make_theta_diagnostics(t_max=t_max)
    check_lyapunov()
    make_fig4b()
    make_fig5d()
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_fig3()
    make_theta_diagnostics()
    check_lyapunov()          
    make_fig4b()          
    make_fig5d()           
end