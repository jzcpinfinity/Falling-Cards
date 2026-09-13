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
    Istar = p
 
    speed2 = vx^2 + vy^2
    speed  = speed2 > 1e-14 ? sqrt(speed2) : 0.0
 
    #circulation (4.7)
    Gamma = speed > 1e-8 ?
        (2/pi) * (-CT * vx * vy / speed + CR * omega) :
        (2/pi) * CR * omega
 
    #(4.8)
    Fnu_coef = speed2 > 1e-14 ?
        (1/pi) * (A - B * (vx^2 - vy^2) / speed2) * speed :
        0.0
    Fx_nu = Fnu_coef * vx
    Fy_nu = Fnu_coef * vy
 
    #Dissipative torque (4.9)
    tau_nu = (mu1 + mu2 * abs(omega)) * omega
 
    #(5.1)-(5.3)
    vx_dot    = ((Istar + 1) * omega * vy - Gamma * vy - sin(theta) - Fx_nu) / Istar
    vy_dot    = (-Istar * omega * vx + Gamma * vx - cos(theta) - Fy_nu) / (Istar + 1)
    omega_dot = (-vx * vy - tau_nu) / (0.25 * (Istar + 0.5))
 
    #Lab-frame velocities
    xdot = vx * cos(theta) - vy * sin(theta)
    ydot = vx * sin(theta) + vy * cos(theta)
 
    du[1] = vx_dot
    du[2] = vy_dot
    du[3] = omega
    du[4] = omega_dot
    du[5] = xdot
    du[6] = ydot
end
 
function simulate(Istar; t_max=150.0, u0=nothing)
    if u0 === nothing
        W = sqrt(pi / (A + B))
        u0 = [0.05, -0.98 * W, 0.05, 0.05, 0.0, 0.0]
    end
    tspan = (0.0, t_max)
    prob = ODEProblem(falling_card!, u0, tspan, Istar)
    sol = solve(prob, Tsit5(); reltol=1e-10, abstol=1e-12, maxiters=1e7, dtmax=0.05)
    return sol
end
 
function make_fig3(; t_max=150.0, outfile="fig3_reproduction.png")
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
    println("saved $(outfile)")
    return fig
end
 
function make_theta_diagnostics(; t_max=150.0, outfile="theta_diagnostics.png")
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
    println("saved $(outfile)")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_fig3()
    make_theta_diagnostics()
end