# Falling Card Simulation

Attempted reproduction of the quasi-steady ODE model for a rigid card falling through air from from **Andersen, Pesavento & Wang (2005), "Analysis of transitions between fluttering, tumbling and steady descent of falling cards,"** *J. Fluid Mech.* 541:91–104.

## Usage

```julia
julia> include("Fallingcard.jl")
julia> run_all()
```

call individual functions:

```julia
julia> simulate(1.4)              # returns an ODE solution for a given I*
julia> make_fig3()                # trajectories
julia> check_lyapunov()           # Lyapunov exponent (I*=2.2)
julia> make_fig4b()               # I*-mu1 phase diagram
julia> make_fig5d()               # bifurcation divergence 
```

Fig. 5(d): couldn't get this one to work. The figure graphs how long the card lingers near the edge-on fixed point before tipping into flutter/tumble, which should blow up as I* → I*_c. Instead it plateaus

My guess: I'm only measuring escape from the first fixed point, whose escape rate barely depends on I*. The divergence probably comes from how closely the trajectory passes the second fixed point, which I've not implemented.
