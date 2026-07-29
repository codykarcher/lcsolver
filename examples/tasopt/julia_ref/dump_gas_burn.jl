using TASOPT, Printf
E = TASOPT.engine
alpha = [0.7532, 0.2315, 0.0006, 0.0020, 0.0127, 0.0]   # airfrac.inc, live line
n = 6
cases = []
for (ifuel, hvap) in [(24, 0.0), (24, 0.0), (40, 0.0), (40, 446000.0)]
    push!(cases, (ifuel, hvap))
end
open("/tmp/burn_ref.csv","w") do io
    println(io, "ifuel,hvap,to,tf,t,f,lam1,lam2,lam3,lam4,lam5,lam6")
    for (ifuel, hvap) in cases, to in [700.0, 900.0], tf in [280.0, 20.4], t in [1400.0, 1800.0]
        beta = zeros(n); beta[n] = 1.0
        gamma = E.gasfuel(ifuel, n)
        f, lam = E.gas_burn(alpha, beta, gamma, n, ifuel, to, tf, t, hvap)
        @printf(io, "%d,%.17g,%.17g,%.17g,%.17g,%.17g", ifuel, hvap, to, tf, t, f)
        for v in lam; @printf(io, ",%.17g", v); end
        println(io)
    end
end
println("ok")
