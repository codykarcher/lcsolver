using TASOPT, Printf
Ts = vcat([21.0, 30.0, 60.0, 111.3, 187.5, 260.0, 312.5, 401.0, 512.5, 637.5,
           763.0, 888.0, 963.0, 1100.0, 1300.0, 1500.0, 1700.0, 1790.0],
          [15.0, 1900.0, 2200.0],          # outside the table -- extrapolated
          [20.369, 300.0, 1000.0, 1800.0]) # exactly on knots
open("/tmp/h2_ref.csv","w") do io
    println(io, "t,s,s_t,h,h_t,cp,r")
    for T in Ts
        s, s_t, h, h_t, cp, r = TASOPT.engine.gasfun(40, T)
        @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                T, s, s_t, h, h_t, cp, r)
    end
end
println("wrote ", length(Ts), " points")
