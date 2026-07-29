using TASOPT, Printf
C = TASOPT.CryoTank; E = TASOPT.engine
open("/tmp/thermal_ref.csv","w") do io
    println(io,"fn,a,b,c,d,r1,r2,r3")
    for T in [200.0, 250.0, 288.2, 300.0, 400.0, 500.0]
        R, Pr, g, cp, mu, k = E.gasPr("air_simple", T)
        @printf(io,"gasPr,%.17g,0,0,0,%.17g,%.17g,%.17g\n", T, Pr, mu, k)
    end
    for z in [0.0, 3000.0, 11000.0], M in [0.0, 0.3, 0.8]
        h, Tair, Taw = C.freestream_heat_coeff(z, 288.2, M, 20.0, 250.0, 1.9)
        @printf(io,"fshc,%.17g,%.17g,20.0,250.0,%.17g,%.17g,%.17g\n", z, M, h, Tair, Taw)
    end
    for ifuel in [11, 40], Tw in [100.0, 200.0], lt in [5.0, 10.0]
        Tf = ifuel == 40 ? 20.4 : 111.5
        h = C.tank_heat_coeff(Tw, ifuel, Tf, lt)
        @printf(io,"thc,%d,%.17g,%.17g,%.17g,%.17g,0,0\n", ifuel, Tw, Tf, lt, h)
    end
    for Tc in [20.0, 100.0], Th in [250.0, 300.0], (Si,So) in [(40.0,45.0),(80.0,95.0)]
        R = C.vacuum_resistance(Tc, Th, Si, So)
        @printf(io,"vac,%.17g,%.17g,%.17g,%.17g,%.17g,0,0\n", Tc, Th, Si, So, R)
    end
end
println("ok")
