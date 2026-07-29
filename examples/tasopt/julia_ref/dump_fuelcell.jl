using TASOPT, Printf
E = TASOPT.engine
open("/tmp/fuelcell_ref.csv","w") do io
    println(io,"kind,a,b,c,d,r1")
    for j in [500.0, 2000.0, 8000.0, 15000.0], T in [340.0, 360.0, 400.0],
        pH2 in [1.5e5, 3.0e5], pair in [1.0e5, 2.5e5]
        V = E.LT_PEMFC_voltage_simple(j, T, pH2, pair)
        @printf(io,"volt,%.17g,%.17g,%.17g,%.17g,%.17g\n", j,T,pH2,pair,V)
    end
    for T in [280.0, 320.0, 350.0, 373.0, 380.0, 420.0]
        @printf(io,"psat,%.17g,0,0,0,%.17g\n", T, E.water_sat_pressure(T))
    end
    for T in [320.0, 350.0], lam in [2.0, 7.0, 14.0, 20.0]
        @printf(io,"nafion,%.17g,%.17g,0,0,%.17g\n", T,lam, E.conductivity_Nafion(T,lam))
    end
    for T in [400.0, 430.0, 460.0], DL in [6.0, 10.0], RH in [0.0, 0.5]
        @printf(io,"pbi,%.17g,%.17g,%.17g,0,%.17g\n", T,DL,RH, E.conductivity_PBI(T,DL,RH))
    end
    for a in [0.0, 0.3, 0.7, 1.0, 1.5, 3.0]
        @printf(io,"lam,%.17g,0,0,0,%.17g\n", a, E.λ_calc(a))
    end
    # Stack weight, written out with the same constants the source uses --
    # the LT_PEMFC input struct is not exported from TASOPT.engine.
    for n in [400.0, 800.0], A in [0.05, 0.2], fo in [0.1, 0.3]
        tM = 2.5e-5; tA = 3e-4; tC = 3e-4
        rho_e = 1.9e3; rho_M = 1970.0
        W = 9.81 * n * A * ((tA + tC)*rho_e + tM*rho_M) * (1 + fo)
        @printf(io,"weight,%.17g,%.17g,%.17g,0,%.17g\n", n,A,fo,W)
    end
end
println("ok")
