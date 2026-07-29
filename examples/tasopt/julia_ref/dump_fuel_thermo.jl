using TASOPT, Printf
C = TASOPT.CryoTank
ps = vcat([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
open("/tmp/fuel_thermo_ref.csv","w") do io
    println(io, "kind,species,p_atm,Tsat,rho,rho_p,h,u,u_p")
    for species in ["H2","CH4"], (kind, f) in [("gas", C.gas_properties), ("liquid", C.liquid_properties)]
        for x in ps
            p = x * TASOPT.p_atm
            r = f(species, p)
            @printf(io, "%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    kind, species, x, r.Tsat, r.ρ, r.ρ_p, r.h, r.u, r.u_p)
        end
    end
end
println("ok")
