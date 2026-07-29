using TASOPT, Printf
C = TASOPT.CryoTank
open("/tmp/mixture_ref.csv","w") do io
    println(io,"species,p_atm,beta,x,T,rho,h,u,u_p,phi,rho_star,hvap")
    for sp in ["H2","CH4"], x_atm in [0.5,1.0,2.0,4.0], b in [0.5,0.8,0.95,0.99]
        p = x_atm * TASOPT.p_atm
        m = C.SaturatedMixture(sp, p, b)
        @printf(io,"%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                sp,x_atm,b,m.x,m.T,m.ρ,m.h,m.u,m.u_p,m.ϕ,m.ρ_star,m.hvap)
    end
end
open("/tmp/betaconv_ref.csv","w") do io
    println(io,"species,p_atm,p0_atm,beta0,beta")
    for sp in ["H2","CH4"], p0 in [1.0,2.0], p in [1.5,3.0], b0 in [0.9,0.95]
        b = C.convert_β_same_ρ(sp, p*TASOPT.p_atm, p0*TASOPT.p_atm, b0)
        @printf(io,"%s,%.17g,%.17g,%.17g,%.17g\n", sp,p,p0,b0,b)
    end
end
println("ok")
