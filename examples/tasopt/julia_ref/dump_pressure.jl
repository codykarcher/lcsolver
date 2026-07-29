using TASOPT, Printf
C = TASOPT.CryoTank
open("/tmp/pressure_ref.csv","w") do io
    println(io,"species,p_atm,beta,Q,W,mdot,xout,mvent,xvent,V,alpha,dpdt,dbdt,mvent_req,mboil")
    for sp in ["H2","CH4"], x_atm in [1.0,2.0], b in [0.9,0.95], Q in [500.0,2600.0],
        mdot in [0.0,0.05], xout in [0.0,1.0]
        p = x_atm*TASOPT.p_atm; V = 50.0; W = 0.0; mvent = 0.01; xvent = 1.0; al = 1.0
        m = C.SaturatedMixture(sp, p, b)
        dp = C.dpdt(m, Q, W, mdot, xout, mvent, xvent, V, al)
        db = C.dβdt(m, dp, mdot + mvent, V)
        mv = C.venting_mass_flow(m, Q, W, mdot, xout, xvent)
        mb = C.mdot_boiloff(m, db, dp, mdot*(1-xout), V)
        @printf(io,"%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                sp,x_atm,b,Q,W,mdot,xout,mvent,xvent,V,al,dp,db,mv,mb)
    end
end
println("ok")
