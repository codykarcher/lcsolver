using TASOPT, Printf
E = TASOPT.engine
open("/tmp/hx_ref.csv","w") do io
    println(io,"kind,a,b,c,d,e,f,g,h,i,r1,r2")
    for Re in [1e3, 1e4, 1e5, 1e6]
        j, Cf = E.jcalc_pipe(Re)
        @printf(io,"jpipe,%.17g,0,0,0,0,0,0,0,0,%.17g,%.17g\n", Re, j, Cf)
    end
    for Re in [20.0, 100.0, 500.0, 5000.0, 1e5, 5e5], Pr in [0.7, 5.0],
        NL in [4.0, 12.0], xt in [1.5, 2.5], xl in [1.0, 1.5]
        Nu = E.Nu_calc_staggered_cyl(Re, Pr, NL, xt, xl)
        @printf(io,"nu,%.17g,%.17g,%.17g,%.17g,%.17g,0,0,0,0,%.17g,0\n", Re,Pr,NL,xt,xl,Nu)
    end
    for Re in [100.0, 500.0, 5000.0], G in [50.0, 200.0], L in [0.2, 0.5]
        rho=1.2; Dv=0.01; tDo=0.006; xt=1.5; xl=1.25; mur=1.0
        dp = E.Δp_calc_staggered_cyl(Re,G,L,rho,Dv,tDo,xt,xl,mur)
        @printf(io,"dp,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,0\n",
                Re,G,L,rho,Dv,tDo,xt,xl,mur,dp)
    end
end
println("ok")
