using TASOPT, Printf
# The Raymer correlations, exercised directly with the same unit factors the
# reference uses, so the reference values are the reference's arithmetic.
open("/tmp/lg_ref.csv","w") do io
    println(io,"WMTO,lgn,lgm,nwn,nwm,nsm,Vstall,Wn,Wm")
    lf = 4.5
    for WMTO in [7.8e5, 1.2e6], lgn in [1.8, 2.6], lgm in [2.0, 3.0],
        (nwn,nwm,nsm) in [(2,4,2),(2,8,4)], Vstall in [60.0, 75.0]
        Wm = 0.0106 * (WMTO * TASOPT.lb_N)^0.888 * lf^0.25 * (lgm / TASOPT.in_to_m)^0.4 *
             nwm^0.321 * nsm^(-0.5) * (Vstall / TASOPT.kts_to_mps)^0.1 / TASOPT.lb_N
        Wn = 0.032 * (WMTO * TASOPT.lb_N)^0.646 * lf^0.2 * (lgn / TASOPT.in_to_m)^0.5 *
             nwn^0.45 / TASOPT.lb_N
        @printf(io,"%.17g,%.17g,%.17g,%d,%d,%d,%.17g,%.17g,%.17g\n",
                WMTO,lgn,lgm,nwn,nwm,nsm,Vstall,Wn,Wm)
    end
end
@printf("lb_N=%.17g in_to_m=%.17g kts_to_mps=%.17g\n", TASOPT.lb_N, TASOPT.in_to_m, TASOPT.kts_to_mps)
println("ok")
