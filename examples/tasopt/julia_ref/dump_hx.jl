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


const OUT = "/Users/codykarcher/Dropbox/research/edi/examples/tasopt/julia_ref"
E = TASOPT.engine

open(joinpath(OUT,"hx_gaspr.csv"),"w") do io
  println(io,"gas,T,R,Pr,gamma,cp,mu,k")
  for g in ["air","air_simple","co2","n2","o2","h2o","ch4","h2"],
      T in [250.0, 400.0, 500.0, 778.0, 1200.0]
    R,Pr,gam,cp,mu,k = E.gasPr(g,T)
    println(io, join([g,T,R,Pr,gam,cp,mu,k], ","))
  end
end

open(joinpath(OUT,"hx_tset_single.csv"),"w") do io
  println(io,"igas,hspec,tguess,T")
  for (ig,T0) in [(40,300.0),(1,400.0),(2,500.0),(4,600.0)]
    s,st,h,ht,cp,r = E.gasfun(ig,T0)
    for dh in [-1e5, 0.0, 3e5, 1e6]
      println(io, join([ig, h+dh, T0, E.gas_tset_single(ig,h+dh,T0)], ","))
    end
  end
end

# hxoptim! reference design vector
HXgas = E.HX_gas(); HXgeom = E.HX_tubular()
HXgas.fluid_p="air"; HXgas.fluid_c="h2"
HXgas.mdot_p=1144/60; HXgas.mdot_c=9.95/60; HXgas.ε=0.8
HXgas.Tp_in=778; HXgas.Tc_in=264; HXgas.Mp_in=0.19
HXgas.pp_in=40e3; HXgas.pc_in=1515e3
HXgas.alpha_p=[0.7532,0.2315,0.0006,0.0020,0.0127]; HXgas.igas_c=40
HXgeom.is_concentric=true; HXgeom.has_recirculation=false
HXgeom.D_i=0.564; HXgeom.l=0.6084530646014857; HXgeom.xl_D=1
HXgeom.Rfp=0.01*0.1761; HXgeom.Rfc=8.815E-05
HXgeom.material=TASOPT.StructuralAlloy("SS-304"); HXgeom.Δpdes=3e6; HXgeom.maxL=0.5

_,_,_,_,cp_p,Rp = E.gassum(HXgas.alpha_p,5,HXgas.Tp_in)
γ = cp_p/(cp_p-Rp); ρ = HXgas.pp_in/(Rp*HXgas.Tp_in)
V = HXgas.Mp_in*sqrt(γ*Rp*HXgas.Tp_in); A_cs = HXgas.mdot_p/(ρ*V)
D_o = sqrt(4*(A_cs+pi*HXgeom.D_i^2/4)/pi); linit = 1.1*(D_o-HXgeom.D_i)/2
E.hxoptim!(HXgas,HXgeom,[3.0,4.0,4.0,linit])
E.hxsize!(HXgas,HXgeom)
open(joinpath(OUT,"hx_optim.csv"),"w") do io
  println(io,"var,value")
  for (k,v) in [("Mc_in",HXgas.Mc_in),("n_stages",HXgeom.n_stages),
                ("xt_D",HXgeom.xt_D),("l",HXgeom.l),
                ("Pl_p",HXgas.Pl_p),("Pl_c",HXgas.Pl_c),
                ("n_passes",HXgeom.n_passes),("N_t",HXgeom.N_t),
                ("Dp_p",HXgas.Δp_p),("Dp_c",HXgas.Δp_c)]
    println(io,"$k,$v")
  end
end
println("done")
