using TASOPT, Printf
E = TASOPT.engine
open("/tmp/dfan_ref.csv","w") do io
    println(io,"Dfan,Nmech,neng,rSnace,fpylon,Weng,Wnac,Webare,Snace1")
    for Dfan in [1.4, 2.0, 2.6], Nmech in [4000.0, 8000.0], neng in [2.0, 4.0],
        rSnace in [12.0, 16.0], fpylon in [0.1, 0.15]
        ARfan = 3.0; bs = 0.4; ktech = 0.5
        Utip = Dfan/2*(2*pi*Nmech/60)
        mfan = ktech*(135.0*Dfan^2.7/sqrt(ARfan)*(bs/1.25)^0.3*(Utip/350.0)^0.3)
        Snace1 = rSnace*0.25*pi*Dfan^2
        Ainlet = 0.4*Snace1; Acowl = 0.2*Snace1; Aexh = 0.4*Snace1
        Wnace = 4.45*(Ainlet/0.3048^2.0)*(2.5+0.0238*Dfan/0.0254) +
                4.45*(Acowl/0.3048^2.0)*1.9 +
                4.45*(Aexh/0.3048^2.0)*(2.5+0.0363*Dfan/0.0254)
        Wfan = (mfan*9.81 + Wnace*0.8)*(1+fpylon)
        Weng = (Wfan + Wnace)*neng
        Webare = Wfan*neng
        Wnac = Wnace*neng
        @printf(io,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                Dfan,Nmech,neng,rSnace,fpylon,Weng,Wnac,Webare,Snace1)
    end
end
println("ok")
