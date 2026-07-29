using TASOPT, Printf
P = TASOPT.propsys.ElectricMachine
open("/tmp/electric_ref.csv","w") do io
    println(io,"kind,a,b,c,d,r1,r2,r3")
    for Pdes in [1.0e6, 5.0e5], fdes in [200.0, 400.0]
        inv = P.Inverter()
        P.size_inverter!(inv, Pdes, fdes)
        @printf(io,"invsize,%.17g,%.17g,0,0,%.17g,%.17g,%.17g\n",
                Pdes,fdes,inv.mass,inv.P_input,inv.P_input/inv.P)
        for frac in [0.1, 0.2, 0.5, 1.0], f in [200.0, 400.0]
            P.operate_inverter!(inv, frac*Pdes, f)
            @printf(io,"invop,%.17g,%.17g,%.17g,%.17g,%.17g,0,0\n",
                    Pdes,fdes,frac,f,inv.P/inv.P_input)
        end
    end
end
println("ok")
# Cable, from the live propsys module.
let C = TASOPT.propsys.ElectricMachine
    open("/tmp/cable_ref.csv","w") do io
        println(io,"P,V,l,mass,R,W")
        for P in [5.0e5, 2.0e6], V in [540.0, 1000.0, 3000.0], l in [10.0, 30.0]
            c = C.Cable()
            c(P, V, l)
            @printf(io,"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", P,V,l,c.mass,c.R,c.W)
        end
    end
end
println("cable ok")
