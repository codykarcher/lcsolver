using TASOPT, Printf, Roots
E = TASOPT.propsys.ElectricMachine
open("/tmp/motor_ref.csv","w") do io
    println(io,"kind,a,b,c,d,e,r1")
    for M in [8.0e5, 1.0e6], th in [0.01, 0.02, 0.03], gap in [0.002, 0.005]
        @printf(io,"airgap,%.17g,%.17g,%.17g,0,0,%.17g\n", M,th,gap, E.airgap_flux(M,th,gap))
    end
    for I in [100.0, 500.0], R in [1.0e-3, 5.0e-3], n in [2, 3]
        @printf(io,"ohmic,%.17g,%.17g,%d,0,0,%.17g\n", I,R,n, E.ohmic_loss(I,R,n))
    end
    st = E.ElectricSteel("M19")
    for m in [10.0, 40.0], f in [200.0, 800.0], B in [1.0, 1.5]
        # steel-only loss forms, exercised with the M19 constants
        h = m * st.kₕ * f * B^st.α
        e = m * st.kₑ * f^2 * B^2
        @printf(io,"hyst,%.17g,%.17g,%.17g,0,0,%.17g\n", m,f,B,h)
        @printf(io,"eddy,%.17g,%.17g,%.17g,0,0,%.17g\n", m,f,B,e)
    end
    for Om in [500.0, 1500.0], rg in [0.15, 0.25], gap in [0.002, 0.004]
        rho = 1.225; nu = 1.5e-5; l = 0.3
        Re = Om*rg*gap/nu
        res(Cf) = 1/sqrt(Cf) - 2.04 - 1.768*log(Re*sqrt(Cf))
        # Bracketed. find_zero(res, 1e-2) -- what the source uses -- throws
        # a DomainError for Re above about 30000; see DISCREPANCIES.md 67.
        Cf = find_zero(res, (1e-8, 1.0), Bisection())
        W = Cf*pi*rho*Om^3*rg^4*l
        @printf(io,"windage,%.17g,%.17g,%.17g,0,0,%.17g\n", Om,rg,gap,W)
    end
    # remanent_flux is not exercised: it reads magnet.remanent_flux,
    # magnet.α and magnet.Tbase, none of which exist on PermanentMagnet
    # (fields: thickness, ρ, M, mass). See DISCREPANCIES.md 68.
    @printf(io,"steel,%.17g,%.17g,%.17g,%.17g,0,0\n", st.kₕ, st.kₑ, st.α, st.ρ)
end
println("ok")
