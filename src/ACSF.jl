module ACSF

using JuLIP
using LinearAlgebra

export acsf, acsf_desc

#acsfPI = 3.14159265358979
acsfPI = 3.14159265359
unit_BOHR = 0.52917721067

function generate_sf_parameters(;n=5, cutoff=6.0, bohr=false):
    #Default values: cutoff = 6.0
    # n = 5 , number of intervals
    if bohr
        cutoff = cutoff/unit_BOHR
    end
    m = collect(0:n)
    #m = np.array(range(n+1), dtype=np.float32)
    n_pow = n.^(m./n)
    eta_m = (n_pow./cutoff).^2
    R_s = cutoff./n_pow
    eta_s = zeros(n+1)
    for mi=1:n+1
        eta_s[mi] = 1. / (R_s[n - mi - 1] - R_s[n - mi])^2
    end
    # !!!!!! FLIPED eta_s ARRAY HERE !!!! REVERSED ORDER NOT MENTIONED IN PAPER !!!!!!
    # ALSO R_s and eta_s shifted !!! NOT MENTIONED AGAIN !!!
    return eta_m, R_s[1:n+1], flip(eta_s[1:n],1)
end

function set_Behler2011()
    cutoff = 6.5
    Gpack = [
        [   9.,  100.,  200.,  350.,  600., 1000., 2000., 4000.], #G2_etas
        [   1.,    1.,    1.,    1.,   30.,   30.,   30.,   30.,  
           80.,   80.,   80.,   80.,  150.,  150.,  150.,  150., 
          150.,  150.,  150.,  150.,  250.,  250.,  250.,  250., 
          250.,  250.,  250.,  250.,  450.,  450.,  450.,  450., 
          450.,  450.,  450.,  450.,  800.,  800.,  800.,  800., 
          800.,  800.,  800.], #G4_etas
        [  -1.,    1.,   -1.,    1.,   -1.,    1.,   -1.,    1.,  
           -1.,    1.,   -1.,    1.,   -1.,    1.,   -1.,    1.,  
           -1.,    1.,   -1.,    1.,   -1.,    1.,   -1.,    1.,  
           -1.,    1.,   -1.,    1.,   -1.,    1.,   -1.,    1.,  
           -1.,    1.,   -1.,    1.,   -1.,    1.,   -1.,    1.,
           -1.,    1.,    1.], #G4_lambdas
        [   1.,    1.,    2.,    2.,    1.,    1.,    2.,    2.,
            1.,    1.,    2.,    2.,    1.,    1.,    2.,    2.,
            4.,    4.,   16.,   16.,    1.,    1.,    2.,    2.,   
            4.,    4.,   16.,   16.,    1.,    1.,    2.,    2.,
            4.,    4.,   16.,   16.,    1.,    1.,    2.,    2.,
            4.,    4.,   16.] #G4_zetas
    ]
    Gpack[1] ./= (10000.0 .* unit_BOHR.^2) 
    Gpack[2] ./= (10000.0 .* unit_BOHR.^2)
    return Dict("G2" => [cutoff,Gpack[1],[ 0.0 for g in Gpack[1]]], 
                "G4" => [cutoff,Gpack[2],Gpack[3],Gpack[4]])
end

function get_max_Rc(Gparams)    
    Rc = 0.0
    if "G2" in keys(Gparams)
        if Gparams["G2"][1] > Rc
            Rc = Gparams["G2"][1]
        end
    end
    if "G4" in keys(Gparams)
        if Gparams["G4"][1] > Rc
            Rc = Gparams["G4"][1]
        end
    end
    return Rc
end

function cutoff_func(dij, Rc)
    fc = zeros(size(dij))
    dij_in_Rc = dij .<= Rc
    fc[dij_in_Rc] = 0.5 .* ( cos.(dij[dij_in_Rc] .* acsfPI./Rc) .+ 1.0)
    return fc  
end

function G2(dj, Rc, eta; Rs=0.0)
    fc_ij = cutoff_func(dj, Rc)
    return 0.5 .* [sum(exp.(-eta[p] .* (dj .- Rs[p]).^2) .* fc_ij) for p=1:length(eta)]
end

#G2(dj, G) = G2(dj, G[1], G[2], Rs=G[3])

function G4(Rj, dj, Rc, eta, lambd, zeta)
    counts = collect(1:size(Rj,1))
    ### This part is very terrible in Julia. 
    # So far no other way to do this simpler.
    Rij = collect(Iterators.flatten([[Rj[j] for k in counts] for j in counts]))
    Rik = collect(Iterators.flatten([[Rj[k] for k in counts] for j in counts]))
    Dij = collect(Iterators.flatten([[dj[j] for k in counts] for j in counts]))
    Dik = collect(Iterators.flatten([[dj[k] for k in counts] for j in counts]))
    Djk = norm.(Rij .- Rik)
    ###
    cos_theta_ijk = [Rik[i,:]' * Rij[i,:] for i=1:size(Rij,1)] ./ (Dij .* Dik)
    exp_arg = Dij.^2 .+ Dik.^2 .+ Dik.^2
    fc_all = cutoff_func(Dij, Rc) .* cutoff_func(Dik, Rc) .* cutoff_func(Djk, Rc)
    jkmask = collect(Iterators.flatten([[k>=(j+1) for k in counts] for j in counts]))
    return [2.0.^(1.0 - zeta[p]) .* 
            sum( (1.0 .+ lambd[p] .* cos_theta_ijk[jkmask]).^zeta[p] .*
                 exp.(-eta[p] .* exp_arg[jkmask]) .* fc_all[jkmask]
               ) for p=1:length(eta)]
end

#G4(dj, G) = G4(dj, G[1], G[2], G[3], G[4])

function acsf_desc(Rs; Gparams=nothing)
    ds = norm.(Rs)
    descriptor = []
    if Gparams == nothing
        Gparams = set_Behler2011()
    end
    if "G2" in keys(Gparams)
        # Concatenate radial basis functions
        descriptor = vcat(descriptor,G2(ds, Gparams["G2"][1],
                                            Gparams["G2"][2],
                                            Rs=Gparams["G2"][3]))
    end
    if "G4" in keys(Gparams)
        # Concatenate angular basis functions
        descriptor = vcat(descriptor,G4(Rs, ds, Gparams["G4"][1],
                                                Gparams["G4"][2],
                                                Gparams["G4"][3],
                                                Gparams["G4"][4]))
    end
    return descriptor
end

function acsf(at; Gparams=nothing)
    if Gparams == nothing
        Gparams = set_Behler2011()
    end
    representation = []
    ni = []
    nj = []
    nR = []
    nd = []
    for (i, j, R) in pairs(neighbourlist(at, get_max_Rc(Gparams)))
        push!(ni, i)
        push!(nj, j)
        push!(nR, R)
        push!(nd, norm(R))
    end
    for i=1:length(at)
        Rs = nR[(ni .== i)]
        push!(representation,acsf_desc(Rs, Gparams=Gparams))
    end
    return representation
end

end # module
