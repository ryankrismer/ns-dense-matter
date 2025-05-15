import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as font_manager
import os
import bilby
import scipy.interpolate as interpolate
import h5py
import scipy.stats as sst 

### Units
gcm3_to_dynecm2=8.9875e20 
MeV_to_gcm3 = 1.7827e12 
dynecm2_to_MeVFermi = 1.6022e33
gcm3_to_fm4 = 3.5178e14
sat_dens = 2.8*(np.power(10.0,14.))
c = bilby.core.utils.speed_of_light
c_cgs = c*(1e2)

### Nucleon mass in grams
mass_of_nucleon = 1/(6.02 * 10**23)
### Nucleon mass in MeV/c2
nucleon_mass_in_MeV = 939
### Conversion factors for Mev,fm to CGS units
gcm3_to_dynecm2=8.9875e20 
MeV_to_gcm3 = 1.7827e12 
dynecm2_to_MeVFermi = 1.6022e33
gcm3_to_fm4 = 3.5178e14

fm_in_cgs = 1e-13
rho_nuc_in_cgs = .16 * mass_of_nucleon / (fm_in_cgs)**3

def find_pts(data, idx, threshold = 0.0):
    """
    Function to find phase transitions within a given EoS set. 
    Threshold acts as minimum difference in the sound speed to qualify as a phase transition. 
    """
    cs = np.gradient(data[idx]["pressurec2"], data[idx]["energy_densityc2"])
    num_neg = np.where(np.diff(cs[data[idx]["baryon_density"] > rho_nuc_in_cgs]) < -threshold)[0]
    num_pts = len(np.where(np.diff(num_neg) > 1.0)[0])
    return num_pts

def plot_eoss(data_with_key, x_key, y_key, alpha, clr,
              axis = None, loglog = False,
              semilogx = False, semilogy = False):
    if semilogx:
        plt.semilogx()
    if semilogy:
        plt.semilogy()
    if loglog:
        plt.loglog()
    if axis:
        plt.axis([axis[0], axis[1], axis[2], axis[3]])
        
    plt.plot(data_with_key[f"{x_key}"], data_with_key[f"{y_key}"], alpha = alpha, color = clr)
    plt.gcf().set_dpi(500)
    plt.grid(alpha = 0.2)
    plt.minorticks_on()
    plt.show()
    
def phi(eps, press):
    try:
        phi = np.log((np.gradient(eps, press) - 1.0))
    except:
        print(f"Error at {eps}.")
    return phi

def trace_anom(pressure, density): ### Trace Anomaly calculation
    trace_anomaly = (1./3.) - (pressure/density)
    return trace_anomaly

def chem_potent(p, epsilon, baryon): ### Chemical Potential Calculation, defined via Enthalpy at zero temperature
    mu = ((p + epsilon)/baryon)
    return mu

def mm_eos_to_cgs(mm_eos):
    mm_eos_cgs = {"baryon_density": np.array(mm_eos["number_density"]) * mass_of_nucleon/fm_in_cgs**3,
                  "energy_densityc2":np.array(mm_eos["energy_densityc2"]),
                  "pressurec2" :np.array(mm_eos["pressure_nuclear"]) / nucleon_mass_in_MeV * mass_of_nucleon / fm_in_cgs**3}
    return mm_eos_cgs

def get_and_load_weights(weights, astro_tag, eos_set_to_use, Neff = False):
    """
    Returns a collection of downsampled EoS according to their likelihood from some astrophysical observation.
    Assumes correlation and matched weights with respective EoS set via "eos_to_be_used".
    """
    ### sample EoS's according to astro weights
    try:
        # Resampling EoS's with weighted values according to likelihoods
        astro_exp_weights = np.exp(weights[astro_tag])
        # Normalize values to have probabilities add up to 1
        astro_exp_weights = astro_exp_weights/(sum(astro_exp_weights))
        # Resample EoS's according to weights --> gives posterior distribution of EoS's
        astro_weight_eos = np.random.choice(eos_set_to_use, size=len(eos_set_to_use), replace=True, p=astro_exp_weights)
        
        if Neff:
            Neff = ((sum(astro_exp_weights))**2)/(sum(astro_exp_weights**2))
            print(f"Number of effective EoS's: {int(Neff)}")
            
        return astro_weight_eos

    except ValueError:
        print("All EoS's are unlikely.")
        
def get_pe_quantiles(eos_set, eos_data, interp_densities, quantiles, cgs_press = True):
    pdvals = []
    # Obtain quantiles for the prior set of EoS's
    for eos in eos_set:
        eos_densities = eos_data[eos]["baryon_density"]
        eos_pressures = eos_data[eos]["pressurec2"]
        if cgs_press:
            eos_pressures = eos_data[eos]["pressurec2"]*gcm3_to_dynecm2
        pdvals.append(np.interp(interp_densities, eos_densities, eos_pressures))
    pdvals = np.array(pdvals)
    
    # Create object to hold upper and lower bounds 
    pd_sigmas = np.zeros((len(interp_densities),len(quantiles)))
    for i in range(len(interp_densities)):
        pd_sigmas[i]=np.percentile(np.array(pdvals[:,i]),quantiles)
    del pdvals
    return pd_sigmas

def get_cs_quantiles(eos_set, eos_data, interp_densities, quantiles):
    csvals = []
    # Obtain sound speed quantiles for a set of EoS's
    for eos in eos_set:
        eos_rest_mass = eos_data[eos]["baryon_density"]
        eos_densities = eos_data[eos]["energy_densityc2"]
        eos_pressures = eos_data[eos]["pressurec2"]
        eos_cs = np.gradient(eos_pressures, eos_densities)
        csvals.append(np.interp(interp_densities, eos_rest_mass, eos_cs))
    csvals = np.array(csvals)

    ### Obtaining quantiles 
    cs_sigmas = np.zeros((len(interp_densities),len(quantiles)))
    for i in range(len(interp_densities)):
        cs_sigmas[i] = np.percentile(np.array(csvals[:,i]), quantiles)
    del csvals
    return cs_sigmas

def get_mr_quantiles(eos_set, eos_mac_data, interp_masses, quantiles, cutoff = 30., verbose =  False):
    mrvals = []
    num_eos_cut = 0
    for eos in eos_set:
        eos_mass = eos_mac_data[eos]["M"][eos_mac_data[eos]["R"] < cutoff] # cuts to ignore radii values at
        eos_radii = eos_mac_data[eos]["R"][eos_mac_data[eos]["R"] < cutoff] # larger mass from white dwarf branch 
        if len(eos_radii) == 0:
            num_eos_cut += 1
            continue
        ### Try interpolating with scipy function
        massrad_func = interpolate.interp1d(eos_mass, eos_radii, bounds_error = False, fill_value = "NaN")
        interp_rad = massrad_func(interp_masses)
        mrvals.append(interp_rad)
    mrvals = np.array(mrvals)
    
    ## Obtaining quantiles 
    mr_sigmas = np.zeros((len(interp_masses),len(quantiles)))

    ### Expecting NaN's from Eos's that don't reach end mass from interpolation range
    for i in range(len(interp_masses)):
        mr_sigmas[i]=np.nanpercentile(mrvals[:,i],quantiles)

    if verbose:
        print("Number of surviving EoS after radius cuts: ", str(len(eos_set) - num_eos_cut))
    
    del mrvals
    return mr_sigmas