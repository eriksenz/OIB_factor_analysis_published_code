import numpy as np
import pandas as pd
from composition_stats import closure

# --- define helper functions ---

# define function to convert concentrations in wt% (majors) and ppm (traces) to cation mol%
def mol_pct(array):
    '''
    input:
        - array: a single major element composition  of shape (25,)
            - provided in wt% for major elements and ppm for trace elements
            - formatted as: ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Rb', 
                             'Sr', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th', 'U']
    '''
    # convert ppm to wt% for the last 17 elements (in place)
    array[-15:] = [trace/10000 for trace in array[-15:]]
    
    # "close" input df so it sums to 100%
    array = closure(array)*100

    # define conversion factors (molecular wts)
    wts = np.array([60.0962,   # SiO2
                    79.8722,   # TiO2
                    50.9684,   # (Al2O3)/2
                    71.8391,   # FeO
                    70.9220,   # MnO
                    40.3063,   # MgO
                    56.0852,   # CaO
                    30.9885,   # (Na2O)/2
                    47.1032,   # (K2O)/2
                    70.9723,   # (P2O5)/2
                    85.4678,   # Rb
                    87.62,     # Sr
                    91.224,    # Zr
                    92.9064,   # Nb
                    137.327,   # Ba
                    138.9055,  # La
                    140.116,   # Ce
                    144.242,   # Nd
                    150.36,    # Sm
                    173.045,   # Yb
                    174.9668,  # Lu
                    178.49,    # Hf
                    207.2,     # Pb
                    232.038,   # Th
                    238.029])  # U

    # divide each composition by the proper conversion factor
    g_per_100g = array/wts    

    # normalize and multiply by 100 to calculate cation mol%
    mol_pct = closure(g_per_100g)*100
    
    # output major element and trace element arrays in mol% (that together sum to 100%)
    return np.array(mol_pct)

# define function to convert cation mol% to wt%
def wt_pct(array):
    '''
    input:
        - array: a single composition of shape (25,)
            - provided in cation mol% for major elements AND trace elements
            - formatted as: ['SiO2', 'TiO2', 'AlO1.5', 'FeO', 'MnO', 'MgO', 'CaO', 'NaO0.5', 'KO0.5', 'PO2.5', 'Rb', 
                             'Sr', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th', 'U']
    '''
    # define conversion factors (molecular wts)
    wts = np.array([60.0962,   # SiO2
                    79.8722,   # TiO2
                    50.9684,   # (Al2O3)/2
                    71.8391,   # FeO
                    70.9220,   # MnO
                    40.3063,   # MgO
                    56.0852,   # CaO
                    30.9885,   # (Na2O)/2
                    47.1032,   # (K2O)/2
                    70.9723,   # (P2O5)/2
                    85.4678,   # Rb
                    87.62,     # Sr
                    91.224,    # Zr
                    92.9064,   # Nb
                    137.327,   # Ba
                    138.9055,  # La
                    140.116,   # Ce
                    144.242,   # Nd
                    150.36,    # Sm
                    173.045,   # Yb
                    174.9668,  # Lu
                    178.49,    # Hf
                    207.2,     # Pb
                    232.038,   # Th
                    238.029])  # U

    # multiply each composition by the proper conversion factor
    g_per_100g = array*wts    

    # normalize and multiply by 100 to calculate wt%
    wt_pct = closure(g_per_100g)*100
    
    # return array in wt%
    return wt_pct

# define function to calculate trace element composition of crystallizing mineral assemblage
def crystal_comp(x, D):
    '''
    input:
        - x: a single trace element composition (melt)
        - D: the bulk partition coefficient of the crystallizing assemblage
    '''
    Cs = D * x  # Concentration in the solid
        
    return Cs

# define function to calculate equilibrium olivine composition and total mineral assemblage composition (ol+cpx), 
# given a melt composition (cation mol%) and a defined ol/cpx ratio
def eq_ol_cpx(array, ol, cpx, step):
    '''
    input:
        - array: a single cation mol% array (shape (25,))
            - formatted as: ['SiO2', 'TiO2', 'AlO1.5', 'FeO', 'MnO', 'MgO', 'CaO', 'NaO0.5', 'KO0.5', 'PO2.5', 'Rb', 
                             'Sr', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th', 'U']
        - ol: amount of olivine (always 1)
        - cpx: amount of cpx (varies depending on slope)
        - step: size of each iterative step
    '''    
    # Major elements_________________________________________________________________________________________________________
    # define Mg and Fe2+
    Fe = array[3]
    Mg = array[5]
        
    # assume 90% of Fe is Fe2+ (10% is Fe3+)
    Fe2 = Fe*0.90
    
    # Roeder and Emslie (1970)
    Kd = 0.30 
    
    # calculate mol fraction of Fe/Mg in eq. olivine
    XFe_XMg = Kd*(Fe2/Mg) 
        
    # calculate olivine compositions (cation mol%) from ratio (multiply by 2/3 because Mg+Fe is 2/3 of the cations in olivine)
    ol_Si = 100*(1*(1/3))
    Ti = 0.0
    Al = 0.0
    ol_Fe = 100*(XFe_XMg/(XFe_XMg + 1)*(2/3))
    Mn = 0.0
    ol_Mg = 100*(1/(XFe_XMg + 1)*(2/3))
    ol_Ca = 0.0
    Na = 0.0
    K = 0.0
    P = 0.0
    
    # combine into array for olivine composition
    ol_comp = np.array([ol_Si, Ti, Al, ol_Fe, Mn, ol_Mg, ol_Ca, Na, K, P])
    
    # define constant cpx composition (cation mol%; Nisbet and Pearce, 1977)
    cpx_Si = 47.69
    cpx_Ti = 0.53
    cpx_Al = 3.37
    cpx_Fe = 6.02
    cpx_Mn = 0.16
    cpx_Mg = 22.69
    cpx_Ca = 19.17
    cpx_Na = 0.38
    
    # combine into array for cpx composition
    cpx_comp = np.array([cpx_Si, cpx_Ti, cpx_Al, cpx_Fe, cpx_Mn, cpx_Mg, cpx_Ca, cpx_Na, K, P])
    
    # proper cation weighting for each phase
    ol_cation_pct = 3/7
    cpx_cation_pct = 4/7
    
    # proper assemblage weighting for each phase
    ol_assemblage_pct = ol/(ol+cpx)
    cpx_assemblage_pct = 1 - ol_assemblage_pct
    
    # determine composition of crystal assemblage
    cryst_major_comp = ol_cation_pct*ol_assemblage_pct*ol_comp + cpx_cation_pct*cpx_assemblage_pct*cpx_comp
    
    # normalize major element composition of crystal assemblage to 100%
    cryst_major_comp = 100*closure(cryst_major_comp)
        
    # calculate eq. olivine Fo percentage
    Fo_pct = 100*(ol_Mg/(ol_Mg + ol_Fe))
    
    # Trace elements_________________________________________________________________________________________________________
    # define element variables
    Rb = array[10]
    Sr = array[11]
    Zr = array[12]
    Nb = array[13]
    Ba = array[14]
    La = array[15]
    Ce = array[16]
    Nd = array[17]
    Sm = array[18]
    Yb = array[19]
    Lu = array[20]
    Hf = array[21]
    Pb = array[22]
    Th = array[23]
    U = array[24]
    
    # define Kds for each element [ol, cpx]
    # olivine (Girnis, 2023)
    # cpx (Nielsen, R., Ustunisik, G. 2022. Clinopyroxene/melt partition coefficient experiments v.2 Interdisciplinary Earth Data Alliance (IEDA). http://doi.org/10.26022/IEDA/112325)
    # cpx (Hart and Dunn, 1993. Experimental cpx/melt partitioning of 24 trace elements)
    Rb_D = np.array([0.0037, 0.00018])   # NU (2022): 0.00018
    Sr_D = np.array([0.00081, 0.1283])    # NU (2022): 0.079;   HD (1993): 0.1283
    Zr_D = np.array([0.00067, 0.1234])    # NU (2022): 0.105;   HD (1993): 0.1234
    Nb_D = np.array([0.001, 0.0077])      # NU (2022): 0.003;   HD (1993): 0.0077
    Ba_D = np.array([4.60e-5, 0.00068])  # NU (2022): 0.00025; HD (1993): 0.00068
    La_D = np.array([7.30e-6, 0.0536])     # NU (2022): 0.03;    HD (1993): 0.0536
    Ce_D = np.array([1.60e-5, 0.0858])    # NU (2022): 0.054;   HD (1993): 0.0858
    Nd_D = np.array([7.30e-5, 0.1873])    # NU (2022): 0.112;   HD (1993): 0.1873
    Sm_D = np.array([0.0003, 0.291])     # NU (2022): 0.201;   HD (1993): 0.291
    Yb_D = np.array([0.016, 0.430])      # NU (2022): 0.255;   HD (1993): 0.430
    Lu_D = np.array([0.024, 0.433])     # NU (2022): 0.5242;  HD (1993): 0.433
    Hf_D = np.array([0.00075, 0.256])   # NU (2022): 0.2046;  HD (1993): 0.256
    Pb_D = np.array([0.0011, 0.072])    # NU (2022): 0.0142;  HD (1993): 0.072
    Th_D = np.array([1.50e-5, 0.0026])   # NU (2022): 0.0026
    U_D = np.array([9.70e-6, 0.0010])    # NU (2022): 0.0010
    
    # define list of element concentrations and Kds
    concs = [Rb, Sr, Zr, Nb, Ba, La, Ce, Nd, Sm, Yb, Lu, Hf, Pb, Th, U]
    Kds = [Rb_D, Sr_D, Zr_D, Nb_D, Ba_D, La_D, Ce_D, Nd_D, Sm_D, Yb_D, Lu_D, Hf_D, Pb_D, Th_D, U_D]

    # loop through trace element concentrations and Kds to calculate the composition of the solid
    cryst_trace_comp = []
    for c, kd in zip(concs, Kds):
        
        # define mineral D-values
        ol_D = kd[0]
        cpx_D = kd[1]
        
        # define ol_pct
        ol_pct = ol/(ol + cpx)
        
        # calculate bulk D
        bulk_D = ol_pct*ol_D + (1 - ol_pct)*cpx_D
        
        # execute function with bulk D
        conc = crystal_comp(c, bulk_D)       # updated version
        
        # append each trace element composition
        cryst_trace_comp.append(conc)
        
    # combine major and trace element compositions into single list
    cryst_total_comp = np.concatenate((cryst_major_comp, cryst_trace_comp))
    
    # normalize major+trace element composition of crystal assemblage to 100%
    cryst_total_comp = 100*closure(cryst_total_comp)

    # return olivine compositions
    return (Fo_pct, np.array(cryst_total_comp))
    
    '''
    returns:
    - 1) Fo_pct: Forserite content in pct
    - 2) cryst_total_comp: An array of shape (25,) containing the major+trace element composition of the mineral assemblage that must be added to or subtracted from the melt
    '''

# --- fractional crystallization correction ---

# define function to correct for fractional crystallization of ol+cpx
def crystal_fract_corr(df, ol, cpx):
    '''
    input:
        - df: dataframe for a given island (wt%)
        - ol: amount of olivine (always 1)
        - cpx: amount of cpx (relative to ol=1)
    '''
    # convert df to array for calculation
    island_array = np.array(df[['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 
                                'Na2O','K2O', 'P2O5', 'Rb', 'Sr', 'Zr', 'Nb', 'Ba', 
                                'La', 'Ce', 'Nd', 'Sm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th', 'U']])
    
    # Define target forsterite content (mantle olivine)
    target_Fo = 90

    # Define tolerance for convergence
    tolerance = target_Fo * 0.0005  # within 0.01% of Fo90
    
    # define step size (0.05%)
    step = 0.0005

    # Initialize lists to record melt compositions and mass added/removed
    all_melt_comps = []
    ol_mass_added = []
    cpx_mass_added = []

    # Define olivine and cpx proportions
    ol_pct = ol / (ol + cpx)

    # Iterate through each sample
    for sample in island_array:
        # Convert to cation mol% and initiate starting melt composition
        melt_comp = mol_pct(sample)

        # Initialize variables
        n = 0
        melt_mass = 1.0
        cumulative_crystal_mass = 0.0
        melt_comps = [melt_comp]
        mass_pct = [0]

        # Calculate initial Fo content and crystallizing assemblage
        Fo_content, cryst_comp = eq_ol_cpx(melt_comp, ol, cpx, step)

        # Normalize crystal composition
        cryst_comp = closure(cryst_comp) * 100

        # Adjust melt composition based on Fo content
        if Fo_content > target_Fo:
            # Subtraction case
            while Fo_content > target_Fo and abs(Fo_content - target_Fo) > tolerance:
                n += 1
                delta_crystal_mass = step * melt_mass
                cumulative_crystal_mass += delta_crystal_mass
                # Adjust melt composition before updating melt mass
                melt_comp = (melt_mass * melt_comp - delta_crystal_mass * cryst_comp) / (melt_mass - delta_crystal_mass)
                # Update melt mass
                melt_mass -= delta_crystal_mass
                melt_comp = closure(melt_comp) * 100
                melt_comps.append(melt_comp)
                mass_pct.append(-delta_crystal_mass)
                Fo_content, cryst_comp = eq_ol_cpx(melt_comp, ol, cpx, delta_crystal_mass / melt_mass)
        else:
            # Addition case
            while Fo_content < target_Fo and abs(Fo_content - target_Fo) > tolerance:
                n += 1
                delta_crystal_mass = step * melt_mass
                cumulative_crystal_mass += delta_crystal_mass
                # Adjust melt composition before updating melt mass
                melt_comp = (melt_mass * melt_comp + delta_crystal_mass * cryst_comp) / (melt_mass + delta_crystal_mass)
                # Update melt mass
                melt_mass += delta_crystal_mass
                melt_comp = closure(melt_comp) * 100
                melt_comps.append(melt_comp)
                mass_pct.append(delta_crystal_mass)
                Fo_content, cryst_comp = eq_ol_cpx(melt_comp, ol, cpx, delta_crystal_mass / melt_mass)

        # Record final compositions and mass percentages
        total_crystal_mass = cumulative_crystal_mass
        ol_mass = (total_crystal_mass * ol_pct) * 100
        cpx_mass = (total_crystal_mass * (1 - ol_pct)) * 100

        if Fo_content > target_Fo:
            ol_mass_added.append(-ol_mass)
            cpx_mass_added.append(-cpx_mass)
        else:
            ol_mass_added.append(ol_mass)
            cpx_mass_added.append(cpx_mass)

        all_melt_comps.append(np.array(melt_comps))

    # pull final corrected composition and convert to wt%
    corr_comp_array = np.array([wt_pct(comp[-1]) for comp in all_melt_comps])
    
    # convert trace element data from wt% to ppm
    corr_comp_array[:, -15:] *= 10000
    
    # reshape ol and cpx arrays for easier concatenating
    ol_array = np.array(ol_mass_added).reshape(-1, 1)
    cpx_array = np.array(cpx_mass_added).reshape(-1, 1)
    
    # concatenate corr_comp_array, ol_array, and cpx_array into a single array
    result_array = np.concatenate((corr_comp_array, ol_array, cpx_array),axis=1)
    
    # convert to dataframe
    result_df = pd.DataFrame(result_array,
                             columns = ['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O','P2O5', 'Rb', 'Sr', 'Zr', 'Nb', 'Ba',
                                        'La', 'Ce', 'Nd', 'Sm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th', 'U', 'ol added (%)','cpx added (%)'],
                             index = df.index)
    
    # pull location/sample data from original dataframe
    loc_df = df[['Group', 'Island', 'Ocean', 'Sample ID', 'Latitude', 'Longitude']]
    
    # combine dataframes into final output df
    output_df = pd.concat([loc_df, result_df],axis=1)
    
    # return output df (wt%) and all_melt_comps array (mol%)
    return (output_df, all_melt_comps)
    '''
    returns:
    1) output_df: dataframe containing the following:
        - Final, near-primitive melt composition (major elements in wt%; trace elements in ppm) corrected for ol+cpx fractionation/accumulation (together summing to 100%)
        - Mass of olivine and/or cpx added to the melt (negative values indicate removal of crystals from the melt) to reach equilibrium with mantle olivine (Fo72)
            - This mass is relative to the initial mass of the system
    2) all_melt_comps: a list of arrays, where:
        - The length of the list is equal to the number of samples provided in the input dataframe, and 
        - The shape of each array is (n ,25), where n is the number of iterations needed for convergence and 25 is the number of elements
    '''