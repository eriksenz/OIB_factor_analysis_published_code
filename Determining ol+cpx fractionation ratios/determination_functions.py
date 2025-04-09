import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from composition_stats import closure

# --- define helper functions ---

# define function to convert wt% oxides to cation mol%
def cation_mol_pct(array):
    '''
    input:
        - array: a single major element composition (shape (,10))
            - provided in wt%
            - formatted as: ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O','P2O5']
    '''
    # "close" input df so it sums to 100%
    array = closure(array)*100

    # define conversion factors (molecular wts)
    wts = np.array([60.0962,  # SiO2
                    79.8722,  # TiO2
                    50.9684,  # (Al2O3)/2
                    71.8391,  # FeO
                    70.9220,  # MnO
                    40.3063,  # MgO
                    56.0852,  # CaO
                    30.9885,  # (Na2O)/2
                    47.1032,  # (K2O)/2
                    70.9723])  # (P2O5)/2

    # divide each composition by the proper conversion factor
    g_per_100g = array/wts    

    # normalize and multiply by 100 to calculate cation mol%
    mol_pct = 100*(g_per_100g/sum(g_per_100g))
    
    return mol_pct

# define function that converts cation mol% to wt% oxides
def wt_pct(array):
    '''
    input:
        - array: a single major element composition (shape (,10))
            - provided in cation mol%
            - formatted as: ['SiO2', 'TiO2', 'AlO1.5', 'FeO', 'MnO', 'MgO', 'CaO', 'NaO0.5', 'KO0.5','PO2.5']
    '''
    # define conversion factors (molecular wts)
    wts = np.array([60.0962,  # SiO2
                    79.8722,  # TiO2
                    50.9684,  # (Al2O3)/2
                    71.8391,  # FeO
                    70.9220,  # MnO
                    40.3063,  # MgO
                    56.0852,  # CaO
                    30.9885,  # (Na2O)/2
                    47.1032,  # (K2O)/2
                    70.9723])  # (P2O5)/2
    
    # multiply each composition by the proper conversion factor
    g_per_100g = array*wts     
    
    wt_pct = 100*(g_per_100g/sum(g_per_100g))
    
    return wt_pct

# define function to calculate equilibrium olivine composition and total mineral assemblage composition (ol+cpx), 
# given a melt composition (cation mol%) and a defined ol/cpx ratio
def eq_ol_cpx(array, ol, cpx):
    '''
    input:
        - array: a single cation mol% array (shape (,10))
            - formatted as: ['SiO2', 'TiO2', 'AlO1.5', 'FeO', 'MnO', 'MgO', 'CaO', 'NaO0.5', 'KO0.5','PO2.5']
        - ol: amount of olivine (always 1)
        - cpx: amount of cpx (varies depending on slope)
    '''
    # Roeder and Emslie (1970)
    Kd = 0.30 
        
    # Define Mg and Fe2+
    Mg = array[5]
    Fe = array[3]
        
    # Assume 90% of Fe is Fe2+ (10% is Fe3+)
    Fe2 = Fe * 0.90
        
    # Calculate mol fraction of Fe/Mg in eq. olivine
    XFe_XMg = Kd * (Fe2 / Mg) 
        
    # Calculate olivine compositions (cation mol%)
    # Multiply by 2/3 because Mg + Fe is 2/3 of the cations in olivine
    ol_Si = 100 * (1 * (1 / 3))
    ol_Ti = 0.0
    ol_Al = 0.0
    ol_Fe = 100 * (XFe_XMg / (XFe_XMg + 1) * (2 / 3))
    ol_Mn = 0.0
    ol_Mg = 100 * (1 / (XFe_XMg + 1) * (2 / 3))
    ol_Ca = 0.0
    ol_Na = 0.0
    ol_K = 0.0
    ol_P = 0.0
    
    # Combine into array for olivine composition
    ol_comp = np.array([ol_Si, ol_Ti, ol_Al, ol_Fe, ol_Mn, ol_Mg, ol_Ca, ol_Na, ol_K, ol_P])
    
    # Define constant cpx composition (cation mol%; Nisbet and Pearce, 1977)
    cpx_Si = 51.0
    cpx_Ti = 0.0
    cpx_Al = 0.0
    cpx_Fe = 6.0
    cpx_Mn = 0.0
    cpx_Mg = 23.0
    cpx_Ca = 20.0
    cpx_Na = 0.0
    cpx_K = 0.0
    cpx_P = 0.0
    
    # Combine into array for cpx composition
    cpx_comp = np.array([cpx_Si, cpx_Ti, cpx_Al, cpx_Fe, cpx_Mn, cpx_Mg, cpx_Ca, cpx_Na, cpx_K, cpx_P])
    
    # Proper cation weighting for each phase
    ol_cation_pct = 3 / 7
    cpx_cation_pct = 4 / 7
    
    # Assemblage weighting for each phase
    ol_assemblage_pct = ol / (ol + cpx)
    cpx_assemblage_pct = 1 - ol_assemblage_pct
    
    # Determine composition of crystal assemblage
    cryst_comp = (
        ol_cation_pct * ol_assemblage_pct * ol_comp +
        cpx_cation_pct * cpx_assemblage_pct * cpx_comp
    )
    
    # Normalize composition of crystal assemblage to 100%
    cryst_comp = closure(cryst_comp) * 100
        
    # Calculate equilibrium olivine Fo percentage
    Fo_pct = 100 * (ol_Mg / (ol_Mg + ol_Fe))
        
    # Return Fo_pct and cryst_comp
    return Fo_pct, cryst_comp

# --- outlier filtering ---

# define function to filter clear outliers
def outlier_filt(df):
    '''
    input:
        - df: dataframe for a given island group (e.g., hawaii = OIB_df[OIB_df['Group']=='Hawaii'])
    '''
    # define list of major elements
    major_elements = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O','P2O5']

    islands = df['Island'].unique()
    
    # define list to append filtered dfs to
    filtered_df_list = []
    
    # loop through each island
    for i in islands:
    
        # unique data for the current island
        island_df = df[df['Island']==i]
    
        # define list to append indices to
        indices = []

        # loop through each major element
        for element in major_elements:

            # Log-transform the data
            log_data = np.log(island_df[element])

            Q1 = np.percentile(log_data, 25)
            Q3 = np.percentile(log_data, 75)

            # IQR is interquartile range
            IQR = Q3 - Q1    

            # define filter, showing which indices fall within the specified range
            filter = (log_data >= Q1 - 2.5*IQR) & (log_data <= Q3 + 2.5*IQR)

            # append filter to the list of indices
            indices.append(filter)

        # combine filters for all major elements
        combined_filter = np.all(indices, axis=0)

        # apply filter
        df_filt = island_df[combined_filter]

        # append filtered dataframe
        filtered_df_list.append(df_filt)
    
    # combine into single dataframe
    filt_combined_df = pd.concat(filtered_df_list, axis=0)
        
    # return filtered dataframe
    return filt_combined_df

# --- visualization ---

# define function to visualize CaO/Al2O3 vs MgO plots and identify statistically significant slopes
def regression_fig(df):
    '''
    input:
        - df: dataframe for a given island group (e.g., hawaii = OIB_df[OIB_df['Group']=='Hawaii'])
    '''
    # convert to cation mol%
    array = np.array(df.select_dtypes(include='number'))
    mol_pct = [cation_mol_pct(sample) for sample in array]

    # reconvert to dataframe
    mol_pct = pd.DataFrame(mol_pct, 
                           columns = ['SiO2','TiO2','AlO1.5','FeO','MnO','MgO','CaO','NaO0.5','KO0.5','PO2.5'],
                           index = df.index)

    # add CaO, AlO1.5 column
    mol_pct['CaO/AlO1.5'] = mol_pct['CaO']/mol_pct['AlO1.5']

    # reappend non-numeric data from input df using index
    non_numeric = df.select_dtypes(exclude='number')
    df_mol_pct = pd.concat([mol_pct, non_numeric], axis=1)

    # Group the DataFrame by the 'Island' column
    grouped = df_mol_pct.groupby('Island')

    # Determine the number of groups
    num_islands = len(grouped)

    # Calculate the number of rows needed
    if num_islands > 4:
        num_rows = 2
    else:
        num_rows = 1

    # Calculate the number of columns needed
    num_cols = (num_islands + num_rows - 1) // num_rows

    # Calculate the figure size for each subplot
    figsize = (num_cols * 4.25, num_rows * 4.25)

    # Create a figure and a set of subplots
    if num_islands > 1:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Iterate over each group (island) and its index
    for idx, (island, data) in enumerate(grouped):
        # Select the current axis to plot on
        if num_islands > 1:  # More than one subplot
            current_ax = axes[idx]
        else:  # Only one subplot
            current_ax = ax

        # Compute linear regression statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['MgO'], data['CaO/AlO1.5'])

        # Check if the correlation is statistically significant (p-value less than 0.05)
        if p_value < 0.05:
            # Create a seaborn regplot for the current island's data
            regplot = sns.regplot(
                x='MgO',
                y='CaO/AlO1.5',
                data=data,
                line_kws={'color': 'black'},
                scatter_kws={'alpha': 0.5},
                ax=current_ax)

            # Retrieve the line from the plot (regression line)
            line = regplot.get_lines()[0]

            # Annotate the plot with the slope
            current_ax.text(0.31, 0.9, f'Slope: {slope:.2f}', transform=current_ax.transAxes,
                            ha='right', va='bottom', fontsize=10, color='crimson')
        else:
            # Plot scatter plot instead of regression line
            current_ax.scatter(data['MgO'], data['CaO/AlO1.5'], alpha=0.5)

        # Set the plot title
        current_ax.set_title(f'{island}',fontsize=20)

        # Set the axis labels
        current_ax.set_xlabel('MgO (mol%)',fontsize=18)
        current_ax.set_ylabel('$\mathrm{CaO/AlO_{1.5}}$',fontsize=18)

    # hide any unused subplots
    if num_islands > 1:  # More than one subplot
        for j in range(idx + 1, num_rows * num_cols):
            fig.delaxes(axes[j])

    # adjust layout and show plot
    plt.tight_layout()

    # show figure
    plt.show()

# define function to visualize liquid line of descent correction path for each sample
def evolution_fig(df, melt_list):
    '''
    input:
        - df: dataframe for a given island (wt%)
        - melt_list: output list from crystal_fract_corr() (all_melt_comps)
        - pdf: pdf file name to save figure to ('string')
    '''
    
    # plot figure showing melt evolution for each data point
    fig, ax = plt.subplots(figsize=(4.25, 4.25))

    # Set the axis labels
    ax.set_xlabel('MgO (mol%)',fontsize=18)
    ax.set_ylabel('$\mathrm{CaO/AlO_{1.5}}$',fontsize=18)
    
    # plot melt correction pathways
    for comp in melt_list:

        # Convert to array
        comp_array = np.array(comp)

        # Define x and y
        MgO = comp_array[:, 5]
        Ca_Al = comp_array[:, 6] / comp_array[:, 2]  # CaO/AlO1.5

        # Plot the melt evolution pathway
        plt.plot(MgO, Ca_Al, color=[0.00392157,0.45098039,0.69803922,0.8], lw=4, alpha=0.1)
    
    # convert df of island data to cation mol% for plotting (array)
    data_array = np.array(df.select_dtypes(include='number'))
    mol_data_array = np.array([cation_mol_pct(s) for s in data_array])
    
    # plot ilsand data (mol%) and associated linear fit
    x = mol_data_array[:, 5]                   # MgO
    y = mol_data_array[:, 6] / mol_data_array[:, 2]  # CaO/AlO1.5

    # Create the regplot using Seaborn
    sns.regplot(x=x, y=y, line_kws={'color': 'black'},scatter_kws={'alpha': 0.5}, ax=ax,)

    # adjust layout and show plot
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# --- fractional crystallization correction ---

# define function to correct for fractional crystallization of ol+cpx
def crystal_fract_corr(df, ol, cpx):
    '''
    input:
        - df: DataFrame for a given island (wt%)
        - ol: Amount of olivine (always 1)
        - cpx: Amount of cpx (relative to ol=1)
    '''

    # Convert df to array for calculation
    island_array = np.array(df.select_dtypes(include='number'))

    # Define target forsterite content (mantle olivine)
    target_Fo = 90

    # Define tolerance for convergence
    tolerance = target_Fo * 0.0001  # within 0.01% of Fo90

    # Define step size (fraction of melt mass)
    step = 0.0005

    # Initialize lists to record melt compositions and mass added/removed
    all_melt_comps = []
    ol_mass_added = []
    cpx_mass_added = []

    # Define olivine and cpx proportions
    ol_pct = ol / (ol + cpx)
    cpx_pct = 1 - ol_pct
    print('Mineral abundance in crystallizing assemblage: olivine =',np.round(ol_pct*100,1),'%;','cpx =',np.round(cpx_pct*100,1),'%')

    # Iterate through each sample
    for sample in island_array:

        # Convert to cation mol%
        mol_pct = cation_mol_pct(sample)

        # Initialize variables
        n = 0
        melt_mass = 1.0  # Initial melt mass
        cumulative_crystal_mass = 0.0
        melt_comps = [mol_pct]
        mass_pct = [0]

        # Calculate initial Fo content and crystallizing assemblage
        Fo_content, cryst_comp = eq_ol_cpx(mol_pct, ol, cpx)

        # Adjust melt composition based on Fo content
        if Fo_content > target_Fo:
            # Subtraction case (removing crystals)
            while Fo_content > target_Fo and abs(Fo_content - target_Fo) > tolerance:
                n += 1

                # Calculate mass of crystals to remove
                delta_crystal_mass = step * melt_mass

                # Update cumulative crystal mass removed
                cumulative_crystal_mass += delta_crystal_mass

                # Adjust melt composition before updating melt mass
                melt_comp = (melt_mass * mol_pct - delta_crystal_mass * cryst_comp) / (melt_mass - delta_crystal_mass)

                # Update melt mass
                melt_mass -= delta_crystal_mass

                # Normalize melt composition to sum to 100%
                mol_pct = closure(melt_comp) * 100

                # Append the evolving melt composition
                melt_comps.append(mol_pct)

                # Append mass percentage removed (negative value)
                mass_pct.append(-delta_crystal_mass)

                # Recalculate Fo_content and cryst_comp
                Fo_content, cryst_comp = eq_ol_cpx(mol_pct, ol, cpx)
        else:
            # Addition case (adding crystals)
            while Fo_content < target_Fo and abs(Fo_content - target_Fo) > tolerance:
                n += 1

                # Calculate mass of crystals to add
                delta_crystal_mass = step * melt_mass

                # Update cumulative crystal mass added
                cumulative_crystal_mass += delta_crystal_mass

                # Adjust melt composition before updating melt mass
                melt_comp = (melt_mass * mol_pct + delta_crystal_mass * cryst_comp) / (melt_mass + delta_crystal_mass)

                # Update melt mass
                melt_mass += delta_crystal_mass

                # Normalize melt composition to sum to 100%
                mol_pct = closure(melt_comp) * 100

                # Append the evolving melt composition
                melt_comps.append(mol_pct)

                # Append mass percentage added
                mass_pct.append(delta_crystal_mass)

                # Recalculate Fo_content and cryst_comp
                Fo_content, cryst_comp = eq_ol_cpx(mol_pct, ol, cpx)

        # Record final compositions and mass percentages
        total_crystal_mass = cumulative_crystal_mass
        ol_mass = (total_crystal_mass * ol_pct) * 100  # Convert to percentage
        cpx_mass = (total_crystal_mass * cpx_pct) * 100

        if Fo_content > target_Fo:
            ol_mass_added.append(-ol_mass)  # Negative values indicate removal
            cpx_mass_added.append(-cpx_mass)
        else:
            ol_mass_added.append(ol_mass)
            cpx_mass_added.append(cpx_mass)

        all_melt_comps.append(np.array(melt_comps))

    # Pull final corrected composition and convert to wt%
    corr_comp_array = np.array([wt_pct(comp[-1]) for comp in all_melt_comps])

    # Reshape ol and cpx arrays for easier concatenation
    ol_array = np.array(ol_mass_added).reshape(-1, 1)
    cpx_array = np.array(cpx_mass_added).reshape(-1, 1)

    # Concatenate corr_comp_array, ol_array, and cpx_array into a single array
    result_array = np.concatenate((corr_comp_array, ol_array, cpx_array), axis=1)

    # Convert to DataFrame
    result_df = pd.DataFrame(result_array,
                             columns=['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO',
                                      'CaO', 'Na2O', 'K2O', 'P2O5', 'ol added (%)', 'cpx added (%)'],
                             index=df.index)

    # Pull non-numeric data from original DataFrame
    loc_df = df.select_dtypes(exclude='number')

    # Combine DataFrames into final output df
    output_df = pd.concat([loc_df, result_df], axis=1)

    # Return output df (wt%) and all_melt_comps array (mol%)
    return output_df, all_melt_comps
