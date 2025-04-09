import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re

# manually assign colors and markers (for plotting)
# colors dictionary 
grp_to_c = {
    'Caroline': 'navajowhite',
    'Iceland': 'thistle',
    'Galapagos': 'gainsboro',
    'Juan Fernandez': 'lavender',
    'Hawaii': 'lightsteelblue',
    'Crozet': 'lightskyblue',
    'Pitcairn': 'mediumseagreen',
    'Kerguelen': 'greenyellow',    
    'Gough': 'yellowgreen',
    'Tristan Da Cunha': 'olivedrab',  
    'Society': 'darkcyan',
    'Samoa': 'lightseagreen',
    'Marquesas': 'mediumspringgreen',
    'Mascarene': 'aquamarine',
    'Cape Verde': 'powderblue',
    'Canary': 'orange',
    'Azores': 'pink',   
    'Austral-Cook': 'tomato',
    'St. Helena': 'red'}

# markers dictionary
grp_to_m = {
    'Caroline': 'v',
    'Iceland': 'D',
    'Galapagos': '^',
    'Juan Fernandez': 'o',
    'Hawaii': 's',
    'Crozet': '<',
    'Pitcairn': 'D',
    'Kerguelen': '^',
    'Gough': 'o',
    'Tristan Da Cunha': 's',
    'Society': 's',
    'Samoa': 'o',
    'Cape Verde': '^',
    'Mascarene': 'D',
    'Marquesas': 'v',
    'Austral-Cook': 's',
    'St. Helena': 'o',
    'Azores': '^',
    'Canary': 'D'}

# define elements list (for plotting)
elements = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Rb', 'Sr', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th', 'U']

# define island_group list (for plotting)
island_group = list(grp_to_c.keys())

# check the normality and stability of element distributions for each factor across bootstraps
def plot_factor_distributions(results_processed, elements, save_path=None):
    """
    Plots the stability of factor distributions and optionally saves the plot as a PDF.
    
    Parameters:
    - results_processed: dict
        Dictionary containing the processed results, including 'loadings'.
    - elements: list
        List of labels for the variables (used as y-axis labels).
    - save_path: str, optional
        Path to save the plot as a PDF. If None, the plot will not be saved.
    
    Returns:
    - None
    """
    # Determine the number of factors and variables
    n_factors = results_processed['loadings'][0].shape[1]  # Number of loadings (factors)
    n_variables = results_processed['loadings'][0].shape[0]  # Number of variables per loading

    # Create a grid of subplots, demonstrating stability of factor distributions
    fig, axes = plt.subplots(n_variables, n_factors, figsize=(15, 35))

    # Loop through each factor (loading) and variable to create the histograms
    for i in range(n_factors):
        for j in range(n_variables):
            reorder = []
            for k in results_processed['loadings']:
                reorder.append(k[j][i])  # Collect the values for the j-th variable and i-th factor

            # Calculate mean and standard deviation for the current distribution
            mean = np.mean(reorder)
            std_dev = np.std(reorder)
            
            # Filter out data outside of 3 standard deviations from the mean
            filtered_reorder = [x for x in reorder if (mean - 3 * std_dev) <= x <= (mean + 3 * std_dev)]
            
            # Plot the histogram in the appropriate subplot
            ax = axes[j, i]  # Access the subplot in the j-th row and i-th column
            ax.hist(filtered_reorder, bins=15)
            
            # Keep the original x-axis limits (20% beyond 3 standard deviations)
            x_min = mean - 3 * std_dev * 1.2
            x_max = mean + 3 * std_dev * 1.2
            ax.set_xlim(x_min, x_max)

            # Set the font size for tick labels
            ax.tick_params(axis='both', which='major', labelsize=9)

            # Only set the x-label for the bottom row of plots
            if j == n_variables - 1:
                ax.set_xlabel('Loading Value', fontsize=15)

            # Only set the y-label for the leftmost column of plots
            if i == 0:
                ax.set_ylabel(elements[j], fontsize=15)  # Use the elements list for y-labels

            # Only set the title for the top row
            if j == 0:
                ax.set_title(f'Factor {i+1}', fontsize=20)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the plot as a PDF if a save path is provided
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

# helper function (format subscripts properly in plots)
def format_label_for_subscript(label):
    """
    Replaces any digit(s) in 'label' with subscript notation,
    and wraps the entire label in \mathrm{ } to keep it upright (non-italic) in math mode.
    """
    # This regex finds groups of digits (e.g., '2', '10', etc.)
    def replacer(match):
        digits = match.group(0)
        return '_{' + digits + '}'
    
    # Replace all digits with subscript
    label_with_subscript = re.sub(r'\d+', replacer, label)
    
    # Wrap in math mode and use \mathrm to make the text upright
    return r'$\mathrm{' + label_with_subscript + '}$'

# helper function (format superscripts properly in plots)
def format_label_for_superscript(label):
    """
    Replaces any digit(s) in 'label' with superscript notation,
    and wraps the entire label in \mathrm{ } for upright text in math mode.
    """
    # This regex will find groups of digits (e.g., '2', '10', etc.)
    def replacer(match):
        digits = match.group(0)
        return '^{' + digits + '}'
    
    # Replace all digits with superscript
    label_with_superscript = re.sub(r'\d+', replacer, label)
    
    # Wrap the entire thing in \mathrm so that it is upright (non-italic)
    # and also ensure we're in math mode with $ ... $
    return r'$\mathrm{' + label_with_superscript + '}$'

# define function to visualize loadings
def plot_loadings(results_processed, elements, num_factors=4, save_path=None):
    """
    Plots the factor loadings with error bars representing 2 standard deviations.

    Parameters:
    - results_processed: dict
        Dictionary containing the processed results, including 'loadings_ave', 
        'loadings_2sd', and 'proportion_variance_ave'.
    - elements: list
        List of element/variable names corresponding to the rows in loadings.
    - num_factors: int, optional
        Number of factors to plot. Default is 4.
    - save_path: str, optional
        Path to save the plot as a PDF. If None, the plot will not be saved.

    Returns:
    - None
    """

    # Specify the desired size for each subplot
    subplot_width = 2.5  # Width of each subplot in inches
    subplot_height = 6.5 # Height of each subplot in inches 6.5

    # Calculate the total figure size
    total_width = subplot_width * num_factors
    total_height = subplot_height

    # Initialize a figure with subplots, one for each factor
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_factors,
        figsize=(total_width, total_height),
        sharey=True
    )
    
    # Loop over each factor to create individual plots
    for i in range(num_factors):
        # Extract average loadings and 2-STD values for this factor
        loadings = results_processed['loadings_ave'][:, i]
        two_sd = results_processed['loadings_2sd'][:, i]
        
        # Extract the proportion of variance explained by this factor (for the title)
        proportion_variance = results_processed['proportion_variance_ave'][i] * 100
        
        # Set colors: lighter if loading is small, darker if significant
        threshold = 0.3
        colors = ['mistyrose' if abs(val) < threshold else 'salmon' for val in loadings]
        error_colors = ['gainsboro' if abs(val) < threshold else 'darkgray' for val in loadings]
        
        # Positions for the bars
        y_positions = np.arange(len(elements))
        
        # Plot horizontal bars
        axes[i].barh(y_positions, loadings, align='center', color=colors, alpha=0.85)
        
        # Add individual error bars with conditional colors
        for y_pos, loading, err, err_color in zip(y_positions, loadings, two_sd, error_colors):
            axes[i].errorbar(
                x=loading,
                y=y_pos,
                xerr=err,
                fmt='none',
                ecolor=err_color,
                elinewidth=1.5,
                capsize=3,
                capthick=1.5,
                alpha=0.8
            )

        # Add a vertical line at x=0 to indicate the center
        axes[i].axvline(0, color='gainsboro', linewidth=1.5)
        
        # Set the x-axis limits
        axes[i].set_xlim([-1, 1])
        
        # Title indicating the factor and its explained variance
        axes[i].set_title(f'Factor {i+1}\n',fontsize=18,y=0.96)
        
        # Add a grid on the y-axis for readability
        axes[i].grid(axis='y', color='gainsboro')
        
        # Remove y-axis ticks (element names) for subplots other than the first one
        if i != 0:
            axes[i].set_yticks([])

    # Format the element labels with subscripts and set them on the leftmost subplot
    formatted_elements = [format_label_for_subscript(elem) for elem in elements]
    plt.yticks(range(len(elements)), formatted_elements, size='small')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Display the plot
    plt.show()

# define function to visualize VIP scores
def plot_vip_scores(pls_bootstrap_results, factor_names, top_n=6, save_path=None):
    """
    Creates a row of subplots (one per factor), each showing the top_n VIP scores
    with error bars. Bars have darker color if VIP >= 1.
    
    Parameters:
    - pls_bootstrap_results: dict
        Dictionary containing PLS results for each factor,
        with 'vip_df' key that has columns ['VIP', 'STD'] and indices as features.
    - factor_names: list of str
        Names of the factors to plot (keys in pls_bootstrap_results).
    - top_n: int
        Number of top VIP features to plot per factor (default = 6).
    - save_path: str, optional
        File path to save the figure (e.g., 'output/vip_scores.png').
    """
    # Number of factors
    num_factors = len(factor_names)
    
    # Specify the desired size for each subplot
    subplot_width = 2.4    # Width of each subplot in inches
    subplot_height = 3.5   # Height of each subplot in inches

    # Calculate the total figure size
    total_width = subplot_width * num_factors
    total_height = subplot_height

    # Set up a row of subplots, sharing the y-axis
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=num_factors, 
        figsize=(total_width, total_height),
        sharey=True
    )
    
    for i, factor_name in enumerate(factor_names):
        ax = axes[i]
        
        # Pull out VIP data for the current factor
        vip_df = pls_bootstrap_results[factor_name]['vip_df'].copy()
        # Sort by VIP descending and take the top N
        vip_df = vip_df.sort_values('VIP', ascending=False).head(top_n)
        
        # Extract values and labels
        features = vip_df.index.tolist()
        vip_values = vip_df['VIP'].values
        vip_stds = vip_df['STD'].values
        
        # Positions along the x-axis
        x_positions = np.arange(len(features))
        
        # Plot bars and error bars individually, to replicate the “per-bar” style
        for x_pos, (vip_val, vip_std, feat) in enumerate(zip(vip_values, vip_stds, features)):
            # Choose color based on VIP threshold
            if vip_val >= 1:
                bar_color = 'salmon'
                err_color = 'darkgray'
            else:
                bar_color = 'mistyrose'
                err_color = 'gainsboro'
            
            # Plot the bar
            ax.bar(
                x=x_pos, 
                width=0.6,
                height=vip_val,
                color=bar_color,
                alpha=0.85
            )
            
            # Add error bars
            ax.errorbar(
                x=x_pos, 
                y=vip_val,
                yerr=vip_std,
                fmt='none',
                ecolor=err_color,
                elinewidth=1.5,
                capsize=3,
                capthick=1.5,
                alpha=0.8
            )
        
        # Set the limits for the y-axis (VIP score axis) from 0 to 2
        ax.set_ylim(0, 3.5)
        
        # Format x-tick labels
        formatted_features = [format_label_for_superscript(f) for f in features]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(formatted_features, rotation=90, ha='right')
        
        # Only label the y-axis on the leftmost subplot
        if i == 0:
            ax.set_ylabel("VIP Score", fontsize=15)
        else:
            # Hide y-tick labels for the other subplots
            ax.tick_params(labelleft=False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

# define function to plot biplot
def plot_biplot(results_processed, 
                    data,
                    x_factor=1,
                    y_factor=2,
                    scale_factor=1.1, 
                    x_lim=None, 
                    y_lim=None, 
                    save_path=None):
    """
    Plots the PFA projection for specified factors and optionally saves the plot as a PDF.
    Now supports custom markers for each group.

    Parameters:
    - results_processed: dict
        Dictionary containing the processed results, including 'scores_ave', 'loadings_ave', and 'proportion_variance_ave'.
    - x_factor: int, optional
        The 1-based index of the factor to plot on the x-axis. Default is 1 (first factor).
    - y_factor: int, optional
        The 1-based index of the factor to plot on the y-axis. Default is 2 (second factor).
    - scale_factor: float, optional
        Factor by which to scale the arrows and text labels in the biplot. Default is 1.1.
    - x_lim: tuple of floats, optional
        Tuple specifying the limits of the x-axis (x_min, x_max). If None, limits are calculated automatically.
    - y_lim: tuple of floats, optional
        Tuple specifying the limits of the y-axis (y_min, y_max). If None, limits are calculated automatically.
    - save_path: str, optional
        Path to save the plot as a PDF. If None, the plot will not be saved.
    """
    # Adjust for 0-based indexing in Python
    x_idx = x_factor - 1
    y_idx = y_factor - 1

    # Define figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Extract factor scores from results
    scores_x = results_processed['scores_ave'][:, x_idx]
    scores_y = results_processed['scores_ave'][:, y_idx]

    # -------------------------------------------------------------
    # Plot each group separately, so we can assign color & marker
    # -------------------------------------------------------------

    for loc in island_group:
        # mask out rows belonging to this group
        mask = (data['Group'] == loc)
        plt.scatter(scores_x[mask],
                    scores_y[mask],
                    c=grp_to_c[loc],
                    marker=grp_to_m[loc],
                    s=50,
                    alpha=0.80)

    # Plot biplot: factor loadings as arrows + text
    x_text_positions = []
    y_text_positions = []

    for i in range(len(results_processed['loadings_ave'])):
        x_end = results_processed['loadings_ave'][i, x_idx] * scale_factor
        y_end = results_processed['loadings_ave'][i, y_idx] * scale_factor

        # Draw the arrow from (0, 0)
        plt.arrow(0, 0, x_end, y_end,
                  color='dimgrey',
                  linestyle='-',
                  linewidth=0.75,
                  overhang=4,
                  head_width=0.06,
                  head_length=0.04)

        # Position the text
        x_text = x_end * 1.15
        y_text = y_end * 1.15
        plt.text(x_text, y_text, elements[i], color='k', fontsize=8)

        x_text_positions.append(x_text)
        y_text_positions.append(y_text)

    # -------------------------------------------------------------
    # Create a legend with correct colors + markers for each group
    # -------------------------------------------------------------
    handles = [
        plt.Line2D(
            [0], [0],
            marker=grp_to_m[loc],
            color='w',  # so markerfacecolor is fully visible
            markerfacecolor=grp_to_c[loc],
            markersize=9,
            label=loc,
            alpha=0.75
        )
        for loc in island_group
    ]

    fig.legend(handles=handles,
               fontsize=12,
               title_fontsize=14,
               fancybox=False,
               loc='center left',
               bbox_to_anchor=(0.98, 0.57))

    # Label figure axes
    ax.set_xlabel(f'F{x_factor}', fontsize=22)
    ax.set_ylabel(f'F{y_factor}', fontsize=22)

    # Dynamic axis limits based on data and text positions
    if x_lim is not None:
        x_min, x_max = x_lim
    else:
        x_min_data, x_max_data = scores_x.min(), scores_x.max()
        x_min_text, x_max_text = min(x_text_positions), max(x_text_positions)
        x_min = min(x_min_data, x_min_text) * 1.15
        x_max = max(x_max_data, x_max_text) * 1.15

    if y_lim is not None:
        y_min, y_max = y_lim
    else:
        y_min_data, y_max_data = scores_y.min(), scores_y.max()
        y_min_text, y_max_text = min(y_text_positions), max(y_text_positions)
        y_min = min(y_min_data, y_min_text) * 1.15
        y_max = max(y_max_data, y_max_text) * 1.15

    # Set axes limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    # Adjust layout
    plt.tight_layout()

    # Save the plot if requested
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    # Show the figure
    plt.show()

# define the Function to Create a Bubble Map with Labels and Custom Colorbar/Label Position
def plot_global_map(df, factor=1, fig_width=800, fig_height=600):
    """
    Creates a bubble map for the specified principal factor number with labels and custom figure size.

    Parameters:
    - df (pd.DataFrame): The summary DataFrame with mean and SD.
    - factor (int): The principal factor number.
    - fig_width (int): The width of the figure.
    - fig_height (int): The height of the figure.

    Returns:
    - fig (plotly.graph_objects.Figure): The Plotly figure object.
    """
    pf_mean = f'F{factor}_mean'
    pf_std = f'F{factor}_std'
    pf_label = f'F{factor}'

    fig = px.scatter_geo(
        df,
        lat='Latitude_mean',
        lon='Longitude_mean',
        hover_name='Group',
        hover_data={
            'Latitude_mean': False,
            'Longitude_mean': False,
            pf_mean: ':.2f',
            pf_std: ':.2f'
        },
        color=pf_mean,
        size=pf_std,
        color_continuous_scale='RdBu',
        size_max=20,
        text='Group',
    )

    # update the traces for text label positioning and outline
    fig.update_traces(
        textposition='top right',
        textfont=dict(
            size=10.5,
            color='black'
        ),
        marker=dict(
            line=dict(width=1, color='DarkSlateGrey'),
            opacity=0.8
        )
    )

    # update layout with a customized colorbar
    fig.update_layout(
        geo=dict(
            scope='world',
            projection_type='natural earth',
            showland=True,
            landcolor='lightgrey',
            showcountries=True,
            countrycolor='white',
            framewidth=2
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text=f'F{factor} (mean)',
                font=dict(size=20)),
            outlinecolor='black',
            outlinewidth=1.75,
            ticks='outside',
            tickwidth=1.75,
            tickcolor='black',
            len=0.85,
            thickness=22,
        ),
        legend_title_text='Standard Deviation',
        margin=dict(l=5, r=5, t=10, b=10),
        width=fig_width,
        height=fig_height
    )

    return fig

# plot local archipelago map, with samples colored by factor score
def plot_local_map(island_scores, archipelago, factor=1):
    """
    Parameters:
        island_scores (pd.DataFrame): DataFrame with columns ['Group', 'Latitude', 'Longitude', 'F1', 'F2', ...].
        archipelago (str): The archipelago (or group) name to filter by.
        factor (int): The factor number (1-indexed) to use for the colorbar (default is 1).
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Filter the DataFrame for the specified archipelago
    loc = island_scores[island_scores['Group'] == archipelago]
    
    # Define the factor column name (e.g., 'F1' for factor=1)
    factor_col = f'F{factor}'
    
    # Extract latitude, longitude, and the factor color values
    latitudes = loc['Latitude']
    longitudes = loc['Longitude']
    color_values = loc[factor_col]

    # Create the Cartopy figure with a Mercator projection
    fig, ax = plt.subplots(figsize=(9, 5), subplot_kw={'projection': ccrs.Mercator()})
    
    # Add high-resolution coastlines and land features
    ax.coastlines(resolution='10m', lw=1)
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot the scatter points with color mapping
    scatter = ax.scatter(longitudes, latitudes, c=color_values, cmap='RdBu', s=75, lw=0.5,
                         alpha=0.85, edgecolor='k', zorder=5, transform=ccrs.PlateCarree())
    
    # Set the plot extent based on the data with a small margin
    margin = 0.25
    ax.set_extent([longitudes.min()-margin, longitudes.max()+margin,
                   latitudes.min()-margin, latitudes.max()+margin], crs=ccrs.PlateCarree())

    # Add a colorbar with the selected factor in the label
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.1, location='right')
    cbar.set_label(f'F{factor}', fontsize=22)
    
    plt.show()