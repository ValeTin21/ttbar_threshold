import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import vector
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import awkward for vector branch handling
try:
    import awkward as ak
    AWKWARD_AVAILABLE = True
except ImportError:
    AWKWARD_AVAILABLE = False

import mplhep as hep
plt.style.use(hep.style.ATLAS)


#####################
def plot_column_distribution(df, column_name, bins=50, title=None, xlabel=None, 
                           log_scale=False, save_path=None, figsize=(10, 6)):
    """
    Plot the distribution of a DataFrame column (handles both scalar and vector columns).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str
        Name of the column to plot
    bins : int, default=50
        Number of histogram bins
    title : str, optional
        Custom title for the plot
    xlabel : str, optional
        Custom x-axis label
    log_scale : bool, default=False
        Whether to use logarithmic y-axis
    save_path : str, optional
        Path to save the plot
    figsize : tuple, default=(10, 6)
        Figure size as (width, height)
    """
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Get the data and determine if it's scalar or vector
    column_data = df[column_name]
    
    # Check if this is a vector column (contains arrays)
    is_vector = False
    sample_value = column_data.iloc[0]
    if isinstance(sample_value, (list, np.ndarray)) and hasattr(sample_value, '__len__'):
        is_vector = True
    
    if is_vector:
        # Flatten all arrays to get individual values
        all_values = []
        total_objects = 0
        events_with_objects = 0
        
        for event_data in column_data:
            if isinstance(event_data, (list, np.ndarray)) and len(event_data) > 0:
                all_values.extend(event_data)
                total_objects += len(event_data)
                events_with_objects += 1
        
        if len(all_values) == 0:
            print(f"Warning: No valid data found in vector column '{column_name}'")
            return
        
        data = np.array(all_values)
        avg_per_event = total_objects / len(column_data) if len(column_data) > 0 else 0
        
    else:
        # Handle scalar column
        data = column_data.dropna()
        if len(data) == 0:
            print(f"Warning: No valid data found in scalar column '{column_name}'")
            return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Auto-detect units and labels
    if xlabel is None:
        xlabel = _get_axis_label(column_name)
    
    if title is None:
        if is_vector:
            title = f'{column_name} Distribution (All Objects)'
        else:
            title = f'{column_name} Distribution'
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    if is_vector:
        ax.set_ylabel('Objects', fontsize=12)
    else:
        ax.set_ylabel('Events', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    # Add statistics text box
    if is_vector:
        stats_text = (f'Events: {len(column_data)}\n'
                     f'Objects: {len(data)}\n'
                     f'Avg/event: {avg_per_event:.2f}\n'
                     f'Mean: {data.mean():.3f}\n'
                     f'Std: {data.std():.3f}')
    else:
        stats_text = (f'Events: {len(data)}\n'
                     f'Mean: {data.mean():.3f}\n'
                     f'Std: {data.std():.3f}\n'
                     f'Range: [{data.min():.3f}, {data.max():.3f}]')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    if is_vector:
        print(f"Vector column '{column_name}' statistics:")
        print(f"   • Total events: {len(column_data)}")
        print(f"   • Total objects: {len(data)}")
        print(f"   • Events with objects: {events_with_objects}")
        print(f"   • Average objects per event: {avg_per_event:.4f}")
        print(f"   • Object values - Mean: {data.mean():.4f}, Std: {data.std():.4f}")
    else:
        print(f"Scalar column '{column_name}' statistics:")
        print(f"   • Total events: {len(data)}")
        print(f"   • Mean: {data.mean():.4f}")
        print(f"   • Std: {data.std():.4f}")
        print(f"   • Min: {data.min():.4f}")
        print(f"   • Max: {data.max():.4f}")


def _get_axis_label(column_name):
    """Auto-generate appropriate axis labels with units"""
    lower_name = column_name.lower()
    
    # Physics variable patterns
    if 'eta' in lower_name:
        return 'η (pseudorapidity)'
    elif 'phi' in lower_name:
        return 'φ [rad]'
    elif any(keyword in lower_name for keyword in ['pt', 'et']):
        return f'{column_name} [GeV]'
    elif any(keyword in lower_name for keyword in ['mass', 'energy']):
        return f'{column_name} [GeV]'
    elif 'met' in lower_name:
        return 'Missing ET [GeV]'
    elif 'weight' in lower_name:
        return 'Weight'
    elif any(keyword in lower_name for keyword in ['number', 'id']):
        return column_name
    else:
        return column_name


def plot_multiplicity(df, column_name, bins=20, title=None):
    """
    Plot the multiplicity distribution for a vector column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with vector columns
    column_name : str
        Name of the vector column
    bins : int, default=20
        Number of histogram bins
    title : str, optional
        Custom title for the plot
    """
    
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in DataFrame")
        return
    
    # Calculate multiplicities
    multiplicities = []
    for event_data in df[column_name]:
        if isinstance(event_data, (list, np.ndarray)):
            multiplicities.append(len(event_data))
        else:
            # Scalar column - multiplicity is always 1 if not NaN
            multiplicities.append(1 if not pd.isna(event_data) else 0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(multiplicities, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel(f'Number of objects per event')
    plt.ylabel('Events')
    
    if title is None:
        title = f'{column_name} Multiplicity Distribution'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    avg_mult = np.mean(multiplicities)
    max_mult = max(multiplicities)
    plt.axvline(avg_mult, color='red', linestyle='--', label=f'Mean: {avg_mult:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Multiplicity statistics for '{column_name}':")
    print(f"   • Average: {avg_mult:.2f}")
    print(f"   • Maximum: {max_mult}")
    print(f"   • Events with 0 objects: {multiplicities.count(0)}")
    print(f"   • Events with 1+ objects: {sum(1 for x in multiplicities if x > 0)}")

###################################################
###################################################
def plot_prod_classification(df, FileName,prod_type_col='prod_type', figsize=(15, 6), colors=None):
    """
    Visualize the production type distribution with pie chart and bar plot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing production type column
    prod_type_col : str, default='prod_type'
        Name of the production type column
    figsize : tuple, default=(15, 6)
        Figure size as (width, height)
    colors : list, optional
        List of colors for the plots. If None, default colors are used.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    
    if prod_type_col not in df.columns:
        raise ValueError(f"Column '{prod_type_col}' not found in DataFrame")
    
    # Default colors if none provided
    if colors is None:
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    # Get production type counts
    prod_type_counts = df[prod_type_col].value_counts()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    wedges, texts, autotexts = axes[0].pie(prod_type_counts.values, 
                                          labels=prod_type_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors[:len(prod_type_counts)],
                                          explode=[0.05]*len(prod_type_counts))
    
    axes[0].set_title('Event Type Distribution\n(Production Mechanism)', fontsize=12, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Bar plot
    bars = axes[1].bar(prod_type_counts.index, prod_type_counts.values, 
                       color=colors[:len(prod_type_counts)], alpha=0.7, edgecolor='black')
    
    axes[1].set_xlabel('Production Type', fontsize=12)
    axes[1].set_ylabel('Number of Events', fontsize=12)
    axes[1].set_title('Event Type Distribution\n(Absolute Counts)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, prod_type_counts.values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Add physics process descriptions
    process_descriptions = {
        'gg': 'Gluon-Gluon\nFusion',
        'qq': 'Quark-Antiquark\nAnnihilation', 
        'gq': 'Gluon-Quark\nScattering'
    }
    
    # Add description text below x-axis labels
    for i, (event_type, description) in enumerate(process_descriptions.items()):
        if event_type in prod_type_counts.index:
            axes[1].text(i, -max(prod_type_counts.values)*0.15, description, 
                        ha='center', va='top', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(FileName)
    plt.show()
    
    return fig

#####################################################################
def plot_variable_by_production_type(df, variable_array, variable_name, FileName, prod_type_col='prod_type', xlabel=None, ylabel='Events', title_prefix=None, figsize=(16, 14),
                                      bins=50, colors=None, overlay_bins=30, density_overlay=True, print_stats=True):
    """
    Plot any variable distribution separated by production type (gg, qq, gq).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing production type column
    variable_array : numpy.array
        Array containing the variable values to plot
    variable_name : str
        Name of the variable for display purposes
    Filename : str
        Name of the output file containing the figure
    prod_type_col : str, default='prod_type'
        Name of the production type column in DataFrame
    xlabel : str, optional
        X-axis label. If None, uses variable_name
    ylabel : str, default='Events'
        Y-axis label for individual plots
    title_prefix : str, optional
        Prefix for plot titles. If None, uses variable_name
    figsize : tuple, default=(16, 14)
        Figure size as (width, height)
    bins : int, default=50
        Number of bins for individual histograms
    colors : dict, optional
        Colors for production types. If None, uses default colors
    overlay_bins : int, default=30
        Number of bins for overlay plot
    density_overlay : bool, default=True
        Whether to use density normalization in overlay plot
    print_stats : bool, default=True
        Whether to print statistics to console
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Set default values
    if xlabel is None:
        xlabel = variable_name
    if title_prefix is None:
        title_prefix = variable_name
    if colors is None:
        colors = {'gg': 'skyblue', 'qq': 'lightcoral', 'gq': 'lightgreen'}
    
    # Validation
    if prod_type_col not in df.columns:
        raise ValueError(f"Column '{prod_type_col}' not found in DataFrame")
    
    if len(variable_array) != len(df):
        raise ValueError(f"Variable array length ({len(variable_array)}) doesn't match DataFrame length ({len(df)})")
    
    print(f"🔬 {variable_name} Analysis by Production Type")
    print("=" * 60)
    
    # Get unique production types
    prod_types = df[prod_type_col].unique()
    
    # FIGURE 1: Individual histograms for each production type (3 separate subplots)
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle(f'{title_prefix} Distribution by Production Type - Individual Plots', fontsize=16, fontweight='bold')
    
    # Individual histograms for each production type
    histograms = {}  # Store histogram data for comparison
    plot_idx = 0
    for prod_type in prod_types:
        if plot_idx < 3:  # First 3 subplots for individual types
            ax = axes[plot_idx]
            
            # Create mask for current production type
            mask = df[prod_type_col] == prod_type
            variable_subset = variable_array[mask]
            
            # Remove NaN values
            valid_data = variable_subset[~np.isnan(variable_subset)]
            
            if len(valid_data) > 0:
                # Calculate histogram for comparison
                hist_counts, hist_edges = np.histogram(valid_data, bins=overlay_bins, density=density_overlay)
                histograms[prod_type] = {'counts': hist_counts, 'edges': hist_edges}
                
                ax.hist(valid_data, bins=bins, alpha=0.7, 
                       color=colors.get(prod_type, 'gray'),
                       edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                
                ax.set_title(f'{title_prefix} Distribution - {prod_type.upper()} Events', fontweight='bold')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box
                stats_text = (f'Events: {len(valid_data):,}\n'
                             f'Mean: {mean_val:.4f}\n'
                             f'Std: {std_val:.4f}\n'
                             f'Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]')
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Print statistics to console
                if print_stats:
                    print(f"\n{prod_type.upper()} Production:")
                    print(f"   • Events: {len(valid_data):,}")
                    print(f"   • Mean: {mean_val:.4f}")
                    print(f"   • Std: {std_val:.4f}")
                    print(f"   • Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]")
                # Add statistics text box
                stats_text = (f'Events: {len(valid_data):,}\n'
                             f'Mean: {mean_val:.3f}\n'
                             f'Std: {std_val:.3f}')
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
            else:
                ax.text(0.5, 0.5, f'No valid\n{variable_name} data\nfor {prod_type.upper()}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title_prefix} Distribution - {prod_type.upper()} Events')
            
            plot_idx += 1
    
    plt.tight_layout()
    individual_filename = FileName.replace('.png', '_individual.png')
    plt.savefig(individual_filename)
    plt.show()
    
    # FIGURE 2: Overlayed distributions + comparison plot
    fig2 = plt.figure(figsize=(12, 10))
    gs = fig2.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    fig2.suptitle(f'{title_prefix} Distribution by Production Type - Overlay & Comparison', fontsize=16, fontweight='bold')
    
    # Overlay plot
    ax_overlay = fig2.add_subplot(gs[0])
    for prod_type in prod_types:
        mask = df[prod_type_col] == prod_type
        variable_subset = variable_array[mask]
        valid_data = variable_subset[~np.isnan(variable_subset)]
        
        if len(valid_data) > 0:
            ax_overlay.hist(valid_data, bins=overlay_bins, alpha=0.6, 
                           color=colors.get(prod_type, 'gray'),
                           edgecolor='black', linewidth=0.5,
                           label=f'{prod_type.upper()} (N={len(valid_data):,})',
                           density=density_overlay)
    
    ax_overlay.set_title(f'Overlayed {title_prefix} Distributions', fontweight='bold')
    ax_overlay.set_xlabel(xlabel)
    ax_overlay.set_ylabel('Density' if density_overlay else ylabel)
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend()
    
    # Comparison subplot: ratio plots
    ax_comp = fig2.add_subplot(gs[1])
    if 'gg' in histograms:
        bin_centers = 0.5 * (histograms['gg']['edges'][:-1] + histograms['gg']['edges'][1:])
        gg_counts = histograms['gg']['counts']
        
        for prod_type in ['qq', 'gq']:
            if prod_type in histograms:
                other_counts = histograms[prod_type]['counts']
                
                # Calculate ratio (avoid division by zero)
                ratio = np.where(gg_counts > 0, other_counts / gg_counts, 1.0)
                
                # Plot ratio
                ax_comp.plot(bin_centers, ratio, 'o-', 
                            color=colors.get(prod_type, 'gray'),
                            label=f'{prod_type.upper()}/gg', linewidth=2, markersize=4)
        
        # Reference line at y=1
        ax_comp.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax_comp.set_xlabel(xlabel)
        ax_comp.set_ylabel('Ratio to gg')
        ax_comp.set_title('Production Type Ratios (relative to gg)')
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend()
        ax_comp.set_ylim(0.5, 1.5)
    else:
        ax_comp.text(0.5, 0.5, 'No gg channel data available for comparison',
                    ha='center', va='center', transform=ax_comp.transAxes)
    
    plt.tight_layout()
    plt.savefig(FileName)
    plt.show()
    
    return fig2
    
#########################################
def fit_linear_on_bin_distributions(df,
                                    betaz_array,
                                    FileName,
                                    beta_col='beta',
                                    prod_type_col='prod_type',
                                    bins=30,
                                    fit_range_beta=(0.5, 0.9),
                                    fit_range_betaz=(0.5, 0.9)):
    """
    For each production type:
      - Histogram β (df[beta_col]) and |β_z| (abs(betaz_array))
      - Restrict to the user-defined range of bin-centers
      - Fit y = m·x + q over that interval
      - Plot the fit line only in that region

    Parameters
    ----------
    df                : pandas.DataFrame
    betaz_array       : np.ndarray
    beta_col          : str
    prod_type_col     : str
    bins              : int or sequence
    fit_range_beta    : tuple (min, max)
    fit_range_betaz   : tuple (min, max)

    Returns
    -------
    dict
      { 'beta':   {prod_type: {'slope':…, 'intercept':…}, …},
        'beta_z': {…} }
    """
    
    # extract numeric arrays
    beta_vals  = df[beta_col].values.astype(float)
    betaz_vals = np.abs(betaz_array).astype(float)

    prod_types = df[prod_type_col].unique()
    fit_results = {'beta': {}, 'beta_z': {}}

    # FIGURE 1: Individual fit plots for beta and beta_z
    fig1 = plt.figure(figsize=(14, 6))
    fig1.suptitle('Beta and Beta_z Linear Fits by Production Type', fontsize=16, fontweight='bold')
    
    variables = [
        ('beta',   beta_vals,  fit_range_beta,  r'β'),
        ('beta_z', betaz_vals, fit_range_betaz, r'|β_z|')
    ]

    for idx, (key, data, fit_range, label) in enumerate(variables):
        # Main fit plot
        ax = fig1.add_subplot(1, 2, idx+1)
        ax.set_title(f"{label} → Linear Fit", fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        
        for pt in prod_types:
            mask = (df[prod_type_col] == pt)
            xvals = data[mask]
            xvals = xvals[np.isfinite(xvals)]
            if len(xvals) < 10:
                continue

            # histogram over full range
            counts, edges = np.histogram(xvals, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            
            ax.hist(xvals, bins=edges, alpha=0.3,
                    label=f"{pt} (N={len(xvals):,})",
                    density=True)

            # select only bins within fit_range
            fm = (centers >= fit_range[0]) & (centers <= fit_range[1])
            xm, ym = centers[fm], counts[fm]

            # drop very small bins (<5% of peak)
            keep = ym > ym.max() * 0.05
            xm, ym = xm[keep], ym[keep]

            if len(xm) >= 3:
                m, q = np.polyfit(xm, ym, 1)
                fit_results[key][pt] = {'slope': m, 'intercept': q}

                # plot fit line only over the fit_range
                xl = np.linspace(fit_range[0], fit_range[1], 200)
                yl = m * xl + q
                ax.plot(xl, yl, '--', lw=2,
                        label=f"{pt.upper()} fit: m={m:.3f}")

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FileName.replace('.png', '_fits.png'))
    plt.show()

    # FIGURE 2: Overlayed distributions + comparison plot with special colors
    fig2 = plt.figure(figsize=(14, 10))
    gs = fig2.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
    fig2.suptitle('Beta and Beta_z Overlay & Comparison Analysis', fontsize=16, fontweight='bold')
    
    colors = {'gg': 'skyblue', 'qq': 'lightcoral', 'gq': 'lightgreen'}
    
    # Store histograms for comparison
    all_histograms = {'beta': {}, 'beta_z': {}}
    
    for idx, (key, data, fit_range, label) in enumerate(variables):
        # Overlayed plot
        ax_overlay = fig2.add_subplot(gs[0, idx])
        histograms = {}
        
        for pt in prod_types:
            mask = (df[prod_type_col] == pt)
            xvals = data[mask]
            xvals = xvals[np.isfinite(xvals)]
            if len(xvals) >= 10:
                counts, edges = np.histogram(xvals, bins=bins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                histograms[pt] = {'counts': counts, 'centers': centers}
                
                ax_overlay.hist(xvals, bins=bins, alpha=0.6,
                               color=colors.get(pt, 'gray'),
                               edgecolor='black', linewidth=0.5,
                               label=f"{pt.upper()} (N={len(xvals):,})", 
                               density=True)
        
        all_histograms[key] = histograms
        ax_overlay.set_title(f"Overlayed {label} Distributions", fontweight='bold')
        ax_overlay.set_xlabel(label)
        ax_overlay.set_ylabel('Density')
        ax_overlay.legend(fontsize=8)
        ax_overlay.grid(True, alpha=0.3)

    # Comparison subplot: ratio plots with special colors for beta/betaz
    ax_comp = fig2.add_subplot(gs[1, :])
    
    # Define special colors for each combination
    special_colors = {
        ('qq', 'beta'): '#FF6B6B',    # Red for qq/gg beta
        ('gq', 'beta'): '#4ECDC4',    # Teal for gq/gg beta  
        ('qq', 'beta_z'): '#9B59B6',  # Purple for qq/gg beta_z
        ('gq', 'beta_z'): '#F39C12'   # Orange for gq/gg beta_z
    }
    
    for idx, (key, data, fit_range, label) in enumerate(variables):
        histograms = all_histograms[key]
        
        if 'gg' in histograms:
            gg_counts = histograms['gg']['counts']
            gg_centers = histograms['gg']['centers']
            
            for prod_type in ['qq', 'gq']:
                if prod_type in histograms:
                    other_counts = histograms[prod_type]['counts']
                    other_centers = histograms[prod_type]['centers']
                    
                    # Calculate ratio (avoid division by zero)
                    ratio = np.where(gg_counts > 0, other_counts / gg_counts, 1.0)
                    
                    # Use special colors and larger markers
                    color = special_colors.get((prod_type, key), 'gray')
                    marker_style = 'o' if key == 'beta' else 's'  # circles for beta, squares for beta_z
                    linestyle = '-' if key == 'beta' else '--'    # solid for beta, dashed for beta_z
                    
                    ax_comp.plot(gg_centers, ratio, marker=marker_style, linestyle=linestyle,
                                color=color, alpha=0.8,
                                label=f'{prod_type}/gg ({label})', 
                                linewidth=2.5, markersize=6)  # Increased marker size
    
    # Reference line at y=1
    ax_comp.axhline(y=1.0, color='black', linestyle=':', alpha=0.7, linewidth=1)
    ax_comp.set_xlabel('β / |βz|')
    ax_comp.set_ylabel('Ratio to gg')
    ax_comp.set_title('Production Type Ratios (relative to gg)')
    ax_comp.grid(True, alpha=0.3)
    ax_comp.legend(fontsize=9, ncol=2)
    ax_comp.set_ylim(0.5, 1.5)

    plt.tight_layout()
    plt.savefig(FileName)
    plt.show()
    return fit_results

#######################################
def plot_jets_per_event_distribution(df, jet_col='jet_eta', figsize=(12, 10), 
                                    prod_type_col='prod_type', colors=None,
                                    save_plot=False, save_path=None):
    """
    Plot the distribution of number of jets per event.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing jet information
    jet_col : str, default='jet_eta'
        Name of the jet column to count jets from
    figsize : tuple, default=(12, 10)
        Figure size as (width, height)
    prod_type_col : str, default='prod_type'
        Name of the production type column for breakdown analysis
    colors : dict, optional
        Colors for production types. If None, uses default colors
    save_plot : bool, default=False
        Whether to save the plot
    save_path : str, optional
        Path to save the plot. If None and save_plot=True, uses default name
    
    Returns:
    --------
    tuple
        (figure_object, jets_per_event_array, statistics_dict)
    """
    
    print("🚀 Analyzing Jets per Event Distribution")
    print("=" * 50)
    
    # Check if required columns exist
    if jet_col not in df.columns:
        raise ValueError(f"Column '{jet_col}' not found in DataFrame")
    
    # Default colors if none provided
    if colors is None:
        colors = {'gg': 'skyblue', 'qq': 'lightcoral', 'gq': 'lightgreen', 'all': 'steelblue'}
    
    # Count jets per event
    jets_per_event = []
    
    print("🔄 Counting jets per event...")
    for idx, jet_data in df[jet_col].items():
        if isinstance(jet_data, (list, np.ndarray)):
            # Count non-NaN jets
            valid_jets = [jet for jet in jet_data if not pd.isna(jet)]
            jets_per_event.append(len(valid_jets))
        else:
            # Single jet or NaN
            jets_per_event.append(1 if not pd.isna(jet_data) else 0)
    
    jets_per_event = np.array(jets_per_event)
    
    # Calculate statistics
    stats = {
        'total_events': len(jets_per_event),
        'mean_jets': jets_per_event.mean(),
        'std_jets': jets_per_event.std(),
        'min_jets': jets_per_event.min(),
        'max_jets': jets_per_event.max(),
        'median_jets': np.median(jets_per_event),
        'total_jets': jets_per_event.sum()
    }
    
    print(f"📊 Overall Statistics:")
    print(f"   • Total events: {stats['total_events']:,}")
    print(f"   • Total jets: {stats['total_jets']:,}")
    print(f"   • Mean jets per event: {stats['mean_jets']:.2f}")
    print(f"   • Median jets per event: {stats['median_jets']:.1f}")
    print(f"   • Std deviation: {stats['std_jets']:.2f}")
    print(f"   • Range: [{stats['min_jets']}, {stats['max_jets']}] jets")
    
    # Create figure with subplots including comparison panel
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 3, 1], hspace=0.3)
    
    # Main plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    # Create sub-gridspec for right column with overlay and comparison
    gs_right = gs[1, 1].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.4)
    ax_overlay = fig.add_subplot(gs_right[0])  # Overlay plot
    ax_comp = fig.add_subplot(gs_right[1])  # Comparison subplot below overlay
    
    fig.suptitle('Jets per Event Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall distribution
    bins = np.arange(stats['min_jets'], stats['max_jets'] + 2) - 0.5
    counts, _, _ = ax1.hist(jets_per_event, bins=bins, alpha=0.7, 
                           color=colors.get('all', 'steelblue'), 
                           edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Number of Jets per Event')
    ax1.set_ylabel('Number of Events')
    ax1.set_title('Overall Jets per Event Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(int(stats['min_jets']), int(stats['max_jets']) + 1))
    
    # Add statistics text
    stats_text = (f'Events: {stats["total_events"]:,}\n'
                 f'Mean: {stats["mean_jets"]:.2f}\n'
                 f'Median: {stats["median_jets"]:.1f}\n'
                 f'Std: {stats["std_jets"]:.2f}')
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot 2: Cumulative distribution
    sorted_jets = np.sort(jets_per_event)
    cumulative = np.arange(1, len(sorted_jets) + 1) / len(sorted_jets)
    ax2.plot(sorted_jets, cumulative, linewidth=2, color=colors.get('all', 'steelblue'))
    ax2.set_xlabel('Number of Jets per Event')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(int(stats['min_jets']), int(stats['max_jets']) + 1))
    
    # Plot 3: By production type (if available)
    if prod_type_col in df.columns:
        prod_types = df[prod_type_col].unique()
        histograms = {}  # Store histogram data for comparison
        
        for prod_type in prod_types:
            mask = df[prod_type_col] == prod_type
            prod_jets = jets_per_event[mask]
            
            # Calculate histogram
            hist_counts, hist_edges = np.histogram(prod_jets, bins=bins, density=True)
            histograms[prod_type] = {'counts': hist_counts, 'edges': hist_edges}
            
            ax3.hist(prod_jets, bins=bins, alpha=0.6, 
                    color=colors.get(prod_type, 'gray'),
                    edgecolor='black', linewidth=0.5,
                    label=f'{prod_type.upper()} (N={len(prod_jets):,})',
                    density=True)
        
        ax3.set_xlabel('Number of Jets per Event')
        ax3.set_ylabel('Density')
        ax3.set_title('Jets per Event by Production Type')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(int(stats['min_jets']), int(stats['max_jets']) + 1))
        
        # Print production type statistics
        print(f"\n📊 Statistics by Production Type:")
        for prod_type in prod_types:
            mask = df[prod_type_col] == prod_type
            prod_jets = jets_per_event[mask]
            if len(prod_jets) > 0:
                print(f"   {prod_type.upper()}:")
                print(f"      • Events: {len(prod_jets):,}")
                print(f"      • Mean jets: {prod_jets.mean():.2f}")
                print(f"      • Std: {prod_jets.std():.2f}")
        
        # Create overlayed comparison plot in the overlay subplot
        for prod_type in prod_types:
            mask = df[prod_type_col] == prod_type
            prod_jets = jets_per_event[mask]
            
            if len(prod_jets) > 0:
                ax_overlay.hist(prod_jets, bins=bins, alpha=0.6, 
                               color=colors.get(prod_type, 'gray'),
                               edgecolor='black', linewidth=0.5,
                               label=f'{prod_type.upper()} (N={len(prod_jets):,})',
                               density=True)
        
        ax_overlay.set_title('Overlayed Jets per Event Distributions', fontweight='bold')
        ax_overlay.set_xlabel('Number of Jets per Event')
        ax_overlay.set_ylabel('Density')
        ax_overlay.grid(True, alpha=0.3)
        ax_overlay.legend()
        ax_overlay.set_xticks(range(int(stats['min_jets']), int(stats['max_jets']) + 1))
        
        # Comparison subplot: ratio plots (now only below overlay plot)
        if 'gg' in histograms:
            bin_centers = 0.5 * (histograms['gg']['edges'][:-1] + histograms['gg']['edges'][1:])
            gg_counts = histograms['gg']['counts']
            
            for prod_type in ['qq', 'gq']:
                if prod_type in histograms:
                    other_counts = histograms[prod_type]['counts']
                    
                    # Calculate ratio (avoid division by zero)
                    ratio = np.where(gg_counts > 0, other_counts / gg_counts, 1.0)
                    
                    # Plot ratio
                    ax_comp.plot(bin_centers, ratio, 'o-', 
                                color=colors.get(prod_type, 'gray'),
                                label=f'{prod_type.upper()}/gg', linewidth=2, markersize=4)
            
            # Reference line at y=1
            ax_comp.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            ax_comp.set_xlabel('Number of Jets per Event')
            ax_comp.set_ylabel('Ratio to gg')
            ax_comp.set_title('Production Type Ratios (relative to gg)')
            ax_comp.grid(True, alpha=0.3)
            ax_comp.legend()
            ax_comp.set_xticks(range(int(stats['min_jets']), int(stats['max_jets']) + 1))
            ax_comp.set_ylim(0.5, 1.5)
        else:
            ax_comp.text(0.5, 0.5, 'No gg channel data available for comparison',
                        ha='center', va='center', transform=ax_comp.transAxes)
            
    else:
        ax3.text(0.5, 0.5, f'Production type\ncolumn not found\n({prod_type_col})',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Jets per Event by Production Type')
        ax_overlay.text(0.5, 0.5, 'No production type data available',
                       ha='center', va='center', transform=ax_overlay.transAxes)
        ax_comp.text(0.5, 0.5, 'No production type data available',
                    ha='center', va='center', transform=ax_comp.transAxes)
    
    # Plot 4: Jet multiplicity table/bar chart (now in bottom row)
    ax4_bottom = fig.add_subplot(gs[2, :])
    unique_jets, jet_counts = np.unique(jets_per_event, return_counts=True)
    percentages = jet_counts / len(jets_per_event) * 100
    
    bars = ax4_bottom.bar(unique_jets, percentages, alpha=0.7, 
                   color=colors.get('all', 'steelblue'), 
                   edgecolor='black', linewidth=0.5)
    
    ax4_bottom.set_xlabel('Number of Jets')
    ax4_bottom.set_ylabel('Percentage of Events (%)')
    ax4_bottom.set_title('Jet Multiplicity Breakdown')
    ax4_bottom.grid(True, alpha=0.3)
    ax4_bottom.set_xticks(unique_jets)
    
    # Add percentage labels on bars
    for bar, count, pct in zip(bars, jet_counts, percentages):
        height = bar.get_height()
        ax4_bottom.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%\n({count})', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if save_path is None:
            save_path = "Plots/jets_per_event_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    plt.show()
    
    # Print jet multiplicity breakdown
    print(f"\n🎯 Jet Multiplicity Breakdown:")
    print("-" * 40)
    for n_jets, count, pct in zip(unique_jets, jet_counts, percentages):
        print(f"   • {n_jets} jets: {count:,} events ({pct:.1f}%)")
    
    return fig, jets_per_event, stats

#######################
def plot_variable_by_production_type_normalized1(df, variable_array, variable_name, 
                                      prod_type_col='prod_type', 
                                      xlabel=None, ylabel='Events',
                                      title_prefix=None, figsize=(16, 12),
                                      bins=50, colors=None, 
                                      overlay_bins=30, density_overlay=True,
                                      physics_interpretation=None,
                                      print_stats=True,
                                      FileName=None):
    """
    Plot variable distribution by production type with SPECIAL NORMALIZATION:
    In the overlay plot, each bin's values from all channels sum to 1.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing production type column
    variable_array : numpy.array
        Array containing the variable values to plot
    variable_name : str
        Name of the variable for display purposes
    prod_type_col : str, default='prod_type'
        Name of the production type column in DataFrame
    xlabel : str, optional
        X-axis label. If None, uses variable_name
    ylabel : str, default='Events'
        Y-axis label for individual plots
    title_prefix : str, optional
        Prefix for plot titles. If None, uses variable_name
    figsize : tuple, default=(16, 12)
        Figure size as (width, height)
    bins : int, default=50
        Number of bins for individual histograms
    colors : dict, optional
        Colors for production types. If None, uses default colors
    overlay_bins : int, default=30
        Number of bins for overlay plot
    density_overlay : bool, default=True
        Whether to use density normalization in overlay plot
    physics_interpretation : list, optional
        List of strings with physics interpretation to print
    print_stats : bool, default=True
        Whether to print statistics to console
    FileName : str, optional
        Filename to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Set default values
    if xlabel is None:
        xlabel = variable_name
    if title_prefix is None:
        title_prefix = variable_name
    if colors is None:
        colors = {'gg': 'skyblue', 'qq': 'lightcoral', 'gq': 'lightgreen'}
    
    # Validation
    if prod_type_col not in df.columns:
        raise ValueError(f"Column '{prod_type_col}' not found in DataFrame")
    
    if len(variable_array) != len(df):
        raise ValueError(f"Variable array length ({len(variable_array)}) doesn't match DataFrame length ({len(df)})")
    
    print(f"🔬 {variable_name} Analysis by Production Type - NORMALIZED TO SUM=1")
    print("=" * 60)
    
    # Get unique production types
    prod_types = df[prod_type_col].unique()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{title_prefix} Distribution by Production Type', fontsize=16, fontweight='bold')
    
    # Individual histograms for each production type
    plot_idx = 0
    for prod_type in prod_types:
        if plot_idx < 3:  # First 3 subplots for individual types
            row = plot_idx // 2
            col = plot_idx % 2
            ax = axes[row, col]
            
            # Create mask for current production type
            mask = df[prod_type_col] == prod_type
            variable_subset = variable_array[mask]
            
            # Remove NaN values
            valid_data = variable_subset
            
            if len(valid_data) > 0:
                ax.hist(valid_data, bins=bins, alpha=0.7, 
                       color=colors.get(prod_type, 'gray'),
                       edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                
                ax.set_title(f'{title_prefix} Distribution - {prod_type.upper()} Events', fontweight='bold')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text box
                stats_text = (f'Events: {len(valid_data):,}\n'
                             f'Mean: {mean_val:.4f}\n'
                             f'Std: {std_val:.4f}\n'
                             f'Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]')
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Print statistics to console
                if print_stats:
                    print(f"\n{prod_type.upper()} Production:")
                    print(f"   • Events: {len(valid_data):,}")
                    print(f"   • Mean: {mean_val:.4f}")
                    print(f"   • Std: {std_val:.4f}")
                    print(f"   • Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]")
                
            else:
                ax.text(0.5, 0.5, f'No valid\n{variable_name} data\nfor {prod_type.upper()}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title_prefix} Distribution - {prod_type.upper()} Events')
            
            plot_idx += 1
    
    # Create NORMALIZED overlayed comparison plot in the remaining subplot
    ax_overlay = axes[1, 1]
    
    # SPECIAL NORMALIZATION: Calculate histograms first, then normalize so each bin sums to 1
    histograms = {}
    bin_edges = None
    
    # Calculate all histograms with the same bins
    for prod_type in prod_types:
        mask = df[prod_type_col] == prod_type
        variable_subset = variable_array[mask]
        valid_data = variable_subset
        
        if len(valid_data) > 0:
            counts, edges = np.histogram(valid_data, bins=overlay_bins, density=False)
            histograms[prod_type] = counts
            if bin_edges is None:
                bin_edges = edges
    
    # Normalize so each bin sums to 1 across all channels
    if len(histograms) > 0 and bin_edges is not None:
        print(f"\n🔄 Applying special normalization: each bin sums to 1 across channels")
        
        # Calculate bin-wise totals
        bin_totals = np.zeros(len(bin_edges) - 1)
        for prod_type, counts in histograms.items():
            bin_totals += counts
        
        # Normalize each histogram so bin-wise sum = 1
        normalized_histograms = {}
        for prod_type, counts in histograms.items():
            # Avoid division by zero
            normalized_counts = np.where(bin_totals > 0, counts / bin_totals, 0)
            normalized_histograms[prod_type] = normalized_counts
        
        # Plot the normalized histograms
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]
        
        for prod_type in prod_types:
            if prod_type in normalized_histograms:
                valid_data_count = len(variable_array[(df[prod_type_col] == prod_type)])
                
                ax_overlay.bar(bin_centers, normalized_histograms[prod_type], 
                              width=bin_width * 0.8, alpha=0.6,
                              color=colors.get(prod_type, 'gray'),
                              edgecolor='black', linewidth=0.5,
                              label=f'{prod_type.upper()} (N={valid_data_count:,})',
                              bottom=np.sum([normalized_histograms[pt] for pt in prod_types 
                                           if pt in normalized_histograms and 
                                           list(prod_types).index(pt) < list(prod_types).index(prod_type)], axis=0))
        
        # Verify normalization
        total_heights = np.sum([normalized_histograms[pt] for pt in normalized_histograms.keys()], axis=0)
        max_total = np.max(total_heights[total_heights > 0])
        min_total = np.min(total_heights[total_heights > 0])
        print(f"   • Bin totals range: [{min_total:.6f}, {max_total:.6f}] (should be close to 1.0)")
        
    ax_overlay.set_title(f'Normalized {title_prefix} Distributions', fontweight='bold')
    ax_overlay.set_xlabel(xlabel)
    ax_overlay.set_ylabel('Normalized Fraction')
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend()
    ax_overlay.set_ylim(0, 1.1)  # Set y-limit since max should be 1
    
    plt.tight_layout()
    
    # Save plot if filename provided
    if FileName:
        plt.savefig(FileName, dpi=50, bbox_inches='tight')    
    plt.show()
    
    return fig










