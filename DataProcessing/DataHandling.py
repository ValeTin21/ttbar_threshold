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

#####################
def root_to_dataframe(file_path, tree_name="reco", convert_units=True, verbose=True):
    """
    Convert a ROOT file to a pandas DataFrame with one column per branch.
    
    Parameters:
    -----------
    file_path : str
        Path to the ROOT file
    tree_name : str, default="reco"
        Name of the tree in the ROOT file
    convert_units : bool, default=True
        Whether to convert energy variables from MeV to GeV
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with events as rows and branches as columns
        Vector branches contain arrays
    """
    
    if verbose:
        print("Converting ROOT file to DataFrame")
        print(f"   File: {file_path}")
        print(f"   Tree: {tree_name}")
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"ROOT file not found: {file_path}")
    
    # Open ROOT file and extract all branches
    data = {}
    
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        
        if verbose:
            print(f"   Total branches: {len(tree.keys())}")
            print(f"   Total events: {tree.num_entries}")
        
        scalar_count = 0
        vector_count = 0
        
        # Extract all branches
        for branch_name in tree.keys():
            branch = tree[branch_name]
            
            try:
                # Try to read as numpy array (works for scalar branches)
                values = branch.array(library="np")
                
                # Apply unit conversion if requested
                if convert_units and _should_convert_to_gev(branch_name):
                    values = values / 1000.0  # MeV → GeV
                
                data[branch_name] = values
                scalar_count += 1
                
            except ValueError:
                # This is a vector branch, use awkward arrays
                if AWKWARD_AVAILABLE:
                    try:
                        # Get awkward array
                        branch_data = branch.array(library="ak")
                        
                        # Convert to list of numpy arrays
                        arrays = []
                        for i in range(tree.num_entries):
                            arr = ak.to_numpy(branch_data[i])
                            
                            # Apply unit conversion if requested
                            if convert_units and _should_convert_to_gev(branch_name):
                                arr = arr / 1000.0  # MeV → GeV
                            
                            arrays.append(arr)
                        
                        data[branch_name] = arrays
                        vector_count += 1
                        
                        if verbose:
                            avg_multiplicity = np.mean([len(arr) for arr in arrays])
                            print(f"   Vector branch: {branch_name} (avg {avg_multiplicity:.1f} objects/event)")
                            
                    except Exception as e:
                        if verbose:
                            print(f"   Warning: Could not process vector branch {branch_name}: {e}")
                        continue
                else:
                    if verbose:
                        print(f"   Skipped vector branch {branch_name} (awkward not available)")
                    continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if verbose:
        print(f"\nConversion complete!")
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Scalar branches: {scalar_count}")
        print(f"   Vector branches: {vector_count}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        if convert_units:
            energy_cols = [col for col in df.columns if _should_convert_to_gev(col)]
            print(f"   Energy variables converted to GeV: {len(energy_cols)} columns")
    
    return df

#########################################
def root_to_dataframe_essential(file_path, tree_name="reco", convert_units=True, verbose=True):
    """
    Convert a ROOT file to a pandas DataFrame loading only essential branches for physics analysis.
    This function is optimized for speed and memory usage by loading only required branches.
    
    Parameters:
    -----------
    file_path : str
        Path to the ROOT file
    tree_name : str, default="reco"
        Name of the tree in the ROOT file
    convert_units : bool, default=True
        Whether to convert energy variables from MeV to GeV
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with events as rows and essential branches as columns
        Vector branches contain arrays
    """
    
    # Define essential branches for physics analysis
    essential_branches = [
        'PDFinfo_PDGID1','PDFinfo_PDGID2', 'el_charge', 'el_eta', 'el_phi', 'jet_eta', 'jet_phi', 'mu_charge', 'mu_eta','mu_phi',
        'TtbarLjetsNu_spanet_down_index_NOSYS', 'TtbarLjetsNu_spanet_had_b_index_NOSYS',
        'TtbarLjetsNu_spanet_had_top_assignment_NOSYS', 'TtbarLjetsNu_spanet_had_top_detection_NOSYS',
        'TtbarLjetsNu_spanet_lep_b_index_NOSYS', 'TtbarLjetsNu_spanet_lep_top_assignment_NOSYS',
        'TtbarLjetsNu_spanet_lep_top_detection_NOSYS', 'TtbarLjetsNu_spanet_reg_nu_eta_NOSYS',
        'TtbarLjetsNu_spanet_reg_nu_px_NOSYS', 'TtbarLjetsNu_spanet_reg_nu_py_NOSYS',
        'TtbarLjetsNu_spanet_reg_nu_pz_NOSYS', 'TtbarLjetsNu_spanet_reg_ttbar_m_NOSYS',
        'TtbarLjetsNu_spanet_up_index_NOSYS', 
        'el_e_NOSYS', 'el_pt_NOSYS', 'jet_e_NOSYS', 'jet_pt_NOSYS', 'mu_e_NOSYS', 'mu_pt_NOSYS',
        'met_met_NOSYS', 'met_phi_NOSYS', 'met_sumet_NOSYS', 'pass_ejets_NOSYS','weight_mc_NOSYS'
    ]
    
    if verbose:
        print("Converting ROOT file to DataFrame (ESSENTIAL BRANCHES ONLY)")
        print(f"   File: {file_path}")
        print(f"   Tree: {tree_name}")
        print(f"   Loading {len(essential_branches)} essential branches")
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"ROOT file not found: {file_path}")
    
    # Open ROOT file and extract only essential branches
    data = {}
    
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        
        if verbose:
            print(f"   Total available branches: {len(tree.keys())}")
            print(f"   Total events: {tree.num_entries}")
        
        scalar_count = 0
        vector_count = 0
        missing_branches = []
        
        # Extract only essential branches
        for branch_name in essential_branches:
            if branch_name not in tree.keys():
                missing_branches.append(branch_name)
                continue
                
            branch = tree[branch_name]
            
            try:
                # Try to read as numpy array (works for scalar branches)
                values = branch.array(library="np")
                
                # Apply unit conversion if requested
                if convert_units and _should_convert_to_gev(branch_name):
                    values = values / 1000.0  # MeV → GeV
                
                data[branch_name] = values
                scalar_count += 1
                
            except ValueError:
                # This is a vector branch, use awkward arrays
                if AWKWARD_AVAILABLE:
                    try:
                        # Get awkward array
                        branch_data = branch.array(library="ak")
                        
                        # Convert to list of numpy arrays
                        arrays = []
                        for i in range(tree.num_entries):
                            arr = ak.to_numpy(branch_data[i])
                            
                            # Apply unit conversion if requested
                            if convert_units and _should_convert_to_gev(branch_name):
                                arr = arr / 1000.0  # MeV → GeV
                            
                            arrays.append(arr)
                        
                        data[branch_name] = arrays
                        vector_count += 1
                        
                        if verbose:
                            avg_multiplicity = np.mean([len(arr) for arr in arrays])
                            print(f"   Vector branch: {branch_name} (avg {avg_multiplicity:.1f} objects/event)")
                            
                    except Exception as e:
                        if verbose:
                            print(f"   Warning: Could not process vector branch {branch_name}: {e}")
                        continue
                else:
                    if verbose:
                        print(f"   Skipped vector branch {branch_name} (awkward not available)")
                    continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if verbose:
        print(f"\nESSENTIAL conversion complete!")
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Scalar branches loaded: {scalar_count}")
        print(f"   Vector branches loaded: {vector_count}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        if missing_branches:
            print(f"   ⚠️ Missing branches: {len(missing_branches)}")
            for branch in missing_branches:
                print(f"     - {branch}")
        
        if convert_units:
            energy_cols = [col for col in df.columns if _should_convert_to_gev(col)]
            print(f"   Energy variables converted to GeV: {len(energy_cols)} columns")
        
        print(f"   🚀 Memory saved by loading only essential branches!")
    
    return df

#####################
def _should_convert_to_gev(branch_name):
    """Helper function to determine if a branch should be converted from MeV to GeV"""
    energy_keywords = ['met', '_e_','energy', 'pt', 'mass', 'm_','_nu_px', '_nu_py', '_nu_pz']
    skip_keywords = ['pdf', 'weight', 'number', 'index', 'id', 'channel', 'run', 'event']
    
    has_energy_keyword = any(keyword in branch_name.lower() for keyword in energy_keywords)
    has_skip_keyword = any(keyword in branch_name.lower() for keyword in skip_keywords)
    
    return has_energy_keyword and not has_skip_keyword

######################
def analyze_dataframe(df, verbose=True):
    """
    Analyze the DataFrame structure and content.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    verbose : bool, default=True
        Whether to print detailed information
    
    Returns:
    --------
    dict
        Analysis results
    """
    
    scalar_cols = []
    vector_cols = []
    
    # Classify columns
    for col in df.columns:
        sample_value = df[col].iloc[0]
        if isinstance(sample_value, (list, np.ndarray)) and hasattr(sample_value, '__len__'):
            vector_cols.append(col)
        else:
            scalar_cols.append(col)
    
    # Physics object analysis
    physics_patterns = ['el_', 'mu_', 'jet_', 'tau_', 'photon_']
    physics_cols = {}
    
    for pattern in physics_patterns:
        matching = [col for col in df.columns if pattern in col.lower()]
        if matching:
            physics_cols[pattern.rstrip('_')] = matching
    
    results = {
        'total_columns': len(df.columns),
        'scalar_columns': scalar_cols,
        'vector_columns': vector_cols,
        'physics_columns': physics_cols,
        'events': len(df)
    }
    
    if verbose:
        print(f"\nDataFrame Analysis:")
        print(f"   • Total columns: {len(df.columns)}")
        print(f"   • Scalar columns: {len(scalar_cols)}")
        print(f"   • Vector columns: {len(vector_cols)}")
        print(f"   • Events: {len(df)}")
        
        if physics_cols:
            print(f"\nPhysics Objects:")
            for obj_type, cols in physics_cols.items():
                print(f"   • {obj_type}: {len(cols)} columns")
                for col in cols[:3]:  # Show first 3
                    print(f"     - {col}")
                if len(cols) > 3:
                    print(f"     ... and {len(cols) - 3} more")
    
    return results


##################################
##################################
class FourVecHandler:
    """
    Class to handle 4-vector creation and manipulation for particle physics analysis.
    This version minimizes for loops and creates all 8 four-vectors in a single efficient function.
    """
    
    def __init__(self):
        """Initialize the FourVecHandler."""
        pass
    
    def create_lepton_4vector(self, event, lepton_type, verbose=False):
        """
        Create a 4-vector for a specified lepton type.
        
        Args:
            event: Single row from DataFrame (pandas Series)
            lepton_type: Type of lepton ('electron' or 'muon')
            verbose: Whether to print detailed information
        
        Returns:
            vector object or None if no lepton found
        """
        # Define column mappings for different lepton types
        lepton_columns = {
            'electron': {
                'pt': 'el_pt_NOSYS',
                'eta': 'el_eta',
                'phi': 'el_phi',
                'energy': 'el_e_NOSYS'
            },
            'muon': {
                'pt': 'mu_pt_NOSYS',
                'eta': 'mu_eta',
                'phi': 'mu_phi',
                'energy': 'mu_e_NOSYS'
            }
        }
        
        if lepton_type not in lepton_columns:
            print(f"❌ Unknown lepton type: {lepton_type}")
            return None
        
        cols = lepton_columns[lepton_type]
        
        # Check if required columns exist
        if not all(col in event.index for col in cols.values()):
            print(f"❌ Missing columns for {lepton_type}")
            return None
        
        # Get lepton data
        pt = event[cols['pt']]
        eta = event[cols['eta']]
        phi = event[cols['phi']]
        energy = event[cols['energy']]
        
        # Handle arrays or single values
        if hasattr(pt, '__len__') and not isinstance(pt, str):
            # If it's an array, take the first lepton
            if len(pt) > 0:
                lepton_vec = vector.obj(
                    pt=float(pt[0]),
                    eta=float(eta[0]),
                    phi=float(phi[0]),
                    energy=float(energy[0])
                )
                if verbose:
                    print(f"Created {lepton_type} 4-vector: pt={lepton_vec.pt:.4f}, eta={lepton_vec.eta:.4f}, phi={lepton_vec.phi:.4f}")
                return lepton_vec
            else:
                print(f"❌ No {lepton_type}s in this event")
                return None
        else:
            # Single value
            lepton_vec = vector.obj(
                pt=float(pt),
                eta=float(eta),
                phi=float(phi),
                energy=float(energy)
            )
            if verbose:
                print(f"Created {lepton_type} 4-vector: pt={lepton_vec.pt:.4f}, eta={lepton_vec.eta:.4f}, phi={lepton_vec.phi:.4f}")
            return lepton_vec

    def create_hadron_4vector(self, event, jet_index=0, verbose=False):
        """
        Create a 4-vector for a hadron jet using the energy directly.
        
        Parameters:
        -----------
        event : pandas.Series
            Event data containing jet information
        jet_index : int
            Index of the jet to use (default: 0)
        verbose : bool
            Whether to print debug information
        
        Returns:
        --------
        vector object or None
        """
        
        # Define jet column names
        jet_columns = {
            'pt': 'jet_pt_NOSYS',
            'eta': 'jet_eta',
            'phi': 'jet_phi',
            'energy': 'jet_e_NOSYS'
        }
        
        # Check if required columns exist
        if not all(col in event.index for col in jet_columns.values()):
            if verbose:
                print(f"Missing required jet columns")
            return None
        
        # Get jet data
        jet_pt = event[jet_columns['pt']]
        jet_eta = event[jet_columns['eta']]
        jet_phi = event[jet_columns['phi']]
        jet_energy = event[jet_columns['energy']]
        
        # Handle arrays or single values
        if hasattr(jet_pt, '__len__') and not isinstance(jet_pt, str):
            if len(jet_pt) > jet_index:
                try:
                    # Create 4-vector
                    jet_vec = vector.obj(
                        pt=float(jet_pt[jet_index]),
                        eta=float(jet_eta[jet_index]),
                        phi=float(jet_phi[jet_index]),
                        energy=float(jet_energy[jet_index])
                    )
                    
                    if verbose:
                        print(f"Created jet {jet_index}: pt={jet_vec.pt:.4f}, eta={jet_vec.eta:.4f}, phi={jet_vec.phi:.4f}")
                    
                    return jet_vec
                    
                except (ValueError, IndexError, TypeError) as e:
                    if verbose:
                        print(f"Error creating jet 4-vector: {e}")
                    return None
            else:
                if verbose:
                    print(f"Jet index {jet_index} not available (only {len(jet_pt)} jets)")
                return None
        else:
            # Single jet value
            if jet_index == 0:
                try:
                    pt_val = float(jet_pt)
                    eta_val = float(jet_eta)
                    phi_val = float(jet_phi)
                    energy_val = float(jet_energy)
                    
                    jet_vec = vector.obj(pt=pt_val, eta=eta_val, phi=phi_val, energy=energy_val)
                    
                    if verbose:
                        print(f"   Created single jet: pt={jet_vec.pt:.3f}, eta={jet_vec.eta:.3f}, mass={jet_vec.mass:.3f}")
                    
                    return jet_vec
                    
                except (ValueError, TypeError) as e:
                    if verbose:
                        print(f"   Error creating single jet: {e}")
                    return None
            else:
                if verbose:
                    print(f"❌ Only single jet available, cannot access index {jet_index}")
                return None

    def create_all_4vectors(self, source_df, verbose=True):
        """
        Function to create all 8 four-vectors
        
        Args:
            source_df: DataFrame containing all required physics data
            verbose: Whether to print detailed information
        
        Returns:
            DataFrame with all 8 four-vector columns and top masses
        """
        
        if verbose:
            print("🚀 Creating all 8 four-vectors using OPTIMIZED approach...")
            print(f"   Processing {len(source_df)} events with minimal for loops")
        
        # Check required columns
        required_cols = {
            'neutrino': ['TtbarLjetsNu_spanet_reg_nu_eta_NOSYS', 'TtbarLjetsNu_spanet_reg_nu_px_NOSYS', 
                        'TtbarLjetsNu_spanet_reg_nu_py_NOSYS', 'TtbarLjetsNu_spanet_reg_nu_pz_NOSYS'],
            'spanet_indices': ['TtbarLjetsNu_spanet_down_index_NOSYS', 'TtbarLjetsNu_spanet_up_index_NOSYS',
                              'TtbarLjetsNu_spanet_had_b_index_NOSYS', 'TtbarLjetsNu_spanet_lep_b_index_NOSYS'],
            'leptons': ['el_eta', 'mu_eta', 'el_pt_NOSYS', 'mu_pt_NOSYS', 'el_phi', 'mu_phi', 
                       'el_e_NOSYS', 'mu_e_NOSYS'],
            'jets': ['jet_pt_NOSYS', 'jet_eta', 'jet_phi', 'jet_e_NOSYS']
        }
        
        # Verify all required columns exist
        missing_cols = []
        for category, cols in required_cols.items():
            missing = [col for col in cols if col not in source_df.columns]
            if missing:
                missing_cols.extend(missing)
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return source_df
        
        # ========================================
        # STEP 1: Get neutrino component columns
        # ========================================
        if verbose:
            print("   Step 1: Preparing neutrino component data...")
        
        # Get neutrino columns (will be used inside the loop)
        nu_px_col = 'TtbarLjetsNu_spanet_reg_nu_px_NOSYS'
        nu_py_col = 'TtbarLjetsNu_spanet_reg_nu_py_NOSYS'
        nu_pz_col = 'TtbarLjetsNu_spanet_reg_nu_pz_NOSYS'
        nu_eta_col = 'TtbarLjetsNu_spanet_reg_nu_eta_NOSYS'
        
        # ========================================
        # STEP 2: Pre-allocate arrays for all 8 4-vectors
        # ========================================
        n_events = len(source_df)
        
        # Initialize arrays for all 8 four-vectors
        down_4vecs = np.empty(n_events, dtype=object)
        up_4vecs = np.empty(n_events, dtype=object)
        had_b_4vecs = np.empty(n_events, dtype=object)
        lep_b_4vecs = np.empty(n_events, dtype=object)
        neutrino_4vecs = np.empty(n_events, dtype=object)
        lepton_4vecs = np.empty(n_events, dtype=object)
        had_t_4vecs = np.empty(n_events, dtype=object)
        lep_t_4vecs = np.empty(n_events, dtype=object)
        ttbar_4vecs = np.empty(n_events, dtype=object)

        # Arrays for invariant lepton types, beta
        lepton_types = np.empty(n_events, dtype=object)
        beta = np.empty(n_events, dtype=object)

        # ========================================
        # STEP 3: Single optimized for loop
        # ========================================
        if verbose:
            print("   Step 2: Single optimized loop to create all 8 four-vectors per event...")
        
        # Counters for statistics
        success_counts = {
            'down': 0, 'up': 0, 'had_b': 0, 'lep_b': 0,
            'neutrino': 0, 'lepton': 0, 'had_t': 0, 'lep_t': 0
        }
        
        for i in range(n_events):
            event = source_df.iloc[i]
            
            # ---- Neutrino 4-vector  ----
            nu_px = event[nu_px_col]
            nu_py = event[nu_py_col]
            nu_pz = event[nu_pz_col]
            nu_eta = event[nu_eta_col]
            
            if all(pd.notna(val) for val in [nu_px, nu_py, nu_pz, nu_eta]):
                try:
                    # Convert px, py, pz to pt, phi, energy
                    nu_pt = np.sqrt(float(nu_px)**2 + float(nu_py)**2)
                    nu_phi = np.arctan2(float(nu_py), float(nu_px))
                    nu_energy = np.sqrt(float(nu_px)**2 + float(nu_py)**2 + float(nu_pz)**2)
                    
                    # Create 4-vector using pt, phi, eta, energy
                    neutrino_4vecs[i] = vector.obj(
                        pt=nu_pt,
                        phi=nu_phi,
                        eta=float(nu_eta),
                        energy=nu_energy
                    )
                    success_counts['neutrino'] += 1
                except Exception as e:
                    if verbose and i < 5:  # Only print first few errors
                        print(f"    Warning: Could not create neutrino for event {i}: {e}")
                    neutrino_4vecs[i] = None
            else:
                neutrino_4vecs[i] = None
            
            # ---- Hadron 4-vectors using SpaNET indices ----
            # Down jet
            down_idx = event['TtbarLjetsNu_spanet_down_index_NOSYS']
            if pd.notna(down_idx) and down_idx >= 0:
                down_4vecs[i] = self.create_hadron_4vector(event, jet_index=int(down_idx), verbose=False)
                if down_4vecs[i] is not None:
                    success_counts['down'] += 1
            else:
                down_4vecs[i] = None
            
            # Up jet
            up_idx = event['TtbarLjetsNu_spanet_up_index_NOSYS']
            if pd.notna(up_idx) and up_idx >= 0:
                up_4vecs[i] = self.create_hadron_4vector(event, jet_index=int(up_idx), verbose=False)
                if up_4vecs[i] is not None:
                    success_counts['up'] += 1
            else:
                up_4vecs[i] = None
            
            # Hadronic b jet
            had_b_idx = event['TtbarLjetsNu_spanet_had_b_index_NOSYS']
            if pd.notna(had_b_idx) and had_b_idx >= 0:
                had_b_4vecs[i] = self.create_hadron_4vector(event, jet_index=int(had_b_idx), verbose=False)
                if had_b_4vecs[i] is not None:
                    success_counts['had_b'] += 1
            else:
                had_b_4vecs[i] = None
            
            # Leptonic b jet
            lep_b_idx = event['TtbarLjetsNu_spanet_lep_b_index_NOSYS']
            if pd.notna(lep_b_idx) and lep_b_idx >= 0:
                lep_b_4vecs[i] = self.create_hadron_4vector(event, jet_index=int(lep_b_idx), verbose=False)
                if lep_b_4vecs[i] is not None:
                    success_counts['lep_b'] += 1
            else:
                lep_b_4vecs[i] = None
            
            # ---- Lepton 4-vector
            if event['pass_ejets_NOSYS'] == 1:
                lepton_4vecs[i] = self.create_lepton_4vector(event, 'electron', verbose=False)
                lepton_types[i] = 'electron'
            else:
                lepton_4vecs[i] = self.create_lepton_4vector(event, 'muon', verbose=False)
                lepton_types[i] = 'muon'
            
            if lepton_4vecs[i] is not None:
                success_counts['lepton'] += 1
            
            # ---- Top quark 4-vectors and observables ----
            # Hadronic top
            if (had_b_4vecs[i] is not None and up_4vecs[i] is not None and down_4vecs[i] is not None):
                try:
                    had_t_4vecs[i] = had_b_4vecs[i] + up_4vecs[i] + down_4vecs[i]
                    success_counts['had_t'] += 1
                except:
                    had_t_4vecs[i] = None
            else:
                had_t_4vecs[i] = None
            
            # Leptonic top
            if (lep_b_4vecs[i] is not None and lepton_4vecs[i] is not None and neutrino_4vecs[i] is not None):
                try:
                    lep_t_4vecs[i] = lep_b_4vecs[i] + lepton_4vecs[i] + neutrino_4vecs[i]
                    success_counts['lep_t'] += 1
                except:
                    lep_t_4vecs[i] = None
            else:
                lep_t_4vecs[i] = None
            
            #ttbar 4-vector
            if (had_t_4vecs[i] is not None and lep_t_4vecs[i] is not None):
                try:
                    ttbar_4vecs[i] = had_t_4vecs[i] + lep_t_4vecs[i]
                except:
                    ttbar_4vecs[i] = None
            else:
                ttbar_4vecs[i] = None
                
            #Beta calculation
            if ttbar_4vecs[i] is not None:
                beta[i] = ttbar_4vecs[i].beta
            else:
                beta[i] = None

            # Progress indicator for large datasets
            if verbose and i > 0 and i % 100 == 0:
                print(f"   Processed {i}/{n_events} events ({i/n_events*100:.1f}%)")
        
        # ========================================
        # STEP 4: Add all columns to DataFrame at once
        # ========================================
        if verbose:
            print("   Step 3: Adding all columns to DataFrame...")
        
        result_df = source_df.copy()
        
        # Add all 8 four-vector columns
        result_df['down_4vec'] = down_4vecs
        result_df['up_4vec'] = up_4vecs
        result_df['had_b_4vec'] = had_b_4vecs
        result_df['lep_b_4vec'] = lep_b_4vecs
        result_df['neutrino_4vec'] = neutrino_4vecs
        result_df['lepton_4vec'] = lepton_4vecs
        result_df['had_t'] = had_t_4vecs
        result_df['lep_t'] = lep_t_4vecs
        
        
        # Add auxiliary columns
        result_df['lepton_type'] = lepton_types
        result_df['ttbar_4vec'] = ttbar_4vecs
        result_df['beta'] = beta
                
        if verbose:
            print(f"\n✅ OPTIMIZED 4-vector creation completed!")
            print(f"   Final DataFrame shape: {result_df.shape}")
            print(f"   Added {8} four-vector columns + {7} auxiliary columns")
            
            print(f"\n📊 Success rates:")
            for vec_type, count in success_counts.items():
                percentage = count / n_events * 100
                print(f"   • {vec_type}: {count}/{n_events} ({percentage:.1f}%)")
            
            # Lepton breakdown
            electron_count = sum(1 for ltype in lepton_types if ltype == 'electron')
            muon_count = sum(1 for ltype in lepton_types if ltype == 'muon')
            print(f"   • Leptons: {electron_count} electrons, {muon_count} muons")
            
            print(f"\n⚡ Performance: Single loop processed {n_events} events efficiently!")
        
        return result_df


###################################################
###################################################
def production_classification(df, col1_name='PDFinfo_PDGID1', col2_name='PDFinfo_PDGID2', verbose=True):
    """
    Classify events based on the initial parton types using PDG IDs.
    
    Parameters:
    -----------
    df : pandas.DataFrame00
        DataFrame containing PDG ID columns
    col1_name : str, default='PDFinfo_PDGID1'
        Name of the first PDG ID column
    col2_name : str, default='PDFinfo_PDGID2'
        Name of the second PDG ID column
    verbose : bool, default=True
        Whether to print classification statistics
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'prod_type' column
    """
    
    if verbose:
        print("🚀 Creating event type classification based on PDG IDs")
    
    # Check if required columns exist
    if col1_name not in df.columns or col2_name not in df.columns:
        missing_cols = [col for col in [col1_name, col2_name] if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Use vectorized operations for efficient classification
    col1 = df[col1_name]
    col2 = df[col2_name]
    
    # Create conditions for each event type
    # gg: both are gluons (PDG ID = 21)
    gg_condition = (col1 == 21) & (col2 == 21)
    
    # qq: both are quarks (neither is 21)
    qq_condition = (col1 != 21) & (col2 != 21)
    
    # gq: one is gluon, one is quark (mixed)
    gq_condition = ((col1 == 21) & (col2 != 21)) | ((col1 != 21) & (col2 == 21))
    
    # Apply conditions using numpy.where for vectorized assignment
    prod_type = np.where(gg_condition, 'gg',
                        np.where(qq_condition, 'qq',
                                np.where(gq_condition, 'gq', 'unknown')))
    
    # Add the new column to the DataFrame
    df_result = df.copy()
    df_result['prod_type'] = prod_type
    
    if verbose:
        print("✅ Production type classification completed!")
        
        # Show classification statistics
        prod_type_counts = pd.Series(prod_type).value_counts()
        total_events = len(df_result)
        
        print(f"\n📊 Event type distribution:")
        for event_type, count in prod_type_counts.items():
            percentage = count / total_events * 100
            print(f"   • {event_type}: {count:,} events ({percentage:.1f}%)")
        
        print(f"\n🎯 Total classified events: {total_events:,}")
        
        # Verify that all events are classified (no 'unknown' types)
        unknown_count = (prod_type == 'unknown').sum()
        if unknown_count > 0:
            print(f"⚠️ Warning: {unknown_count} events could not be classified")
        else:
            print("✅ All events successfully classified!")
    
    return df_result

###################################################
###################################################
def calculateD(ttbar_4vec, thad_4vec, tlep_4vec, down_4vec, lep_4vec, apply_ttbar_boost=True):
    """
    Calculate D variable: cosine of angle between lepton and down-type quark
    after appropriate boosts to reference frames.
    
    Parameters:
    -----------
    ttbar_4vec : vector object
        ttbar system 4-vector
    thad_4vec : vector object  
        hadronic top 4-vector
    tlep_4vec : vector object
        leptonic top 4-vector
    down_4vec : vector object
        down-type quark 4-vector
    lep_4vec : vector object
        lepton 4-vector
    apply_ttbar_boost : bool, default=True
        Whether to boost to ttbar rest frame first
        
    Returns:
    --------
    float
        D variable (cosine of angle), or -55.0 if calculation fails
    """
    
    try:
        # Boost everything to ttbar rest frame
        if apply_ttbar_boost:
            # Get boost vector using to_beta3() method to get 3D velocity
            boost_to_ttbar = ttbar_4vec.to_beta3()
            
            # Use boostCM_of_beta3() to boost TO the center-of-mass frame
            thad = thad_4vec.boostCM_of_beta3(boost_to_ttbar)
            tlep = tlep_4vec.boostCM_of_beta3(boost_to_ttbar)
            down = down_4vec.boostCM_of_beta3(boost_to_ttbar)
            lep = lep_4vec.boostCM_of_beta3(boost_to_ttbar)
        else:
            thad = thad_4vec
            tlep = tlep_4vec
            down = down_4vec
            lep = lep_4vec
        
        # Boost down-type quark to hadronic top rest frame
        boost_to_thad = thad.to_beta3()
        down_boosted = down.boostCM_of_beta3(boost_to_thad)
        
        # Boost lepton to leptonic top rest frame  
        boost_to_tlep = tlep.to_beta3()
        lep_boosted = lep.boostCM_of_beta3(boost_to_tlep)
        
        # Calculate cosine of angle between 3-momentum vectors
        lep_3d = lep_boosted.to_3D()
        down_3d = down_boosted.to_3D()
        
        # Get unit vectors
        lep_unit = lep_3d.unit()
        down_unit = down_3d.unit()
        
        # Dot product gives cosine of angle
        D = lep_unit.dot(down_unit)
        
        # Check for NaN
        if np.isnan(D):
            return -55.0
        else:
            return float(D)
            
    except Exception as e:
        # Return error value if any calculation fails
        return -55.0

###################################
###################################
def calculateCosHan(ttbar_4vec, thad_4vec, tlep_4vec, down_4vec, lep_4vec, apply_ttbar_boost=True):
    """
    Calculate C_han variable: like D variable but with lepton z-component flip
    
    """
    try:
        if apply_ttbar_boost:
            # Get boost vector to ttbar rest frame
            boost_to_ttbar = ttbar_4vec.to_beta3()
            
            # Use boostCM_of_beta3() to boost TO the center-of-mass frame
            thad = thad_4vec.boostCM_of_beta3(boost_to_ttbar)
            tlep = tlep_4vec.boostCM_of_beta3(boost_to_ttbar)
            down = down_4vec.boostCM_of_beta3(boost_to_ttbar)
            lep = lep_4vec.boostCM_of_beta3(boost_to_ttbar)
        else:
            # Use particles in their original frame
            thad = thad_4vec
            tlep = tlep_4vec 
            down = down_4vec
            lep = lep_4vec

        # Calculate boost vectors for individual top rest frames
        boost_to_thad = thad.to_beta3()
        boost_to_tlep = tlep.to_beta3()
        
        # Boost down-quark to hadronic top rest frame
        down_boosted = down.boostCM_of_beta3(boost_to_thad)
        
        # Boost lepton to leptonic top rest frame  
        lep_boosted = lep.boostCM_of_beta3(boost_to_tlep)
        
        # Convert to 3D vectors
        down_3d = down_boosted.to_3D()
        lep_3d = lep_boosted.to_3D()
        
        # *** KEY DIFFERENCE FOR C_HAN: Flip z-component of lepton ***
        # Create new vector with flipped z-component
        lep_3d_flipped = vector.obj(
            px=lep_3d.x,
            py=lep_3d.y,
            pz=-lep_3d.z  # FLIP THE Z-COMPONENT
        )
        
        # Calculate unit vectors
        down_3d_unit = down_3d.unit()
        lep_3d_flipped_unit = lep_3d_flipped.unit()
        
        # Calculate cosine of angle between down-quark and z-flipped lepton
        cos_angle = down_3d_unit.dot(lep_3d_flipped_unit)
        
        return cos_angle
        
    except Exception as e:
        # Return failure value
        return -55.0

##############################
##############################
def calculateCosTstar(top_4vec, ttbar_4vec):
    """
    Calculate cos(theta*): cosine of angle between top quark (boosted to ttbar rest frame) 
    and the ttbar direction in the lab frame.
        
    Parameters:
    -----------
    top_4vec : vector object
        Top quark 4-vector (can be either t or tbar)
    ttbar_4vec : vector object
        ttbar system 4-vector
        
    Returns:
    --------
    float
        cos(theta*) variable, or -55.0 if calculation fails
    """
    
    try:
        # Boost top to ttbar rest frame using proper vector library methods
        boost_to_ttbar = ttbar_4vec.to_beta3()
        top_boosted = top_4vec.boostCM_of_beta3(boost_to_ttbar)
        
        # Get ttbar direction vector (3-momentum) in lab frame
        ttbar_3d = ttbar_4vec.to_3D()
        ttbar_direction = ttbar_3d.unit()
        
        # Get boosted top direction vector  
        top_boosted_3d = top_boosted.to_3D()
        top_boosted_direction = top_boosted_3d.unit()
        
        # Calculate cosine of angle between boosted top and ttbar direction
        cos_tstar = top_boosted_direction.dot(ttbar_direction)
        
        # Check for NaN
        if np.isnan(cos_tstar):
            return -55.0
        else:
            return float(cos_tstar)
            
    except Exception as e:
        # Return error value if any calculation fails
        return -55.0




























