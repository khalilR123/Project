import streamlit as st
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
import pubchempy as pcp
import re
import math
from PIL import Image, ImageDraw, ImageFont

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(page_title="Advanced Cheminformatics Visualizer", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 20px;
    }
    </style>
    <div class="main-title">🧪 Comprehensive Molecular Visualizer & Workbench</div>
    <div class="sub-title">Supports IUPAC names, common names, ionic species, condensed formulas, and SMILES strings.</div>
""", unsafe_allow_html=True)

# 2. DICTIONARIES & PARSERS FOR CONDENSED & IONIC SPECIES
IONIC_DATA = {
    "NACL": "[Na+].[Cl-]",
    "SODIUM CHLORIDE": "[Na+].[Cl-]",
    "KCL": "[K+].[Cl-]",
    "POTASSIUM CHLORIDE": "[K+].[Cl-]",
    "C(CH3)4": "CC(C)(C)C",
    "CH3COOH": "CC(=O)O",
    "CH3OH": "CO",
    "CH3CH2OH": "CCO",
    "BENZENE": "c1ccccc1",
    "PHENOL": "Oc1ccccc1"
}

def resolve_molecule(user_input):
    clean_q = user_input.strip()
    
    # Check known mapped formulas/names
    if clean_q.upper() in IONIC_DATA:
        smiles = IONIC_DATA[clean_q.upper()]
        mol = Chem.MolFromSmiles(smiles)
        return mol, smiles, clean_q.title()
    
    # Check general condensed regex like C(CH3)4
    condensed_fix = re.sub(r'C\(CH3\)4', 'CC(C)(C)C', clean_q, flags=re.IGNORECASE)
    
    # 1. Direct SMILES Attempt
    mol = Chem.MolFromSmiles(condensed_fix)
    if mol:
        return mol, condensed_fix, clean_q
    
    # 2. PubChem Lookup Attempt
    try:
        compounds = pcp.get_compounds(clean_q, 'name')
        if compounds and compounds[0].smiles:
            smiles = compounds[0].smiles
            mol = Chem.MolFromSmiles(smiles)
            iupac = compounds[0].iupac_name or clean_q
            return mol, smiles, iupac
    except Exception:
        pass
    
    return None, None, None

# 3. ADVANCED BOND & HYBRIDIZATION CALCULATORS
def calculate_bonds(mol):
    mol_with_h = Chem.AddHs(mol)
    sigma_bonds = 0
    pi_bonds = 0
    
    for bond in mol_with_h.GetBonds():
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            sigma_bonds += 1
        elif btype == Chem.rdchem.BondType.DOUBLE:
            sigma_bonds += 1
            pi_bonds += 1
        elif btype == Chem.rdchem.BondType.TRIPLE:
            sigma_bonds += 1
            pi_bonds += 2
        elif btype == Chem.rdchem.BondType.AROMATIC:
            sigma_bonds += 1
            pi_bonds += 0.5
            
    return sigma_bonds, int(pi_bonds)

def get_hybridization_table(mol):
    table_data = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        hyb = str(atom.GetHybridization())
        
        # Calculate VSEPR Geometry based on hybridization & degree
        deg = atom.GetTotalDegree()
        if hyb == "SP3":
            geom = "Tetrahedral" if deg == 4 else ("Trigonal Pyramidal" if deg == 3 else "Bent")
        elif hyb == "SP2":
            geom = "Trigonal Planar" if deg == 3 else "Bent"
        elif hyb == "SP":
            geom = "Linear"
        else:
            geom = "Unspecified"
            
        table_data.append({
            "Atom Index": idx,
            "Symbol": symbol,
            "Hybridization": hyb,
            "VSEPR Geometry": geom
        })
    return table_data

# 4. PHYSICAL PROPERTIES LOOKUP WITH FALLBACKS
def get_physical_properties(smiles_or_name):
    props = {
        "BP": "Data Dependent on VP / Not Available",
        "FP": "Data Dependent on Temp / Not Available",
        "Solubility": "Calculated via LogP"
    }
    try:
        compounds = pcp.get_compounds(smiles_or_name, 'smiles')
        if compounds:
            c = compounds[0]
            if hasattr(c, 'boiling_point') and c.boiling_point:
                props["BP"] = f"{c.boiling_point} °C"
            if hasattr(c, 'melting_point') and c.melting_point:
                props["FP"] = f"{c.melting_point} °C"
    except Exception:
        pass
    return props

# 5. USER INTERFACE
user_input = st.text_input("Enter Molecule Name, Formula, or SMILES:", value="C(CH3)4")

if user_input:
    mol, active_smiles, resolved_name = resolve_molecule(user_input)
    
    if not mol:
        st.error(f"Could not parse molecule: '{user_input}'. Please check spelling or chemical syntax.")
    else:
        # Prepare 2D coordinates for rendering
        mol_h = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        AllChem.Compute2DCoords(mol_h)
        
        col_left, col_right = st.columns([1.1, 1.0])
        
        # LEFT COLUMN: STRUCTURAL VIEWS & VISUALIZATIONS
        with col_left:
            st.markdown("### 🖼️ Lewis & Structural Display")
            
            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                "Skeletal", "Chirality & Mirror", "Resonance Forms", "Full Lewis (Explicit H)"
            ])
            
            with sub_tab1:
                img_skel = Draw.MolToImage(mol, size=(450, 400))
                st.image(img_skel, caption="Standard Skeletal Notation", use_container_width=True)
                
            with sub_tab2:
                chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                img_chir = Draw.MolToImage(mol_h, size=(450, 400))
                
                # Add central dashed division line
                draw = ImageDraw.Draw(img_chir)
                w, h = img_chir.size
                for y in range(0, h, 10):
                    draw.line([(w // 2, y), (w // 2, y + 5)], fill="red", width=2)
                
                st.image(img_chir, caption="Stereochemistry & Symmetry Mirror Axis", use_container_width=True)
                
                if len(chiral_centers) == 0:
                    st.info("Molecule is **Achiral / Meso** (No non-superimposable mirror image; 0 stereocenters).")
                else:
                    st.success(f"Detected **{len(chiral_centers)}** Stereocenter(s):")
                    for idx, config in chiral_centers:
                        st.write(f"- Atom Index `{idx}`: Configuration **{config}**")

            with sub_tab3:
                try:
                    kek_mol = Chem.Mol(mol)
                    Chem.Kekulize(kek_mol, clearAromaticFlags=True)
                    img_res = Draw.MolsToGridImage([kek_mol], subImgSize=(400, 350), legends=["Kekulé Resonance Form"])
                    st.image(img_res, use_container_width=True)
                    st.caption("Conjugated pi-systems exchange dynamic electron density across aromatic or double bonds.")
                except Exception:
                    st.info("No distinct localized resonance contributors found for this structure.")

            with sub_tab4:
                img_full = Draw.MolToImage(mol_h, size=(450, 400))
                st.image(img_full, caption="Lewis Structure with Explicit Hydrogens and Lone Pairs", use_container_width=True)

        # RIGHT COLUMN: COMPREHENSIVE MOLECULAR PROPERTIES
        with col_right:
            st.markdown("### 📊 Comprehensive Molecular Properties")
            
            sigma_cnt, pi_cnt = calculate_bonds(mol)
            phys = get_physical_properties(active_smiles)
            log_p = Descriptors.MolLogP(mol)
            solubility_class = "Water Soluble / Polar" if log_p < 1.0 else "Lipid Soluble / Nonpolar"
            
            st.write(f"**Resolved Name / IUPAC:** `{resolved_name}`")
            st.write(f"**Molecular Formula:** `{Chem.CalcMolFormula(mol)}`")
            st.write(f"**Molecular Weight (MW):** `{Descriptors.MolWt(mol):.3f} g/mol`")
            st.write(f"**Boiling Point (BP):** {phys['BP']}")
            st.write(f"**Melting / Freezing Point (FP):** {phys['FP']}")
            st.write(f"**Solubility Classification:** {solubility_class}")
            st.write(f"**Bond Counts:** `{sigma_cnt}` $\sigma$ bonds, `{pi_cnt}` $\pi$ bonds")
            st.write(f"**H-Bond Donors / Acceptors:** `{Descriptors.NumHDonors(mol)}` Donors, `{Descriptors.NumHAcceptors(mol)}` Acceptors")
            
            st.markdown("---")
            st.markdown("### 📐 Structural Geometry & Surface Parameters")
            
            st.write(f"**Topological Polar Surface Area (TPSA):** `{Descriptors.TPSA(mol):.2f} Å²`")
            st.write(f"**Octanol-Water Partition Coefficient (LogP):** `{log_p:.2f}`")
            
            # Atom Hybridization & VSEPR Table
            with st.expander("🔬 Atom Hybridization & Geometry Table", expanded=True):
                st.dataframe(get_hybridization_table(mol), use_container_width=True)
