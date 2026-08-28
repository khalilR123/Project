import streamlit as st
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
import pubchempy as pcp
import re
from PIL import Image, ImageDraw

# Page Configuration
st.set_page_config(page_title="Cheminformatics Visualizer", layout="wide")
st.title("🧪 Advanced Molecular Visualizer & Workbench")

# Pre-processor for condensed formulas and common names
def preprocess_input(query):
    query = query.strip()
    condensed_map = {
        "C(CH3)4": "CC(C)(C)C",
        "CH3COOH": "CC(=O)O",
        "CH3OH": "CO",
        "CH3CH2OH": "CCO",
        "NACL": "[Na+].[Cl-]",
        "SODIUM CHLORIDE": "[Na+].[Cl-]",
        "BENZENE": "c1ccccc1",
        "PHENOL": "Oc1ccccc1"
    }
    if query.upper() in condensed_map:
        return condensed_map[query.upper()]
    
    # Handle general structural patterns like C(CHx)
    query_clean = re.sub(r'C\(CH3\)4', 'CC(C)(C)C', query, flags=re.IGNORECASE)
    return query_clean

# Robust Molecule Resolver
def resolve_molecule(user_input):
    cleaned = preprocess_input(user_input)
    
    # Try direct SMILES parsing first
    mol = Chem.MolFromSmiles(cleaned)
    if mol:
        return mol, cleaned
    
    # Fallback to PubChem database lookup for IUPAC or Common names
    try:
        compounds = pcp.get_compounds(cleaned, 'name')
        if compounds and compounds[0].smiles:
            smiles = compounds[0].smiles
            mol = Chem.MolFromSmiles(smiles)
            return mol, smiles
    except Exception:
        pass
    
    return None, None

# Precise Bond Counter (Sigma and Pi)
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

# Physical Properties Lookup with standard fallback strings
def get_physical_properties(smiles_or_name):
    props = {
        "BP": "Not Available in Database",
        "FP": "Not Available in Database"
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

# --- Main Application Interface ---
user_query = st.text_input("Enter Molecule Name, Formula, or SMILES:", value="C(CH3)4")

if user_query:
    mol, active_smiles = resolve_molecule(user_query)
    
    if not mol:
        st.error(f"Could not parse molecule: '{user_query}'. Please check chemical syntax or enter a valid name/SMILES.")
    else:
        mol_h = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        AllChem.Compute2DCoords(mol_h)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Skeletal", "Chirality & Symmetry", "Resonance", "Properties"])
        
        with tab1:
            st.subheader("Skeletal Structure")
            img = Draw.MolToImage(mol, size=(450, 450))
            st.image(img, use_container_width=False)
            
        with tab2:
            st.subheader("Chirality & Stereochemistry")
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            
            img = Draw.MolToImage(mol_h, size=(450, 450))
            
            if len(chiral_centers) == 0:
                st.info("Molecule is Achiral (No non-superimposable mirror image; no stereocenters).")
            else:
                st.write(f"**Detected Stereocenters:** {len(chiral_centers)}")
                for idx, (atom_idx, config) in enumerate(chiral_centers, 1):
                    st.write(f"- Center {idx} (Atom Index {atom_idx}): Configuration **{config}**")
            
            # Visual reference line for structural examination
            draw = ImageDraw.Draw(img)
            w, h = img.size
            for y in range(0, h, 12):
                draw.line([(w // 2, y), (w // 2, y + 6)], fill="red", width=2)
                
            st.image(img, caption="Red dashed axis marks the central geometric alignment.", use_container_width=False)

        with tab3:
            st.subheader("Resonance Structures")
            try:
                kek_mol = Chem.Mol(mol)
                Chem.Kekulize(kek_mol, clearAromaticFlags=True)
                img_res = Draw.MolsToGridImage([kek_mol], subImgSize=(350, 350), legends=["Kekulé Contributor"])
                st.image(img_res)
                st.caption("Conjugated systems dynamically distribute electron density across double/aromatic bonds.")
            except Exception:
                st.info("No localized resonance forms available for this structure.")

        with tab4:
            st.subheader("Comprehensive Properties")
            sigma_cnt, pi_cnt = calculate_bonds(mol)
            phys_props = get_physical_properties(active_smiles)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**IUPAC / SMILES:** `{active_smiles}`")
                st.write(f"**Molecular Formula:** {Chem.CalcMolFormula(mol)}")
                st.write(f"**Molecular Weight:** {Descriptors.MolWt(mol):.3f} g/mol")
                st.write(f"**Sigma ($\sigma$) Bonds:** {sigma_cnt}")
                st.write(f"**Pi ($\pi$) Bonds:** {pi_cnt}")
            with col2:
                st.write(f"**Boiling Point:** {phys_props['BP']}")
                st.write(f"**Melting Point:** {phys_props['FP']}")
                st.write(f"**TPSA:** {Descriptors.TPSA(mol):.2f} Å²")
                st.write(f"**LogP:** {Descriptors.MolLogP(mol):.2f}")
