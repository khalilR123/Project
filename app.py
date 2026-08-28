import io
import math
import re
import matplotlib.pyplot as plt
import pubchempy as pcp
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cheminformatics Visualizer & Workbench",
    page_icon="🧪",
    layout="wide",
)

# Custom CSS styling for interactive buttons and layout cards
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .stButton>button { width: 100%; border-radius: 6px; height: 3em; font-weight: 600; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧪 Molecular Visualizer & Workbench</div>', unsafe_allow_html=True)
st.caption("Supports IUPAC names, common names, ionic species, proteins/drugs, condensed formulas, and stereochemistry.")

# -----------------------------------------------------------------------------
# 2. INORGANIC, IONIC & CONDENSED FORMULA PARSER
# -----------------------------------------------------------------------------
IONIC_LOOKUP = {
    "NACL": "[Na+].[Cl-]",
    "KCL": "[K+].[Cl-]",
    "SO4-": "[O-]S(=O)(=O)[O-]",
    "SO42-": "[O-]S(=O)(=O)[O-]",
    "SO4-2": "[O-]S(=O)(=O)[O-]",
    "SULFATE": "[O-]S(=O)(=O)[O-]",
    "NO3-": "[O-][N+](=O)[O-]",
    "NITRATE": "[O-][N+](=O)[O-]",
    "NH4+": "[NH4+]",
    "AMMONIUM": "[NH4+]",
    "OH-": "[OH-]",
    "HYDROXIDE": "[OH-]",
}

ACID_SMARTS = [
    ("[CX3](=O)[OX2H1]", "Carboxylic Acid (Strong Organic Acid, pKa ~4-5)"),
    ("S(=O)(=O)[OX2H1]", "Sulfonic Acid (Very Strong Acid, pKa ~ -2)"),
    ("P(=O)(O)O", "Phosphoric acid derivative (Moderate Acid, pKa ~2)"),
    ("c1ccccc1[OX2H1]", "Phenol (Weak Acid, pKa ~10)"),
    ("[#6][OX2H1]", "Alcohol (Very Weak Acid / Neutral, pKa ~15-17)"),
    ("[CX3](=O)[NX3H2,NX3H1]", "Amide (Neutral / Weak Acid, pKa ~15-17)"),
    ("[NX3;H2,H1,H0]", "Amine (Basic, conjugate pKa ~9-11)"),
]


def expand_condensed_formula(query_str: str) -> str:
    """Expands generalized condensed structures like C(CH3)4, C(Cl)4, C(F)4, C(Z)4."""
    q = query_str.strip()

    # Direct Ionic/Inorganic match
    if q.upper() in IONIC_LOOKUP:
        return IONIC_LOOKUP[q.upper()]

    # Match central atom + substituent group + count e.g., C(CH3)4, C(F)4
    pattern_central = r"^([A-Z][a-z]?)\(([^()]+)\)(\d+)$"
    match = re.match(pattern_central, q)
    if match:
        central = match.group(1)
        group = match.group(2)
        count = int(match.group(3))
        return central + f"({group})" * (count - 1) + group

    # Match CH3(CH2)nCH3 or alkyl chains
    pattern_alkyl = r"\(CH3\)(\d+)"
    q = re.sub(pattern_alkyl, lambda m: "C" * int(m.group(1)), q)
    return q


def resolve_input(query: str):
    """
    Converts user input (IUPAC, phenol, caffeine, nicotine, 1-bromobutanol,
    proteins, ionic salts, formulas) to a clean RDKit Mol object and PubChem data.
    """
    clean_q = expand_condensed_formula(query)

    # Try SMILES parse
    mol = Chem.MolFromSmiles(clean_q)
    iupac_name = "N/A"
    formula_str = "N/A"
    pubchem_data = {}

    # Query PubChem online engine for name/formula lookup
    try:
        compounds = pcp.get_compounds(clean_q, "name")
        if not compounds:
            compounds = pcp.get_compounds(clean_q, "formula")

        if compounds and compounds[0].canonical_smiles:
            cpd = compounds[0]
            mol = Chem.MolFromSmiles(cpd.canonical_smiles)
            iupac_name = cpd.iupac_name or (cpd.synonyms[0] if cpd.synonyms else query)
            formula_str = cpd.molecular_formula

            pubchem_data = {
                "bp": getattr(cpd, "boiling_point", "N/A"),
                "fp": getattr(cpd, "flash_point", "N/A"),
                "state": "Solid / Liquid / Gas (Dependent on Temp)",
                "solubility": "Water soluble / Polar" if "O" in formula_str or "N" in formula_str else "Lipid / Nonpolar",
                "tpsa": getattr(cpd, "tpsa", Descriptors.TPSA(mol) if mol else "N/A"),
                "complexity": getattr(cpd, "complexity", "N/A"),
            }
    except Exception:
        pass

    if mol is None:
        mol = Chem.MolFromSmiles(query)

    if mol is None:
        return None, None, None, {}

    if iupac_name == "N/A":
        try:
            smiles = Chem.MolToSmiles(mol)
            cpds = pcp.get_compounds(smiles, "smiles")
            if cpds:
                iupac_name = cpds[0].iupac_name or (cpds[0].synonyms[0] if cpds[0].synonyms else "Custom Molecule")
                formula_str = cpds[0].molecular_formula
        except Exception:
            iupac_name = "Resolved Structure"

    if formula_str == "N/A":
        formula_str = rdMolDescriptors.CalcMolFormula(mol)

    # Sanitize & Assign Cis/Trans and Stereochemistry
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    return mol, iupac_name, formula_str, pubchem_data


# -----------------------------------------------------------------------------
# 3. LEWIS STRUCTURE & STEREOCHEMISTRY DRAWING ENGINE
# -----------------------------------------------------------------------------
def draw_lewis_structure(
    mol: Chem.Mol, canvas_size=450, mark_chiral=False, draw_symmetry=False
) -> Image.Image:
    """Renders Lewis structure with stereochemical wedges, clean non-overlapping lone pairs."""
    mol_copy = Chem.Mol(mol)
    rdMolDraw2D.PrepareMolForDrawing(mol_copy)

    drawer = rdMolDraw2D.MolDraw2DCairo(canvas_size, canvas_size)
    opts = drawer.drawOptions()
    opts.explicitMethyl = True
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 2.2
    opts.padding = 0.20  # Generous padding eliminates atom label overlaps

    # Mark chiral centers with red star (★)
    if mark_chiral:
        centers = Chem.FindMolChiralCenters(mol_copy, includeUnassigned=True)
        for idx, _ in centers:
            opts.atomLabels[idx] = f"★ {mol_copy.GetAtomWithIdx(idx).GetSymbol()}"

    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()

    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Valence dict for accurate non-bonding lone pairs
    valence_dict = {"N": 5, "O": 6, "F": 7, "Cl": 7, "Br": 7, "I": 7, "S": 6, "P": 5}

    for atom in mol_copy.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in valence_dict:
            bonds = atom.GetBonds()
            shared = sum(int(b.GetBondTypeAsDouble()) for b in bonds)
            non_bonding = valence_dict[symbol] - shared - atom.GetFormalCharge()
            lone_pairs = max(0, non_bonding // 2)

            if lone_pairs > 0:
                pos = drawer.GetDrawCoords(atom.GetIdx())
                cx, cy = pos.x, pos.y
                vx, vy = 0.0, 0.0

                for nbr in atom.GetNeighbors():
                    npos = drawer.GetDrawCoords(nbr.GetIdx())
                    dx, dy = cx - npos.x, cy - npos.y
                    dist = math.hypot(dx, dy) or 1.0
                    vx += dx / dist
                    vy += dy / dist

                if vx == 0 and vy == 0:
                    vx, vy = 0.0, -1.0
                norm = math.hypot(vx, vy) or 1.0
                ux, uy = vx / norm, vy / norm

                angles = (
                    [0]
                    if lone_pairs == 1
                    else ([-0.6, 0.6] if lone_pairs == 2 else [-0.8, 0, 0.8])
                )
                for angle in angles[:lone_pairs]:
                    dir_x = ux * math.cos(angle) - uy * math.sin(angle)
                    dir_y = ux * math.sin(angle) + uy * math.cos(angle)
                    lpx, lpy = cx + dir_x * 22, cy + dir_y * 22
                    px, py = -dir_y, dir_x

                    for side in [-1, 1]:
                        draw.ellipse(
                            [
                                lpx + px * (side * 4) - 2.5,
                                lpy + py * (side * 4) - 2.5,
                                lpx + px * (side * 4) + 2.5,
                                lpy + py * (side * 4) + 2.5,
                            ],
                            fill=(0, 0, 0, 255),
                        )

    # Draw small red dashed line indicating internal plane of symmetry for meso/achiral molecules
    if draw_symmetry:
        w, h = img.size
        # Draw dashed line
        for y in range(20, h - 20, 10):
            draw.line([(w // 2, y), (w // 2, y + 5)], fill=(255, 0, 0, 255), width=3)

    return Image.alpha_composite(img, overlay)


def draw_skeletal_structure(mol: Chem.Mol, canvas_size=400) -> Image.Image:
    """Renders pure skeletal structure (no hydrogens on carbon chain)."""
    mol_skel = Chem.RemoveHs(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(canvas_size, canvas_size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.padding = 0.15
    drawer.DrawMolecule(mol_skel)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def draw_enantiomers_side_by_side(mol_orig: Chem.Mol, mol_enant: Chem.Mol) -> Image.Image:
    """Renders original and enantiomer side-by-side with a red separating line."""
    img1 = draw_lewis_structure(mol_orig, canvas_size=350)
    img2 = draw_lewis_structure(mol_enant, canvas_size=350)

    w1, h1 = img1.size
    w2, h2 = img2.size
    sep = 30
    total_w = w1 + w2 + sep
    total_h = max(h1, h2) + 40

    combined = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    combined.paste(img1, (0, 30))
    combined.paste(img2, (w1 + sep, 30))

    draw = ImageDraw.Draw(combined)
    mid_x = w1 + (sep // 2)

    # Red dividing mirror line
    for y in range(20, total_h - 20, 8):
        draw.line([(mid_x, y), (mid_x, y + 4)], fill=(255, 0, 0, 255), width=3)

    draw.text((w1 // 3, 5), "Original Structure", fill=(0, 0, 0, 255))
    draw.text((w1 + sep + (w2 // 3), 5), "Enantiomer (Mirror)", fill=(0, 0, 0, 255))

    return combined


def draw_resonance_structures(mol: Chem.Mol) -> Image.Image:
    """Draws resonance contributors with double-headed arrows (<--->)."""
    mols = [Chem.Mol(mol)]
    try:
        mol_c = Chem.Mol(mol)
        for b in mol_c.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                b.SetBondType(Chem.BondType.SINGLE)
                mols.append(mol_c)
                break
    except Exception:
        pass

    imgs = [draw_lewis_structure(m, canvas_size=280) for m in mols[:2]]
    w1, h1 = imgs[0].size
    sep = 60
    combined = Image.new("RGBA", (w1 * len(imgs) + sep, h1 + 20), (255, 255, 255, 255))

    combined.paste(imgs[0], (0, 10))
    if len(imgs) > 1:
        combined.paste(imgs[1], (w1 + sep, 10))
        draw = ImageDraw.Draw(combined)
        y = h1 // 2
        x1, x2 = w1 + 10, w1 + sep - 10
        draw.line([(x1, y), (x2, y)], fill=(0, 0, 255, 255), width=3)
        draw.polygon([(x1, y), (x1 + 8, y - 5), (x1 + 8, y + 5)], fill=(0, 0, 255, 255))
        draw.polygon([(x2, y), (x2 - 8, y - 5), (x2 - 8, y + 5)], fill=(0, 0, 255, 255))

    return combined


# -----------------------------------------------------------------------------
# 4. STEREOCHEMISTRY ANALYSIS
# -----------------------------------------------------------------------------
def analyze_stereochemistry(mol: Chem.Mol):
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_centers = len(centers)

    opts = StereoEnumerationOptions(onlyUnassigned=False, maxIsomers=16)
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    for iso in isomers:
        Chem.AssignStereochemistry(iso, force=True, cleanIt=True)

    orig_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    # Invert stereocenters to build mirror structure
    inv_mol = Chem.Mol(mol)
    for idx, _ in centers:
        atom = inv_mol.GetAtomWithIdx(idx)
        if atom.HasProp("_CIPCode"):
            cip = atom.GetProp("_CIPCode")
            atom.SetProp("_CIPCode", "S" if cip == "R" else "R")

    inv_smiles = Chem.MolToSmiles(inv_mol, isomericSmiles=True)

    if num_centers == 0:
        return "ACHIRAL", "No", None, []

    if orig_smiles == inv_smiles:
        return "ACHIRAL (MESO COMPOUND)", "Yes", None, isomers

    return "CHIRAL", "No", inv_mol, isomers


# -----------------------------------------------------------------------------
# 5. STREAMLIT APP FRONTEND (MATCHING YOUR HANDWRITTEN LAYOUT)
# -----------------------------------------------------------------------------
user_input = st.text_input(
    "Enter Molecule Name, Formula, or SMILES:",
    value="C(CH3)4",
    placeholder="e.g., phenol, hexanes, caffeine, nicotine, 1-bromobutanol, NaCl, C(CH3)4",
)

if user_input:
    mol, iupac_name, formula_str, pubchem_meta = resolve_input(user_input)

    if mol is None:
        st.error(f"Could not parse molecule: '{user_input}'. Please check spelling or chemical syntax.")
    else:
        chiral_label, is_meso_str, enant_mol, isomers = analyze_stereochemistry(mol)

        # MAIN 2-COLUMN HANDWRITTEN LAYOUT
        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            st.markdown("### 🖼️ Lewis Structure Display")

            if "view_mode" not in st.session_state:
                st.session_state.view_mode = "lewis"

            # 5 Interactive Action Buttons Grid (from your sketch)
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                if st.button("Chirality"):
                    st.session_state.view_mode = "chirality"
            with b2:
                if st.button("Enantiomers"):
                    st.session_state.view_mode = "enantiomers"
            with b3:
                if st.button("Diastereomers"):
                    st.session_state.view_mode = "diastereomers"
            with b4:
                if st.button("Resonance"):
                    st.session_state.view_mode = "resonance"
            with b5:
                if st.button("Skeletal"):
                    st.session_state.view_mode = "skeletal"

            # Render selected interactive mode
            if st.session_state.view_mode == "lewis":
                st.image(draw_lewis_structure(mol), use_container_width=True)

            elif st.session_state.view_mode == "chirality":
                show_red_line = is_meso_str == "Yes" or chiral_label.startswith("ACHIRAL")
                st.image(
                    draw_lewis_structure(mol, mark_chiral=True, draw_symmetry=show_red_line),
                    use_container_width=True,
                )
                st.caption(f"**Status:** {chiral_label}")
                if show_red_line:
                    st.caption("🔴 *Red dashed line indicates internal plane of symmetry.*")

            elif st.session_state.view_mode == "enantiomers":
                if enant_mol:
                    st.image(draw_enantiomers_side_by_side(mol, enant_mol), use_container_width=True)
                else:
                    st.warning("Molecule is Achiral / Meso; it has no non-superimposable enantiomer.")
                    st.image(draw_lewis_structure(mol), use_container_width=True)

            elif st.session_state.view_mode == "diastereomers":
                if len(isomers) > 1:
                    st.subheader("Diastereomer Isomers")
                    for idx, iso in enumerate(isomers[:3]):
                        st.image(draw_lewis_structure(iso), caption=f"Isomer #{idx+1}", use_container_width=True)
                else:
                    st.info("No distinct diastereomers exist for this structure.")
                    st.image(draw_lewis_structure(mol), use_container_width=True)

            elif st.session_state.view_mode == "resonance":
                st.image(draw_resonance_structures(mol), use_container_width=True)

            elif st.session_state.view_mode == "skeletal":
                st.image(draw_skeletal_structure(mol), use_container_width=True)

        with col_right:
            st.markdown("### 📊 Comprehensive Molecular Properties")

            # Basic Metrics Table
            st.write(f"**Resolved IUPAC / Name:** {iupac_name}")
            st.write(f"**Molecular Formula:** {formula_str}")
            st.write(f"**Molecular Weight (MW):** {Descriptors.MolWt(mol):.3f} g/mol")

            # Physical & Temperature Parameters (from notebook)
            st.write(f"**Boiling Point (BP):** {pubchem_meta.get('bp', 'N/A')}")
            st.write(f"**Melting / Freezing Point (FP):** {pubchem_meta.get('fp', 'N/A')}")
            st.write(f"**State at Room Temp:** {pubchem_meta.get('state', 'Solid/Liquid/Gas')}")
            st.write(f"**Solubility:** {pubchem_meta.get('solubility', 'Lipid / Water soluble')}")

            # Counts & Bonds
            num_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
            num_h = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
            hetero = [a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() not in ["C", "H"]]

            st.write(f"**# of Carbon & Hydrogen Atoms:** {num_c} Carbons, {num_h} Hydrogens")
            st.write(f"**# of Heteroatoms:** {len(hetero)} {dict((x, hetero.count(x)) for x in set(hetero)) if hetero else ''}")

            # Sigma and Pi Bond Calculation
            num_bonds = mol.GetNumBonds()
            pi_bonds = sum(int(b.GetBondTypeAsDouble()) - 1 for b in mol.GetBonds())
            sigma_bonds = num_bonds - pi_bonds
            st.write(f"**Bond Counts ($\sigma$ and $\pi$):** {sigma_bonds} $\sigma$ bonds, {int(pi_bonds)} $\pi$ bonds")

            # Acidity & Functional Groups
            acid_desc = "Neutral / Extremely Weak Acid"
            for smarts, label in ACID_SMARTS:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    acid_desc = label
                    break
            st.write(f"**Acid / Base Classification:** {acid_desc}")

            # Hydrogen Bonding & Meso Status
            num_hbd = Descriptors.NumHDonors(mol)
            num_hba = Descriptors.NumHAcceptors(mol)
            st.write(f"**H-Bond Donors / Acceptors:** {num_hbd} Donors, {num_hba} Acceptors")
            st.write(f"**Meso Compound Status:** {is_meso_str}")

            # Topological Polar Surface Area & Vapor Pressure Table
            st.markdown("#### 📐 Structural Geometry & Surface Parameters")
            st.table(
                {
                    "Parameter": ["Topological Polar Surface Area (TPSA)", "Polar / Nonpolar", "Vapor Pressure"],
                    "Value": [
                        f"{Descriptors.TPSA(mol):.2f} Å²",
                        "Polar" if Descriptors.TPSA(mol) > 20 else "Nonpolar",
                        "Dependent on VP Data",
                    ],
                }
            )

            # Atom Geometry & Hybridization Breakdown Table
            with st.expander("🔬 Atom Hybridization & Geometry Table", expanded=True):
                atom_data = []
                for a in mol.GetAtoms():
                    if a.GetSymbol() != "H":
                        hyb = str(a.GetHybridization())
                        geom = "Linear" if hyb == "SP" else ("Trigonal Planar" if hyb == "SP2" else "Tetrahedral")
                        atom_data.append(
                            {
                                "Atom Index": a.GetIdx() + 1,
                                "Symbol": a.GetSymbol(),
                                "Hybridization": hyb,
                                "Geometry": geom,
                            }
                        )
                st.dataframe(atom_data, use_container_width=True)
