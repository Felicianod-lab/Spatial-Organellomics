# -*- coding: utf-8 -*-
"""
Cross-modal inference pipeline for sOrganellomics-to-proteomics linkage.

This plots:
  0. Baseline CONSENSUS OH-to-PH heatmap
  1. Ordering agreement (soft)
  2. Sensitivity to tau (correlation with baseline)
  3. Hard matching stability (bootstrap frequency)
  4. Bootstrap distribution of tau
  5. OH and PH pathway enrichment z-score heatmaps
  6. Distribution reconstruction error
  7. Single density-vs-inferred plots for mitochondria, peroxisome, and lipid droplets
  8. Combined Net Nitrogen Functional Indices bubble panel

"""

import argparse
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wasserstein_distance, zscore


# ============================================================
# USER SETTINGS
# ============================================================

# ORG_PATH = r"X:\For_Alex_and_Mark\FOR_ALEX\NEW_11_Cat_FULL_CONCAT_FOR_PROTEOMIC_LINK.csv"
ORG_PATH = r"...add path...\Fig3_Cross_Modal_Inference_Organellomics_Matrix.csv"
PROT_PATH = r"...add path...\Fig3_Cross_Modal_Inference_Proteomics_Matrix.csv"

# Leave as None to place results beside ORG_PATH when possible.
OUTDIR = None

EPS = 1e-9
RANDOM_SEED = 0
PLOT_DPI = 400
SAVE_FIGURES = True
SHOW_FIGURES = True

# Heatmap style requested for z-score enrichment plots.
ZSCORE_CMAP = "rocket_r"
ZSCORE_VMIN = -2
ZSCORE_VMAX = 2

# ============================================================
# MOLECULAR FEATURE MODE SELECTION
# ============================================================
# Options:
#   "all"      -> all numeric gene/protein columns in the proteomics dataset
#   "162"      -> curated 162-gene/protein panel used in the original publication:
#                 Spatial single-cell mass spectrometry defines zonation of the
#                 hepatocyte proteome. Nature Methods 20, 1530-1536 (2023).
#                 DOI: 10.1038/s41592-023-02007-6
#   "expanded" -> 162 genes + mitochondrial, ER, lipid, peroxisome, redox, and
#                 zonation landmark modules

MOLECULAR_OT_MODE = "expanded"

# ============================================================
# ALIGNMENT AND BOOTSTRAP SETTINGS
# ============================================================

NORMALIZE_X_WITHIN_ACINUS = True

# sOrganellomics and proteomics come from independent mice/acini.
# Therefore all cross-modal spatial alignment is performed cross-acini, not within shared acinus IDs.
# This matches the original RIGOR diagnostic bootstrap logic.
ALIGNMENT_MODE = "cross_acini_independent_mice"
MAX_ACINUS_PAIRS_PER_ENTRY = 500
MIN_SUPPORT_PAIRS = 10
TAU = None  # None -> median finite Wasserstein distance
TAU_MULTS = [0.5, 1.0, 2.0]

CONSENSUS_WEIGHTS = {"spatial": 1.0, "rank": 1.0, "interval": 1.0}

# Batch correction toggle.
#   True  -> apply location correction: X - mean(batch) + mean(global)
#   False -> use the raw proteomics matrix for PH means and OH inference
# You can also override this from the command line:
#   python cross_modal_inference_cleaned_original_plot_style_urea_fixed_batch_toggle.py --batch-correction false
DO_BATCH_LOCATION_CORRECTION = True

# Keep batch-corrected and non-batch-corrected runs in separate result folders.
# This prevents overwriting figures/tables when comparing True vs False.
APPEND_BATCH_MODE_TO_OUTDIR = True

DO_BOOTSTRAP_INFERENCE = True
N_BOOT_INFERENCE = 500
N_BOOT_ALIGNMENT = 300
MIX_N_RECONSTRUCTION = 3000

# Category relabeling.
ORG_CAT_TO_H = {0: "OH1", 1: "OH2", 2: "OH3", 3: "OH4", 4: "OH5"}
PROT_CAT_TO_H = {1: "PH4", 2: "PH2", 3: "PH1", 4: "PH3", 5: "PH5"}

# Common aliases to reduce silent gene dropping in pathway definitions.
GENE_ALIASES = {
    "Glut2": "Slc2a2",
    "Calnexin": "Canx",
}


# ============================================================
# GENE SET DEFINITIONS
# ============================================================

GENE_LIST_162 = ['H2ax',
 'Cps1',
 'Gstm4',
 'Ass1',
 'Aldh1a1',
 'Arg1',
 'Cyp2a12',
 'Gstm1',
 'Cyb5a',
 'Acaa1a',
 'Hbb-bs',
 'Acly',
 'Aldh1b1',
 'Gsta3',
 'Asl',
 'Sptan1',
 'Por',
 'Ugt2b34',
 'Actb',
 'Rack1',
 'Otc',
 'Akr1c6',
 'Ssr1',
 'Myh9',
 'Iqgap2',
 'Fbp1',
 'Gstp1',
 'Rab1a',
 'Alb',
 'Ugt1a5',
 'Acsf2',
 'Gapdh',
 'Aass',
 'Rpl31',
 'Krt18',
 'Ndufa8',
 'C3',
 'Cth',
 'Cyp2d9',
 'Krt8',
 'Uox',
 'Aldob',
 'Cyp2d10',
 'Uqcrc1',
 'Ephx1',
 'Sord',
 'Ak2',
 'Hspe1',
 'Slc27a5',
 'Atp5me',
 'Acsm1',
 'Rplp2',
 'Ces3a',
 'Ehhadh',
 'Ugt2a3',
 'Rpl12',
 'Cltc',
 'Pecr',
 'Sucla2',
 'Tkt',
 'Prdx1',
 'Anxa6',
 'Eef1d',
 'Idh1',
 'Ppia',
 'Acox1',
 'Sod2',
 'Rps25',
 'Cycs',
 'Abcd3',
 'Ugt2b5',
 'Comt',
 'Gnmt',
 'Rpl8',
 'Cox4i1',
 'Hsd17b2',
 'Atp5po',
 'Slc25a5',
 'Calr',
 'Tuba1c',
 'Sec14l2',
 'Ces1d',
 'Aldh2',
 'Ca3',
 'Atp5f1c',
 'Sdhb',
 'Hsd17b10',
 'Acadl',
 'Vdac1',
 'Rps8',
 'Rpl17',
 'Atp5pd',
 'Eci1',
 'Slc25a3',
 'Nme1nme2',
 'Tkfc',
 'Bhmt',
 'Atp5f1b',
 'Atp5pb',
 'Decr1',
 'Sdha',
 'Glyat',
 'Suclg2',
 'Adh1',
 'Rpl11',
 'Fh',
 'Eef1g',
 'Rpl6',
 'Rpl5',
 'Acsl1',
 'Prdx2',
 'Aadac',
 'Dld',
 'Uqcrc2',
 'Snd1',
 'Vcp',
 'Tubb4b',
 'Slc27a2',
 'Hadh',
 'Cat',
 'Hdlbp',
 'Got2',
 'Rdh7',
 'Rplp0',
 'Rpl21',
 'Atp5f1a',
 'Rpl7a',
 'Ndufa4',
 'Etfb',
 'Pdia4',
 'Eef2',
 'Rpl7',
 'Eno1',
 'Ces1f',
 'Aldh4a1',
 'Acadvl',
 'Rrbp1',
 'Cyp2d26',
 'Aldh1l1',
 'Phb1',
 'Mdh2',
 'Hsp90ab1',
 'Ahcy',
 'Aldh8a1',
 'Aco2',
 'Sardh',
 'Aldh7a1',
 'Gstz1',
 'Aifm1',
 'Fasn',
 'Lap3',
 'Acadm',
 'Mthfd1',
 'Ephx2',
 'Acat1',
 'Rps2',
 'Cyb5r3',
 'Rpl4',
 'Aldh6a1',
 'Phb2',
 'Scp2',
 'Pygl']

MITO_SYSTEM_GENES = ['Tomm20',
 'Tomm40',
 'Timm23',
 'Timm44',
 'Mrpl12',
 'Mrpl32',
 'Mrpl44',
 'Mrps18b',
 'Mrps22',
 'Lonp1',
 'Clpp',
 'Hspd1',
 'Mfn1',
 'Mfn2',
 'Opa1',
 'Dnm1l',
 'Idh2',
 'Me2',
 'Me3',
 'Got1',
 'Ndufaf2',
 'Sco1',
 'Cox17',
 'Cs',
 'Hadha',
 'Ogdh']

ER_FUNCTION_GENES = ['Hspa5',
 'Hsp90b1',
 'Calnexin',
 'Canx',
 'Pdia3',
 'Pdia6',
 'P4hb',
 'Derl1',
 'Hrd1',
 'Sel1l',
 'Xbp1',
 'Atf4',
 'Atf6']

LIPID_TRAFFICKING_GENES = ['Plin2',
 'Plin3',
 'Lipa',
 'Agpat2',
 'Agpat3',
 'Lpcat1',
 'Lpcat3',
 'Crot',
 'Acot1',
 'Acot2',
 'Acot8',
 'Hmgcs1',
 'Fdft1',
 'Mlxipl',
 'Prkaa2',
 'Mtor',
 'Slc2a2',
 'Slc2a9',
 'Pdhb',
 'Pdha1',
 'Pdk2',
 'Pdp2',
 'Insr',
 'Creb1',
 'Abcb11',
 'Abcb4',
 'Slc10a1',
 'Abcg8',
 'Abcc3',
 'Abcc2',
 'Abcc6',
 'Sult2a1',
 'Cyp3a41a',
 'Ugt2b5',
 'Ugt2b34',
 'Ugt2b1',
 'Ugt2b36',
 'Ugt1a1',
 'Ugt1a6',
 'Ugt2b1',
 'Ugt2b5',
 'Cyp7a1',
 'Cyp8b1',
 'Cyp27a1',
 'Cyp2f2',
 'Cyp2a4',
 'Cyp2a5',
 'Slco1b2',
 'Cyp3a11',
 'Apoe',
 'Apob',
 'Mttp',
 'Dgat1',
 'Ldlr',
 'Lrp1',
 'Mgll',
 'Fitm2',
 'Gpat4',
 'Acsl5']

PEROXISOME_EXPANSION_GENES = ['Pex1', 'Pex3', 'Pex5', 'Pex19', 'Pex14', 'Abcd2', 'Acot3', 'Acot4']

REDOX_GENES = ['Txn',
 'Txnrd1',
 'Txnrd2',
 'Gclc',
 'Gclm',
 'Gpx1',
 'Gpx4',
 'Me1',
 'G6pdx',
 'Pgd',
 'Sod1',
 'Glud1',
 'Gpt',
 'Gpt2',
 'Shmt1',
 'Shmt2',
 'Bcat2',
 'Ivd',
 'Bckdha',
 'Bckdhb',
 'Mccc1',
 'Mccc2',
 'Hmgcl',
 'Slc25a12',
 'Slc25a13',
 'Slc25a15',
 'Slc25a22',
 'Hal',
 'Aspg',
 'Prodh',
 'Slc38a4',
 'Nags',
 'Gck',
 'Hk1',
 'Pfkl',
 'Pfkm',
 'Pfkfb1',
 'Ldha',
 'Ldhb',
 'Pgk1',
 'Dlat',
 'Slc25a1',
 'Mpc1',
 'Mpc2',
 'Acaca',
 'Acss2',
 'Ctnnb1']

landmarks = ['Gls2',
 'Mat1a',
 'Sds',
 'Ass1l',
 'Hao1',
 'Pck1',
 'Cyp4a10',
 'Bhmt2',
 'Cdh1',
 'Hsd17b13',
 'Pkm',
 'Aldoa',
 'Cyp2e1',
 'Cyp1a2',
 'Cyp2c29',
 'Glul',
 'Slc1a2',
 'Aldh1a7',
 'Oat']



# ============================================================
# PATHWAY DEFINITIONS - 
# ============================================================

PATHWAY_GENES = OrderedDict({
    "Wnt_beta_catenin_signaling": ["Ctnnb1"],
    "Bile_acid_metabolism": [
        "Cyp2a12", "Cyp2d9", "Cyp2d10", "Cyp2d26",
        "Ugt1a5", "Ugt2b5", "Ugt2b34",
    ],
    "Xenobiotic_metabolism": [
        "Gstm1", "Gstm4", "Gsta3", "Gstp1", "Gstz1", "Gclc", "Gclm",
        "Ephx1", "Ephx2", "Comt", "Txn", "Txnrd1", "Txnrd2",
    ],
    "Cholesterol_biosynthesis": ["Hmgcs1", "Fdft1"],
    "Oxidative_stress": ["Sod1", "Prdx1", "Prdx2", "Cat", "Gpx1", "Gpx4"],
    "Lipogenesis_NADPH_support": ["Idh1", "Me1", "G6pdx", "Pgd"],
    "Glycolysis": [
        "Gapdh", "Aldoa", "Eno1", "Pkm", "Gck", "Hk1",
        "Pfkl", "Pfkm", "Pfkfb1", "Pgk1",
    ],
    "Lipoprotein_breakdown_reprocessing_VLDL_export": [
        "Lipa", "Scp2", "Abcd2", "Abcd3", "Acox1", "Ehhadh", "Acot3", "Acot4",
        "Gpat4", "Agpat2", "Dgat1", "Apob", "Mttp", "Sec14l2",
    ],
    "Pyruvate_to_Citrate": [
        "Mpc1", "Mpc2", "Dlat", "Pdhb", "Pdha1", "Pdk2", "Pdp2", "Cs",
        "Slc25a1", "Acss2",
    ],
    "Pyruvate_to_Lactate": ["Ldha", "Ldhb"],
    "Glucose_induced_lipogenesis": ["Slc2a2", "Mlxipl", "Fasn", "Acly", "Acaca"],
    "Lipogenesis_core": ["Fasn", "Acly", "Acaca"],
    "Gluconeogenesis": ["Fbp1", "Pck1", "G6pc", "Aldob"],
    "Urea_cycle": ["Cps1", "Nags", "Ass1", "Asl", "Arg1", "Otc"],
    "Oxidative_phosphorylation": [
        "Cox4i1", "Cox17", "Ndufa4", "Ndufa8", "Ndufaf2", "Uqcrc1", "Uqcrc2",
        "Sdha", "Sdhb", "Atp5f1a", "Atp5f1b", "Atp5f1c", "Atp5pb", "Atp5po",
        "Atp5me",
    ],
    "Mitochondrial_import_biogenesis": ["Tomm20", "Tomm40", "Timm23", "Timm44"],
    "Peroxisome_biogenesis_maintenance": ["Pex1", "Pex3", "Pex5", "Pex19", "Pex14"],
    "Lipid_droplet_accumulation": [
        "Slc27a2", "Slc27a5", "Slc2a2", "Mlxipl", "Fasn", "Acly", "Acaca", "Hsd17b13",
    ],
})

NITROGEN_PATHWAYS = OrderedDict({
    "AA_Degradation_Leucine": ["Mccc1", "Mccc2", "Hmgcl"],
    "AA_Degradation_Lysine": ["Aass", "Aldh7a1"],
    "AA_Degradation_Val_Ile": ["Bckdha", "Bckdhb", "Ivd", "Aldh6a1"],
    "AA_Degradation_Proline": ["Prodh", "Aldh4a1"],
    "AA_Degradation_Histidine": ["Hal"],
    "AA_Degradation_Arginine": ["Aspg"],
    "AA_Degradation_Serine_Glycine": ["Sds", "Shmt1", "Shmt2"],
    "Other_Ammonia_Generating": ["Sardh", "Gnmt", "Cth"],
    "Oxidative_Deamination_Glutamate": ["Glud1"],

  
    "Urea_cycle_Mito": ["Cps1", "Nags"],


    "OneCarbon_Methyl": ["Bhmt", "Ahcy"],
    "OneCarbon_Folate": ["Mthfd1", "Aldh1l1"],
    "Nitrogen_Consumer": ["Glul"],
})

AA_BREAKDOWN_PATHS = [
    "AA_Degradation_Leucine",
    "AA_Degradation_Lysine",
    "AA_Degradation_Val_Ile",
    "AA_Degradation_Proline",
    "AA_Degradation_Histidine",
    "AA_Degradation_Arginine",
    "AA_Degradation_Serine_Glycine",
]

AMMONIA_GEN_PATHS = [
    "AA_Degradation_Histidine",
    "AA_Degradation_Arginine",
    "AA_Degradation_Serine_Glycine",
    "Other_Ammonia_Generating",
    "Oxidative_Deamination_Glutamate",
]


UREA_PATHS = ["Urea_cycle_Mito"]
ONE_CARBON_PATHS = ["OneCarbon_Methyl", "OneCarbon_Folate"]
GLUTAMINE_SYNTHESIS_PATHS = ["Nitrogen_Consumer"]


# ============================================================
# SMALL UTILITIES
# ============================================================

def make_outdir(org_path, outdir=None):
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        return outdir
    base = os.path.dirname(org_path)
    if not base:
        base = os.getcwd()
    outdir = os.path.join(base, "sOrganellomics_proteomics_results")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_table(df, filename):
    path = os.path.join(OUTDIR, filename)
    df.to_csv(path)
    print(f"[saved] {filename}")
    return path


def finish_plot(fig, filename):
    if SAVE_FIGURES:
        fig.savefig(os.path.join(OUTDIR, filename), dpi=fig.dpi, bbox_inches="tight")
        print(f"[saved] {filename}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def ordered_unique(items):
    return list(OrderedDict.fromkeys(items))


def mapped_gene(gene):
    return GENE_ALIASES.get(gene, gene)


def filter_gene_dict(gene_dict, available_genes):
    filtered = OrderedDict()
    missing = OrderedDict()
    for name, genes in gene_dict.items():
        mapped = ordered_unique([mapped_gene(g) for g in genes])
        present = [g for g in mapped if g in available_genes]
        absent = [g for g in mapped if g not in available_genes]
        if present:
            filtered[name] = present
        if absent:
            missing[name] = absent
    return filtered, missing


def zscore_columns(df):
    return df.apply(lambda col: zscore(col, nan_policy="omit"), axis=0).replace([np.inf, -np.inf], np.nan)


def row_normalize(df, fallback_value=None):
    out = df.copy().astype(float)
    if fallback_value is None:
        fallback_value = 1.0 / out.shape[1]
    row_sums = out.sum(axis=1)
    good = row_sums.replace([np.inf, -np.inf], np.nan).notna() & (np.abs(row_sums) > EPS)
    out.loc[good, :] = out.loc[good, :].div(row_sums[good], axis=0)
    out.loc[~good, :] = fallback_value
    return out.fillna(fallback_value)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 2:
        return np.nan
    a = a[good]
    b = b[good]
    if np.std(a) < EPS or np.std(b) < EPS:
        return 1.0 if np.allclose(a, b) else np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# ACINUS ID EXTRACTION AND DATA LOADING
# ============================================================

def extract_acinus_id(label):
    label = str(label)
    parts = label.split("_", 1)
    if len(parts) < 2:
        return None
    tail = parts[1]
    s_idx = tail.find("s")
    if s_idx > 0:
        return tail[:s_idx]
    m = re.search(r"L\d+L\d+a\d+", tail)
    return m.group(0) if m else None


def extract_acinus_id_from_sample(label):
    s = str(label)
    liver = s[:3]
    m = re.search(r"(target[^_]+)", s)
    return f"{liver}{m.group(1)}" if m else None


def minmax_norm_within_group(df, group_col="acinus", x_col="x", out_col="x_norm"):
    x = df[x_col].astype(float)
    gmin = x.groupby(df[group_col]).transform("min")
    gmax = x.groupby(df[group_col]).transform("max")
    denom = (gmax - gmin).replace(0, np.nan)
    df[out_col] = ((x - gmin) / denom).fillna(0.5)
    return df


def load_and_prepare_data(org_path, prot_path):
    org = pd.read_csv(org_path)
    prot = pd.read_csv(prot_path)

    org = org.rename(columns={"labels": "cell_id", "CAT": "category", "position": "x"})
    prot = prot.rename(columns={"labels": "cell_id", "CAT": "category", "position": "x"})

    required_cols = {"cell_id", "category", "x"}
    for name, df in (("sOrganellomics", org), ("proteomics", prot)):
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} data is missing required columns after renaming: {sorted(missing)}")

    org["acinus"] = org["cell_id"].apply(extract_acinus_id)
    prot["acinus"] = prot["cell_id"].apply(extract_acinus_id_from_sample)

    if org["acinus"].isna().any():
        raise ValueError("Some sOrganellomics rows did not receive an acinus ID. Check cell_id format.")
    if prot["acinus"].isna().any():
        raise ValueError("Some proteomics rows did not receive an acinus ID. Check cell_id format.")

    org["category_num"] = org["category"]
    prot["category_num"] = prot["category"]
    org["category"] = org["category"].map(ORG_CAT_TO_H)
    prot["category"] = prot["category"].map(PROT_CAT_TO_H)

    if org["category"].isna().any():
        raise ValueError("Some sOrganellomics CAT values are not present in ORG_CAT_TO_H.")
    if prot["category"].isna().any():
        raise ValueError("Some proteomics CAT values are not present in PROT_CAT_TO_H.")

    if NORMALIZE_X_WITHIN_ACINUS:
        org["x_raw"] = org["x"].astype(float)
        prot["x_raw"] = prot["x"].astype(float)
        org = minmax_norm_within_group(org, group_col="acinus", x_col="x_raw", out_col="x_norm")
        prot = minmax_norm_within_group(prot, group_col="acinus", x_col="x_raw", out_col="x_norm")
        org["x"] = org["x_norm"]
        prot["x"] = prot["x_norm"]
        print("[ok] x replaced with within-acinus min-max normalized coordinate.")

    print(f"sOrganellomics cells: {org.shape[0]}")
    print(f"Proteomics cells:     {prot.shape[0]}")
    print("sOrganellomics categories:", sorted(org["category"].unique()))
    print("Proteomics categories:    ", sorted(prot["category"].unique()))

    return org, prot


# ============================================================
# FEATURE SELECTION AND PROTEOMICS MEANS
# ============================================================

def select_molecular_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata = {"x", "x_raw", "x_norm", "category_num"}
    numeric_cols = [c for c in numeric_cols if c not in metadata]
    col_map = {c.lower(): c for c in numeric_cols}

    if MOLECULAR_OT_MODE == "all":
        genes = numeric_cols
    elif MOLECULAR_OT_MODE == "162":
        genes = [col_map[g.lower()] for g in GENE_LIST_162 if g.lower() in col_map]
    elif MOLECULAR_OT_MODE == "expanded":
        expanded = ordered_unique(
            GENE_LIST_162
            + MITO_SYSTEM_GENES
            + ER_FUNCTION_GENES
            + LIPID_TRAFFICKING_GENES
            + PEROXISOME_EXPANSION_GENES
            + REDOX_GENES
            + landmarks
        )
        genes = [col_map[mapped_gene(g).lower()] for g in expanded if mapped_gene(g).lower() in col_map]
    else:
        raise ValueError(f"Invalid MOLECULAR_OT_MODE: {MOLECULAR_OT_MODE}")

    genes = ordered_unique(genes)
    if not genes:
        raise ValueError(f"No molecular features found for MOLECULAR_OT_MODE={MOLECULAR_OT_MODE!r}")

    print("\n=== Molecular feature selection ===")
    print(f"Mode selected:        {MOLECULAR_OT_MODE}")
    print(f"Numeric protein cols: {len(numeric_cols)}")
    print(f"Proteins selected:    {len(genes)}")
    return genes


def extract_batch(label):
    m = re.search(r"(target[^_]+)", str(label))
    return m.group(1) if m else "unknown"


def compute_batch_corrected_ph_means(prot, prot_feats, ph_cats):
    prot = prot.copy()
    prot["batch"] = prot["cell_id"].apply(extract_batch)

    prot_mat = prot[prot_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if DO_BATCH_LOCATION_CORRECTION:
        global_mean = prot_mat.mean(axis=0)
        batch_mean = prot_mat.groupby(prot["batch"]).transform("mean")
        prot_mat_corrected = prot_mat - batch_mean + global_mean
        print("[ok] Applied batch location correction: X - mean(batch) + mean(global)")
    else:
        prot_mat_corrected = prot_mat.copy()
        print("[ok] Batch location correction skipped.")

    prot_category_means = prot_mat_corrected.groupby(prot["category"]).mean().reindex(ph_cats)
    return prot, prot_mat_corrected, prot_category_means


# ============================================================
# SPATIAL ALIGNMENT
# ============================================================

def build_spatial_distributions(org, prot):
    """Build acinus-by-category x-position distributions for both modalities.

    Acinus IDs are used only as replicate labels within each modality. Because the
    two modalities come from different mice/acini, we do not try to match acinus
    IDs across modalities.
    """
    org_dist = {(a, c): g["x"].astype(float).values for (a, c), g in org.groupby(["acinus", "category"])}
    prot_dist = {(a, c): g["x"].astype(float).values for (a, c), g in prot.groupby(["acinus", "category"])}

    org_categories = sorted(org["category"].unique())
    prot_categories = sorted(prot["category"].unique())

    org_cat_to_acini = {c: [] for c in org_categories}
    prot_cat_to_acini = {c: [] for c in prot_categories}
    for a, c in org_dist:
        org_cat_to_acini[c].append(a)
    for a, c in prot_dist:
        prot_cat_to_acini[c].append(a)

    return org_dist, prot_dist, org_cat_to_acini, prot_cat_to_acini


def compute_wasserstein_matrix(
    org_categories,
    prot_categories,
    org_dist,
    prot_dist,
    org_cat_to_acini,
    prot_cat_to_acini,
    seed=RANDOM_SEED,
):
    """Compute OH-by-PH Wasserstein distances using cross-acini comparisons.

    This is the appropriate mode when sOrganellomics and proteomics come from
    different mice/acini. For the main W matrix, the original RIGOR code compares
    all cross-acini pairs when feasible and samples up to MAX_ACINUS_PAIRS_PER_ENTRY
    otherwise.
    """
    rng = np.random.default_rng(seed)
    print("\n=== Computing spatial Wasserstein matrix: cross-acini across independent mice ===")

    W = np.full((len(org_categories), len(prot_categories)), np.nan, dtype=float)
    N = np.zeros((len(org_categories), len(prot_categories)), dtype=int)

    for i, oc in enumerate(org_categories):
        ao_list = org_cat_to_acini.get(oc, [])
        for j, pc in enumerate(prot_categories):
            ap_list = prot_cat_to_acini.get(pc, [])
            dists = []

            if len(ao_list) == 0 or len(ap_list) == 0:
                pairs = []
            else:
                total_pairs = len(ao_list) * len(ap_list)
                if total_pairs <= MAX_ACINUS_PAIRS_PER_ENTRY:
                    pairs = [(ao, ap) for ao in ao_list for ap in ap_list]
                else:
                    pairs = [
                        (
                            ao_list[int(rng.integers(0, len(ao_list)))],
                            ap_list[int(rng.integers(0, len(ap_list)))],
                        )
                        for _ in range(MAX_ACINUS_PAIRS_PER_ENTRY)
                    ]

            for ao, ap in pairs:
                xo = org_dist.get((ao, oc))
                xp = prot_dist.get((ap, pc))
                if xo is not None and xp is not None:
                    dists.append(wasserstein_distance(xo, xp))

            if dists:
                W[i, j] = float(np.mean(dists))
                N[i, j] = len(dists)

    W_df = pd.DataFrame(W, index=org_categories, columns=prot_categories)
    N_df = pd.DataFrame(N, index=org_categories, columns=prot_categories)
    return W_df, N_df


def hard_match_from_W(W_df):
    W_assign = W_df.copy()
    finite = np.isfinite(W_assign.values)
    if not finite.any():
        raise ValueError("W_df has no finite entries; check x values and category distributions.")
    fill_value = float(np.nanmax(W_assign.values[finite]) * 1.1)
    W_assign = W_assign.fillna(fill_value)
    rows, cols = linear_sum_assignment(W_assign.values)
    matching = pd.DataFrame({
        "sOrganellomics_CAT": W_assign.index.values[rows],
        "Proteomics_CAT": W_assign.columns.values[cols],
        "Wasserstein_distance": W_assign.values[rows, cols],
    })
    return matching


def quantile_interval(df, cat, q=(0.1, 0.9)):
    vals = df.loc[df["category"] == cat, "x"].astype(float).values
    if len(vals) < 2:
        return np.nan, np.nan
    return tuple(np.quantile(vals, q))


def compute_supports_and_consensus(W_df, N_df, org, prot):
    oh_cats = list(W_df.index)
    ph_cats = list(W_df.columns)

    finite_vals = W_df.values[np.isfinite(W_df.values)]
    if finite_vals.size == 0:
        raise ValueError("Cannot compute tau: W_df has no finite entries.")
    tau = float(np.median(finite_vals)) if TAU is None else float(TAU)
    tau = max(tau, EPS)

    S_spatial = np.exp(-W_df / tau)
    S_spatial = S_spatial.where(N_df >= MIN_SUPPORT_PAIRS, 0.0)
    S_spatial = row_normalize(S_spatial)

    org_median_x = org.groupby("category")["x"].median().reindex(oh_cats)
    prot_median_x = prot.groupby("category")["x"].median().reindex(ph_cats)
    org_rank = org_median_x.rank(method="average")
    prot_rank = prot_median_x.rank(method="average")

    rank_support = pd.DataFrame(index=oh_cats, columns=ph_cats, dtype=float)
    for oc in oh_cats:
        for pc in ph_cats:
            rank_support.loc[oc, pc] = 1.0 / (1.0 + abs(org_rank.loc[oc] - prot_rank.loc[pc]))
    rank_support = row_normalize(rank_support)

    org_intervals = {oc: quantile_interval(org, oc) for oc in oh_cats}
    prot_intervals = {pc: quantile_interval(prot, pc) for pc in ph_cats}

    interval_support = pd.DataFrame(index=oh_cats, columns=ph_cats, dtype=float)
    for oc in oh_cats:
        o_lo, o_hi = org_intervals[oc]
        for pc in ph_cats:
            p_lo, p_hi = prot_intervals[pc]
            if np.any(np.isnan([o_lo, o_hi, p_lo, p_hi])):
                interval_support.loc[oc, pc] = 0.0
            else:
                overlap = max(0.0, min(o_hi, p_hi) - max(o_lo, p_lo))
                union = max(o_hi, p_hi) - min(o_lo, p_lo)
                interval_support.loc[oc, pc] = overlap / union if union > 0 else 0.0
    interval_support = row_normalize(interval_support)

    w_spatial = float(CONSENSUS_WEIGHTS.get("spatial", 1.0))
    w_rank = float(CONSENSUS_WEIGHTS.get("rank", 1.0))
    w_interval = float(CONSENSUS_WEIGHTS.get("interval", 1.0))
    total = max(w_spatial + w_rank + w_interval, EPS)

    consensus = (w_spatial * S_spatial + w_rank * rank_support + w_interval * interval_support) / total
    consensus = row_normalize(consensus)

    print(f"tau = {tau:.6g}")
    return tau, S_spatial, rank_support, interval_support, consensus, org_median_x, prot_median_x


def consensus_for_tau(W_df, N_df, tau_value, S_spatial_template, rank_support, interval_support):
    # Match the original RIGOR tau-sensitivity diagnostic: recompute exp(-W/tau)
    # and row-normalize directly, without reapplying the support mask inside the
    # sensitivity loop.
    S_tau = np.exp(-W_df / max(float(tau_value), EPS))
    S_tau = row_normalize(S_tau)

    w_spatial = float(CONSENSUS_WEIGHTS.get("spatial", 1.0))
    w_rank = float(CONSENSUS_WEIGHTS.get("rank", 1.0))
    w_interval = float(CONSENSUS_WEIGHTS.get("interval", 1.0))
    total = max(w_spatial + w_rank + w_interval, EPS)

    C_tau = (w_spatial * S_tau + w_rank * rank_support + w_interval * interval_support) / total
    return row_normalize(C_tau)


# ============================================================
# INFERENCE, PATHWAYS, AND BOOTSTRAPS
# ============================================================

def infer_oh_proteome(consensus, prot_category_means):
    ph_cats = list(consensus.columns)
    prot_category_means = prot_category_means.reindex(ph_cats)
    inferred_vals = consensus.values @ prot_category_means.values
    return pd.DataFrame(inferred_vals, index=consensus.index, columns=prot_category_means.columns)


def bootstrap_inferred_proteome(consensus, prot, prot_mat_corrected, prot_category_means):
    rng = np.random.default_rng(RANDOM_SEED)
    oh_names = consensus.index.tolist()
    ph_names = consensus.columns.tolist()
    gene_names = prot_category_means.columns.tolist()

    ph_to_idx = {ph: np.where(prot["category"].values == ph)[0] for ph in ph_names}
    boot = np.zeros((N_BOOT_INFERENCE, len(oh_names), len(gene_names)), dtype=float)

    for b in range(N_BOOT_INFERENCE):
        boot_means = np.zeros((len(ph_names), len(gene_names)), dtype=float)
        for j, ph in enumerate(ph_names):
            idx = ph_to_idx.get(ph, np.array([], dtype=int))
            if idx.size == 0:
                boot_means[j, :] = np.nan
                continue
            sampled = rng.choice(idx, size=idx.size, replace=True)
            boot_means[j, :] = prot_mat_corrected.iloc[sampled].mean(axis=0).values
        boot[b, :, :] = consensus.values @ boot_means

    low = pd.DataFrame(np.nanpercentile(boot, 2.5, axis=0), index=oh_names, columns=gene_names)
    high = pd.DataFrame(np.nanpercentile(boot, 97.5, axis=0), index=oh_names, columns=gene_names)
    return boot, low, high


def compute_pathway_scores(matrix, pathway_genes):
    scores = pd.DataFrame(index=matrix.index, columns=pathway_genes.keys(), dtype=float)
    for pathway, genes in pathway_genes.items():
        scores[pathway] = matrix[genes].mean(axis=1)
    return scores


def bootstrap_alignment(
    org,
    prot,
    org_categories,
    prot_categories,
    org_dist,
    prot_dist,
    base_tau=None,
):
    """Bootstrap hard matching stability using the original RIGOR cross-acini logic.

    Each bootstrap replicate samples organellomics acini and proteomics acini
    independently with replacement, then compares every sampled organellomics
    acinus to every sampled proteomics acinus for each OH/PH pair. This is the
    same logic used in the original diagnostic block and avoids any within-common-
    acinus branch.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    org_acini = np.array(sorted(org["acinus"].unique()))
    prot_acini = np.array(sorted(prot["acinus"].unique()))

    tau_boot = []
    match_counts = np.zeros((len(org_categories), len(prot_categories)), dtype=float)

    for _ in range(N_BOOT_ALIGNMENT):
        sampled_org = rng.choice(org_acini, size=len(org_acini), replace=True)
        sampled_prot = rng.choice(prot_acini, size=len(prot_acini), replace=True)

        Wb = np.zeros((len(org_categories), len(prot_categories)), dtype=float)

        for i, oc in enumerate(org_categories):
            for j, pc in enumerate(prot_categories):
                dists = []
                for ao in sampled_org:
                    xo = org_dist.get((ao, oc))
                    if xo is None:
                        continue
                    for ap in sampled_prot:
                        xp = prot_dist.get((ap, pc))
                        if xp is None:
                            continue
                        dists.append(wasserstein_distance(xo, xp))

                if dists:
                    Wb[i, j] = float(np.mean(dists))
                else:
                    Wb[i, j] = np.nan

        finite_vals = Wb[np.isfinite(Wb)]
        if finite_vals.size == 0:
            tau_boot.append(float(base_tau) if base_tau is not None else np.nan)
            fill_value = 1.0
        else:
            tau_boot.append(float(np.median(finite_vals)))
            fill_value = float(np.nanmax(finite_vals) * 1.1)

        Wb_filled = np.nan_to_num(Wb, nan=fill_value)
        rows, cols = linear_sum_assignment(Wb_filled)
        for ii, jj in zip(rows, cols):
            match_counts[ii, jj] += 1

    tau_boot = np.array(tau_boot, dtype=float)
    match_prob = match_counts / float(N_BOOT_ALIGNMENT)
    match_prob_df = pd.DataFrame(match_prob, index=org_categories, columns=prot_categories)
    tau_boot_df = pd.DataFrame({"tau": tau_boot})
    return match_prob_df, tau_boot_df


def compute_reconstruction_error(consensus, org, prot):
    # Match original RIGOR diagnostic: allocate samples by int(round(weight * MIX_N)).
    rng = np.random.default_rng(RANDOM_SEED)
    oh_cats = list(consensus.index)
    ph_cats = list(consensus.columns)
    prot_x_by_cat = {pc: prot.loc[prot["category"] == pc, "x"].dropna().values for pc in ph_cats}
    org_x_by_cat = {oc: org.loc[org["category"] == oc, "x"].dropna().values for oc in oh_cats}

    recon = OrderedDict()
    for oc in oh_cats:
        weights = consensus.loc[oc, ph_cats].values.astype(float)
        weights = weights / max(weights.sum(), EPS)
        mix_samples = []

        for pc, weight in zip(ph_cats, weights):
            vals = prot_x_by_cat.get(pc, np.array([]))
            n = int(round(float(weight) * MIX_N_RECONSTRUCTION))
            if n > 0 and vals.size > 0:
                mix_samples.append(rng.choice(vals, size=n, replace=True))

        if mix_samples and org_x_by_cat[oc].size > 0:
            mix = np.concatenate(mix_samples)
            recon[oc] = wasserstein_distance(org_x_by_cat[oc], mix)
        else:
            recon[oc] = np.nan

    return pd.Series(recon, name="reconstruction_wasserstein_distance")


def compute_density_zscores(org, oh_order):
    density_cols = ["mito_density", "peroxisome_density", "ld_density"]
    missing = [c for c in density_cols if c not in org.columns]
    if missing:
        raise ValueError(f"Missing density columns in sOrganellomics data: {missing}")

    means = org.groupby("category")[density_cols].mean().reindex(oh_order)
    return zscore_columns(means)


def compute_net_nitrogen_indices(inferred_proteome):
    available_genes = set(inferred_proteome.columns)
    nitrogen_filtered, missing = filter_gene_dict(NITROGEN_PATHWAYS, available_genes)

    if not nitrogen_filtered:
        raise ValueError("No nitrogen pathway genes were found in inferred_proteome.")

    nitrogen_scores = compute_pathway_scores(inferred_proteome, nitrogen_filtered)
    nitrogen_z = zscore_columns(nitrogen_scores)

    def mean_program(paths):
        valid = [p for p in paths if p in nitrogen_z.columns]
        if not valid:
            return pd.Series(np.nan, index=nitrogen_z.index)
        return nitrogen_z[valid].mean(axis=1)

    aa_breakdown = mean_program(AA_BREAKDOWN_PATHS)
    ammonia_gen = mean_program(AMMONIA_GEN_PATHS)
    urea_cycle = mean_program(UREA_PATHS)
    one_carbon = mean_program(ONE_CARBON_PATHS)
    glutamine_syn = mean_program(GLUTAMINE_SYNTHESIS_PATHS)

    net_indices = pd.DataFrame(index=nitrogen_z.index)
    net_indices["Net_AA_Catabolism"] = aa_breakdown - (urea_cycle + glutamine_syn)
    net_indices["Ammonia_Release_Index"] = ammonia_gen - (urea_cycle + glutamine_syn)
    net_indices["Urea_Release_Bias"] = urea_cycle - glutamine_syn
    net_indices["Glutamine_Release_Bias"] = glutamine_syn - urea_cycle

    return nitrogen_scores, nitrogen_z, net_indices, nitrogen_filtered


# ============================================================
# PLOTS REQUESTED
# ============================================================

def plot_baseline_consensus(consensus):
    """Plot the baseline row-normalized CONSENSUS matrix used for OH proteome inference."""
    plot_df = consensus.loc[sorted(consensus.index), sorted(consensus.columns)]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=PLOT_DPI)
    sns.heatmap(
        plot_df,
        cmap="rocket_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Baseline CONSENSUS OH-to-PH")
    ax.set_ylabel("sOrganellomics category (OH)")
    ax.set_xlabel("Proteomics category (PH)")
    fig.tight_layout()
    finish_plot(fig, "00_baseline_CONSENSUS_OH_to_PH.png")


def plot_ordering_agreement(org_median_x, prot_median_x, consensus):
    expected_ph_x = (consensus * prot_median_x.values).sum(axis=1)
    rho, p_value = spearmanr(org_median_x.values, expected_ph_x.values)

    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=PLOT_DPI)
    for oc in consensus.index:
        x = org_median_x.loc[oc]
        y = expected_ph_x.loc[oc]
        ax.scatter(x, y, s=70)
        ax.text(x + 0.005, y + 0.003, oc, fontsize=9)

    ax.set_xlabel("OH median x")
    ax.set_ylabel("Expected PH median x (CONSENSUS-weighted)")
    ax.set_title(f"Ordering agreement (soft)\nSpearman ρ={rho:.2f}, p={p_value:.1e}")
    fig.tight_layout()
    finish_plot(fig, "01_ordering_agreement_soft.png")

    return pd.Series({"spearman_rho": rho, "p_value": p_value})

def plot_tau_sensitivity(W_df, N_df, tau, consensus, rank_support, interval_support):
    sens = pd.DataFrame(index=consensus.index)
    for mult in TAU_MULTS:
        C_tau = consensus_for_tau(W_df, N_df, tau * mult, None, rank_support, interval_support)
        sens[f"tau_x{mult:g}"] = [safe_corr(consensus.loc[oh], C_tau.loc[oh]) for oh in consensus.index]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=PLOT_DPI)
    sns.heatmap(
        sens,
        vmin=0.9,
        vmax=1.0,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set_title("Sensitivity to τ (corr with baseline)")
    fig.tight_layout()
    finish_plot(fig, "02_tau_sensitivity_corr_with_baseline.png")
    return sens

def plot_hard_matching_stability(match_prob_df):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=PLOT_DPI)
    sns.heatmap(
        match_prob_df,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Hard matching stability (bootstrap frequency)")
    fig.tight_layout()
    finish_plot(fig, "03_hard_matching_stability_bootstrap_frequency.png")

def plot_tau_bootstrap_distribution(tau_boot_df):
    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=PLOT_DPI)
    ax.hist(tau_boot_df["tau"].dropna().values, bins=25)
    ax.set_xlabel("τ (bootstrap median distance)")
    ax.set_ylabel("count")
    ax.set_title("Bootstrap distribution of τ")
    fig.tight_layout()
    finish_plot(fig, "04_bootstrap_tau_distribution.png")

def plot_pathway_zscore_heatmaps(oh_zscores, ph_zscores):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=PLOT_DPI)
    sns.heatmap(
        oh_zscores.loc[sorted(oh_zscores.index)],
        cmap=ZSCORE_CMAP,
        center=0,
        vmin=ZSCORE_VMIN,
        vmax=ZSCORE_VMAX,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(f"Inferred OH Pathway Enrichment (Z-score)\n(Per pathway; OT mode: {MOLECULAR_OT_MODE})")
    ax.set_ylabel("sOrganellomics category (OH)")
    ax.set_xlabel("Metabolic / Organelle pathway")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    finish_plot(fig, "05_inferred_OH_pathway_enrichment_zscore.png")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=PLOT_DPI)
    sns.heatmap(
        ph_zscores.loc[sorted(ph_zscores.index)],
        cmap=ZSCORE_CMAP,
        center=0,
        vmin=ZSCORE_VMIN,
        vmax=ZSCORE_VMAX,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Proteomics PH Pathway Enrichment (Z-score)\n(Per pathway)")
    ax.set_ylabel("Proteomics category (PH)")
    ax.set_xlabel("Metabolic / Organelle pathway")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    finish_plot(fig, "06_proteomics_PH_pathway_enrichment_zscore.png")

def plot_reconstruction_error(recon_error):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=PLOT_DPI)
    ax.bar(recon_error.index, recon_error.values)
    ax.set_ylabel("Wasserstein distance (normalized x)")
    ax.set_title("Distribution reconstruction error\n(lower is better)")
    fig.tight_layout()
    finish_plot(fig, "07_distribution_reconstruction_error.png")

def plot_density_pair(title, oh_order, density_y, pathway_y, color, density_label, pathway_label, filename):
    x = np.arange(len(oh_order))
    fig, ax = plt.subplots(figsize=(2, 7), dpi=300)

    ax.plot(x, density_y, color=color, linewidth=2, alpha=0.8, label=density_label)
    ax.scatter(x, density_y, color="black", s=20, zorder=3)
    ax.plot(x, pathway_y, color=color, linestyle=":", linewidth=2, alpha=0.8, label=pathway_label)
    ax.scatter(x, pathway_y, color="black", s=20, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(oh_order)
    ax.set_ylim(-2, 2)
    ax.set_ylabel("Z-score")
    ax.set_xlabel("sOrganellomics category (OH)")
    ax.set_title(title)
    fig.tight_layout()
    finish_plot(fig, filename)

def plot_density_vs_inferred(density_z, oh_zscores):
    oh_order = sorted(density_z.index)

    specs = [
        (
            "Mitochondria: Density vs Inferred Program",
            "mito_density",
            "Mitochondrial_import_biogenesis",
            "green",
            "Mito density",
            "Mitochondrial import/biogenesis",
            "08_density_vs_inferred_mitochondria.png",
        ),
        (
            "Peroxisome: Density vs Inferred Program",
            "peroxisome_density",
            "Peroxisome_biogenesis_maintenance",
            "magenta",
            "Peroxisome density",
            "Peroxisome biogenesis/maintenance",
            "09_density_vs_inferred_peroxisome.png",
        ),
        (
            "Lipid Droplets: Density vs Inferred Program",
            "ld_density",
            "Lipid_droplet_accumulation",
            "#D4A017",
            "LD density",
            "LD accumulation program",
            "10_density_vs_inferred_lipid_droplets.png",
        ),
    ]

    for title, density_col, pathway, color, density_label, pathway_label, filename in specs:
        if pathway not in oh_zscores.columns:
            print(f"[warn] Skipping {title} because pathway {pathway!r} is absent.")
            continue
        plot_density_pair(
            title=title,
            oh_order=oh_order,
            density_y=density_z.loc[oh_order, density_col].values,
            pathway_y=oh_zscores.loc[oh_order, pathway].values,
            color=color,
            density_label=density_label,
            pathway_label=pathway_label,
            filename=filename,
        )


def plot_net_nitrogen_bubble_panel(net_indices):
    ordered_oh = sorted(net_indices.index)
    ordered_indices = net_indices.columns.tolist()
    values_all = net_indices.loc[ordered_oh, ordered_indices].values.astype(float)
    global_abs = np.nanmax(np.abs(values_all))
    global_abs = max(global_abs, EPS)

    fig, ax = plt.subplots(figsize=(4.3, 4.5), dpi=200)
    size_scale = 1200

    for i, col in enumerate(ordered_indices):
        values = net_indices.loc[ordered_oh, col].values.astype(float)
        y_positions = np.arange(len(ordered_oh))
        sizes = np.abs(values) / global_abs * size_scale
        ax.scatter(
            np.full_like(values, i, dtype=float),
            y_positions,
            s=sizes,
            c=values,
            cmap="RdGy_r",
            vmin=-global_abs,
            vmax=global_abs,
            edgecolor="black",
            linewidth=1.0,
        )

    ax.set_xticks(range(len(ordered_indices)))
    ax.set_xticklabels([c.replace("_", " ") for c in ordered_indices], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ordered_oh)))
    ax.set_yticklabels(ordered_oh)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(ordered_indices) - 0.4)
    ax.set_ylim(-0.7, len(ordered_oh) - 0.3)
    ax.set_xlabel("Functional Index")
    ax.set_ylabel("sOrganellomics Category (OH)")
    ax.set_title("Net Nitrogen Functional Indices")

    sm = plt.cm.ScalarMappable(cmap="RdGy_r", norm=plt.Normalize(vmin=-global_abs, vmax=global_abs))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Index Value (Z-score based)")

    fig.tight_layout()
    finish_plot(fig, "11_net_nitrogen_functional_indices_bubble_panel.png")

# ============================================================
# COMMAND-LINE OVERRIDES
# ============================================================

def str_to_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_cli_overrides():
    parser = argparse.ArgumentParser(
        description="Cross-modal sOrganellomics-to-proteomics inference pipeline."
    )
    parser.add_argument(
        "--batch-correction",
        type=str_to_bool,
        default=None,
        help=(
            "Override DO_BATCH_LOCATION_CORRECTION. Use true to apply "
            "X - mean(batch) + mean(global); use false to infer from raw proteomics values."
        ),
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[info] Ignoring unrelated command-line arguments: {unknown}")
    return args


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    sns.set_context("paper")

    args = parse_cli_overrides()
    if args.batch_correction is not None:
        DO_BATCH_LOCATION_CORRECTION = args.batch_correction

    batch_mode_label = "batch_corrected" if DO_BATCH_LOCATION_CORRECTION else "no_batch_correction"

    OUTDIR = make_outdir(ORG_PATH, OUTDIR)
    if APPEND_BATCH_MODE_TO_OUTDIR:
        OUTDIR = os.path.join(OUTDIR, batch_mode_label)
        os.makedirs(OUTDIR, exist_ok=True)

    print(f"Results directory: {OUTDIR}")
    print(f"Batch location correction: {DO_BATCH_LOCATION_CORRECTION} ({batch_mode_label})")

    org, prot = load_and_prepare_data(ORG_PATH, PROT_PATH)
    org_categories = sorted(org["category"].unique())
    prot_categories = sorted(prot["category"].unique())

    org_dist, prot_dist, org_cat_to_acini, prot_cat_to_acini = build_spatial_distributions(org, prot)

    W_df, N_df = compute_wasserstein_matrix(
        org_categories,
        prot_categories,
        org_dist,
        prot_dist,
        org_cat_to_acini,
        prot_cat_to_acini,
    )
    matching = hard_match_from_W(W_df)

    save_table(W_df, "Wasserstein_distance_matrix.csv")
    save_table(N_df, "Wasserstein_support_counts.csv")
    save_table(matching, "Hard_Hungarian_matching.csv")

    tau, S_spatial, rank_support, interval_support, CONSENSUS, org_median_x, prot_median_x = compute_supports_and_consensus(W_df, N_df, org, prot)

    save_table(S_spatial, "Spatial_support_row_normalized.csv")
    save_table(rank_support, "Rank_support_row_normalized.csv")
    save_table(interval_support, "Interval_support_row_normalized.csv")
    save_table(CONSENSUS, "CONSENSUS_OH_to_PH.csv")
    plot_baseline_consensus(CONSENSUS)

    prot_feats = select_molecular_features(prot)
    prot, prot_mat_corrected, prot_category_means = compute_batch_corrected_ph_means(prot, prot_feats, prot_categories)
    inferred_proteome = infer_oh_proteome(CONSENSUS, prot_category_means)

    save_table(prot_category_means, f"Proteomics_PH_category_means_{MOLECULAR_OT_MODE}.csv")
    save_table(inferred_proteome, f"Inferred_proteomes_by_sOrganellomics_category_{MOLECULAR_OT_MODE}.csv")

    if DO_BOOTSTRAP_INFERENCE:
        boot_inferred, ci_low, ci_high = bootstrap_inferred_proteome(CONSENSUS, prot, prot_mat_corrected, prot_category_means)
        save_table(ci_low, f"InferredProteome_CI_low_{MOLECULAR_OT_MODE}.csv")
        save_table(ci_high, f"InferredProteome_CI_high_{MOLECULAR_OT_MODE}.csv")

    pathway_genes_filtered, missing_pathway_genes = filter_gene_dict(PATHWAY_GENES, set(inferred_proteome.columns))
    print("\nPathways retained after filtering:")
    for pathway, genes in pathway_genes_filtered.items():
        print(f"  {pathway}: {len(genes)} genes")

    oh_pathway_scores = compute_pathway_scores(inferred_proteome, pathway_genes_filtered)
    ph_pathway_scores = compute_pathway_scores(prot_category_means, pathway_genes_filtered)

    # Match original RIGOR plotting logic: compute relative contributions per pathway,
    # then z-score those contributions across OH/PH categories.
    oh_pathway_contributions = oh_pathway_scores.div(oh_pathway_scores.sum(axis=0).replace(0, np.nan), axis=1)
    ph_pathway_contributions = ph_pathway_scores.div(ph_pathway_scores.sum(axis=0).replace(0, np.nan), axis=1)
    oh_zscores = zscore_columns(oh_pathway_contributions)
    ph_zscores = zscore_columns(ph_pathway_contributions)

    save_table(oh_pathway_scores, f"OH_pathway_scores_{MOLECULAR_OT_MODE}.csv")
    save_table(ph_pathway_scores, f"PH_pathway_scores_{MOLECULAR_OT_MODE}.csv")
    save_table(oh_pathway_contributions, f"OH_pathway_contributions_{MOLECULAR_OT_MODE}.csv")
    save_table(ph_pathway_contributions, f"PH_pathway_contributions_{MOLECULAR_OT_MODE}.csv")
    save_table(oh_zscores, f"OH_pathway_enrichment_zscores_{MOLECULAR_OT_MODE}.csv")
    save_table(ph_zscores, f"PH_pathway_enrichment_zscores_{MOLECULAR_OT_MODE}.csv")

    ordering_stats = plot_ordering_agreement(org_median_x, prot_median_x, CONSENSUS)
    ordering_stats.to_csv(os.path.join(OUTDIR, "Ordering_agreement_soft_stats.csv"))

    tau_sensitivity = plot_tau_sensitivity(W_df, N_df, tau, CONSENSUS, rank_support, interval_support)
    save_table(tau_sensitivity, "Tau_sensitivity_corr_with_baseline.csv")

    match_prob_df, tau_boot_df = bootstrap_alignment(
        org,
        prot,
        org_categories,
        prot_categories,
        org_dist,
        prot_dist,
        base_tau=tau,
    )
    save_table(match_prob_df, "Hard_matching_bootstrap_frequency.csv")
    save_table(tau_boot_df, "Tau_bootstrap_distribution.csv")
    plot_hard_matching_stability(match_prob_df)
    plot_tau_bootstrap_distribution(tau_boot_df)

    plot_pathway_zscore_heatmaps(oh_zscores, ph_zscores)

    recon_error = compute_reconstruction_error(CONSENSUS, org, prot)
    recon_error.to_csv(os.path.join(OUTDIR, "Distribution_reconstruction_error.csv"), header=True)
    plot_reconstruction_error(recon_error)

    density_z = compute_density_zscores(org, sorted(CONSENSUS.index))
    save_table(density_z, "Organelle_density_zscores_by_OH.csv")
    plot_density_vs_inferred(density_z, oh_zscores)

    nitrogen_scores, nitrogen_z, net_indices, nitrogen_filtered = compute_net_nitrogen_indices(inferred_proteome)
    save_table(nitrogen_scores, "Nitrogen_pathway_scores.csv")
    save_table(nitrogen_z, "Nitrogen_pathway_zscores.csv")
    save_table(net_indices, "Net_nitrogen_functional_indices.csv")
    plot_net_nitrogen_bubble_panel(net_indices)

    params = pd.Series({
        "MOLECULAR_OT_MODE": MOLECULAR_OT_MODE,
        "NORMALIZE_X_WITHIN_ACINUS": NORMALIZE_X_WITHIN_ACINUS,
        "ALIGNMENT_MODE": ALIGNMENT_MODE,
        "MAX_ACINUS_PAIRS_PER_ENTRY": MAX_ACINUS_PAIRS_PER_ENTRY,
        "MIN_SUPPORT_PAIRS": MIN_SUPPORT_PAIRS,
        "tau": tau,
        "CONSENSUS_WEIGHTS": str(CONSENSUS_WEIGHTS),
        "DO_BATCH_LOCATION_CORRECTION": DO_BATCH_LOCATION_CORRECTION,
        "BATCH_MODE_LABEL": batch_mode_label,
        "APPEND_BATCH_MODE_TO_OUTDIR": APPEND_BATCH_MODE_TO_OUTDIR,
        "DO_BOOTSTRAP_INFERENCE": DO_BOOTSTRAP_INFERENCE,
        "N_BOOT_INFERENCE": N_BOOT_INFERENCE,
        "N_BOOT_ALIGNMENT": N_BOOT_ALIGNMENT,
        "ZSCORE_CMAP": ZSCORE_CMAP,
        "ZSCORE_VMIN": ZSCORE_VMIN,
        "ZSCORE_VMAX": ZSCORE_VMAX,
    })
    params.to_csv(os.path.join(OUTDIR, "Pipeline_parameters.csv"), header=False)
    print("\n[done] Cross-modal inference pipeline complete.")
