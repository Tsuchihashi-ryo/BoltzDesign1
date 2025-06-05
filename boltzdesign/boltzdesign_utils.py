import os
import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional
import copy
import random
from boltz.data import const
from boltz.data.types import MSA, Connection, Input, Structure, Interface
from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
import yaml
import shutil
from Bio.PDB import PDBParser, MMCIFParser  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import csv
import gc
import json
import logging

logging.basicConfig(level=logging.WARNING)

def save_confidence_scores(folder_dir, output, structure,name, model_idx=0):
    output_dir = os.path.join(folder_dir, f"boltz_results_{name}", "predictions", name)

    os.makedirs(output_dir, exist_ok=True)
    atoms = structure.atoms
    atoms['coords'] = output['coords'][0].detach().cpu().numpy()[:atoms['coords'].shape[0],:]
    atoms["is_present"] = True
    residues = structure.residues
    residues["is_present"] = True
    interfaces = np.array([], dtype=Interface)
    new_structure: Structure = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
    )
    plddts= output['plddt'].detach().cpu().numpy()[0]        
    path = Path(output_dir) / f"{name}_model_{model_idx}.cif"
    with path.open("w") as f:
        f.write(to_mmcif(new_structure, plddts=plddts))

    # Save confidence summary
    if "plddt" in output:
        confidence_summary_dict = {}
        for key in [
            "confidence_score",
            "ptm", 
            "iptm",
            "ligand_iptm",
            "protein_iptm",
            "complex_plddt",
            "complex_iplddt", 
            "complex_pde",
            "complex_ipde",
        ]:
            if key in output:
                confidence_summary_dict[key] = output[key].item()
        
        if "pair_chains_iptm" in output:
            confidence_summary_dict["chains_ptm"] = {
                idx: output["pair_chains_iptm"][idx][idx].item()
                for idx in output["pair_chains_iptm"]
            }
            confidence_summary_dict["pair_chains_iptm"] = {
                idx1: {
                    idx2: output["pair_chains_iptm"][idx1][idx2].item()
                    for idx2 in output["pair_chains_iptm"][idx1]
                }
                for idx1 in output["pair_chains_iptm"]
            }

        json_path = os.path.join(output_dir, f"confidence_{name}_model_{model_idx}.json")
        with open(json_path, 'w') as f:
            json.dump(confidence_summary_dict, f, indent=4)
        # Save plddt
        plddt = output["plddt"]
        plddt_path = os.path.join(output_dir, f"plddt_{name}_model_{model_idx}.npz")
        np.savez_compressed(plddt_path, plddt=plddt.cpu().detach().numpy())

    if "pae" in output:
        pae = output["pae"]
        pae_path = os.path.join(output_dir, f"pae_{name}_model_{model_idx}.npz")
        np.savez_compressed(pae_path, pae=pae.cpu().detach().numpy())


tokens = [
    "<pad>",
    "-",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]


chain_to_number = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
}
def visualize_training_history(best_batch, loss_history, sequence_history, distogram_history, length, binder_chain='A', save_dir=None, save_filename=None):
    """
    Visualize training history including loss plot, distogram animation, and sequence evolution animation.
    Args:
        loss_history (list): List of loss values over training
        sequence_history (list): List of sequence probability matrices over training
        distogram_history (list): List of distogram matrices over training
        length (int): Length of sequence to visualize
        save_dir (str): Directory to save visualizations
    """

    mask = (best_batch['entity_id']==chain_to_number[binder_chain]).squeeze(0).detach().cpu().numpy()
    sequence_history = [seq[mask] for seq in sequence_history]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


    def create_distogram_animation():
        plt.style.use('default')  # Use default white background style
        fig, ax = plt.subplots(figsize=(6,6))
        distogram_2d = distogram_history[0]
        im = ax.imshow(distogram_2d)
    
        plt.colorbar(im, ax=ax)
        ax.set_title('Distogram Evolution')

        def update(frame):
            distogram_2d = distogram_history[frame]
            im.set_data(distogram_2d)
            ax.set_title(f'Distogram Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(distogram_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_distogram_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Create sequence evolution animation
    def create_sequence_animation():
        plt.style.use('default')  # Use default white background style
        fig, ax = plt.subplots(figsize=(12,3.5))
        im = ax.imshow(sequence_history[0].T, vmin=0, vmax=1, cmap='Blues', aspect='auto', alpha=0.8)
        plt.colorbar(im, ax=ax)
        ax.set_yticks(np.arange(20))
        ax.set_yticklabels(list('ARNDCQEGHILKMFPSTWYV'))
        ax.set_title('Sequence Evolution')

        def update(frame):
            im.set_data(sequence_history[frame].T)
            ax.set_title(f'Sequence Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(sequence_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_sequence_evolution.gif'), writer='pillow')
        plt.close()
        return ani

    # Create and save animations
    distogram_ani = create_distogram_animation()
    sequence_ani = create_sequence_animation()

    return distogram_ani, sequence_ani

def get_mid_points(pdistogram):
    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
        pdistogram.device
    )

    return mid_points


def get_CA_and_sequence(structure_file, chain_id='A'):
    # Determine file type and use appropriate parser
    if structure_file.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("File must be either .cif or .pdb format")
        
    structure = parser.get_structure("structure", structure_file)
    xyz = []
    sequence = []
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    model = structure[0]  # Get first model (default for most structures)
    
    if chain_id in model:
        chain = model[chain_id]
        for residue in chain:
            if "CA" in residue:
                xyz.append(residue["CA"].coord)
                sequence.append(aa_map.get(residue.resname, 'X'))
    else:
        raise ValueError(f"Chain {chain_id} not found in {structure_file}")
    
    return xyz, sequence


def np_kabsch(a, b, return_v=False):
    '''Get alignment matrix for two sets of coordinates using numpy
    
    Args:
        a: First set of coordinates
        b: Second set of coordinates
        return_v: If True, return U matrix from SVD. If False, return rotation matrix
        
    Returns:
        Rotation matrix (or U matrix if return_v=True) to align coordinates
    '''
    # Calculate covariance matrix
    ab = np.swapaxes(a, -1, -2) @ b
    
    # Singular value decomposition
    u, s, vh = np.linalg.svd(ab, full_matrices=False)
    
    # Handle reflection case
    flip = np.linalg.det(u @ vh) < 0
    if flip:
        u[...,-1] = -u[...,-1]
    
    return u if return_v else (u @ vh)


def align_points(a, b):
    a_centroid = a.mean(axis=0)
    b_centroid = b.mean(axis=0)

    a_centered = a - a_centroid
    b_centered = b - b_centroid

    R = np_kabsch(a_centered, b_centered)
    a_aligned = a_centered @ R + b_centroid
    return a_aligned


def np_rmsd(true, pred):
    '''Compute RMSD of coordinates after alignment using numpy
    
    Args:
        true: Reference coordinates
        pred: Predicted coordinates to align
        
    Returns:
        Root mean square deviation after optimal alignment
    '''
    # Center coordinates
    p = true - np.mean(true, axis=-2, keepdims=True)
    q = pred - np.mean(pred, axis=-2, keepdims=True)
    
    # Get optimal rotation matrix and apply it
    p = p @ np_kabsch(p, q)
    
    # Calculate RMSD
    return np.sqrt(np.mean(np.sum(np.square(p-q), axis=-1)) + 1e-8)

    
def min_k(x, k=1, mask=None):
    # Convert mask to boolean if it's not None
    if mask is not None:
        mask = mask.bool()  # Convert to boolean tensor
    
    # Sort the tensor, replacing masked values with Nan
    y = torch.sort(x if mask is None else torch.where(mask, x, float('nan')))[0]

    # Create a mask for the top k value
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    # Compute the mean of the top k values
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def get_con_loss(dgram, dgram_bins, num=None, seqsep=None, num_pos = float("inf"), cutoff=None, binary=False, mask_1d=None, mask_1b=None):
    con_loss = _get_con_loss(dgram, dgram_bins, cutoff, binary)
    idx = torch.arange(dgram.shape[1])
    offset = idx[:,None] - idx[None,:]
    # Add mask for position separation > 3
    m =(torch.abs(offset)>=seqsep).to(dgram.device)
    if mask_1d is None: mask_1d = torch.ones(m.shape[0])
    if mask_1b is None: mask_1b = torch.ones(m.shape[0])

    m = torch.logical_and(m, mask_1b)
    p = min_k(con_loss, num, m).to(dgram.device)
    p = min_k(p, num_pos, mask_1d).to(dgram.device)
    return p


def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=False):
    '''dgram to contacts'''
    if cutoff is None: cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff  
    px = torch.softmax(dgram, dim=-1)
    px_ = torch.softmax(dgram - 1e7 * (~ bins), dim=-1)        
    # binary/categorical cross-entropy
    con_loss_cat_ent = -(px_ * torch.log_softmax(dgram, dim=-1)).sum(-1)
    con_loss_bin_ent = -torch.log((bins * px + 1e-8).sum(-1))

    return binary * con_loss_bin_ent + (1 - binary) * con_loss_cat_ent


def mask_loss(x, mask=None, mask_grad=False):
    if mask is None:
        return x.mean()
    else:
        x_masked = (x * mask).sum() / (1e-8 + mask.sum())
        if mask_grad:
            return (x.mean() - x_masked).detach() + x_masked
        else:
            return x_masked


def get_plddt_loss(plddt, mask_1d=None):
    p = 1 - plddt
    return mask_loss(p, mask_1d)


def get_pae_loss(pae, mask_1d=None, mask_1b=None, mask_2d=None):
  pae = pae/31.0
  L = pae.shape[1]
  if mask_1d is None: mask_1d = torch.ones(L).to(pae.device)
  if mask_1b is None: mask_1b = torch.ones(L).to(pae.device)
  if mask_2d is None: mask_2d = torch.ones((L, L)).to(pae.device)
  mask_2d = mask_2d * mask_1d[:, :, None] * mask_1b[:, None, :]
  return mask_loss(pae, mask_2d)


def _get_helix_loss(dgram, dgram_bins, offset=None, mask_2d=None, binary=False, **kwargs):
    '''helix bias loss'''
    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)
    if offset is None:
        if mask_2d is None:
            return x.diagonal(offset=3).mean()
        else:
            mask_2d = mask_2d.float() 
            return (x * mask_2d).diagonal(offset=3, dim1=-2, dim2=-1).sum() / (torch.diagonal(mask_2d, offset=3, dim1=-2, dim2=-1).sum() + 1e-8)

    else:
        mask = (offset == 3).float()
        if mask_2d is not None:
            mask = mask * mask_2d.float()
        return (x * mask).sum() / (mask.sum() + 1e-8)


def get_ca_coords(sample_atom_coords, batch, binder_chain='A'):
    atom_to_token = batch['atom_to_token'] * (batch['entity_id']==chain_to_number[binder_chain])
    atom_order = torch.cumsum(atom_to_token, dim=1)
    ca_mask = torch.sum((atom_order == 2).to(atom_to_token.dtype), dim=-1)[0]
    ca_coords = sample_atom_coords[:,ca_mask==1,:]
    return ca_coords


def add_rg_loss(sample_atom_coords, batch, length, binder_chain='A'):
    ca_coords = get_ca_coords(sample_atom_coords, batch, binder_chain)
    center_of_mass = ca_coords.mean(1, keepdim=True)  # keepdim for proper broadcasting
    squared_distances = torch.sum(torch.square(ca_coords - center_of_mass), dim=-1)
    rg = torch.sqrt(squared_distances.mean() + 1e-8)
    rg_th = 2.38 * ca_coords.shape[1] ** 0.365
    loss = torch.nn.functional.elu(rg - rg_th)
    return loss, rg


def get_boltz_model(checkpoint: Optional[str] = None, predict_args=None, device: Optional[str] = None) -> Boltz1:
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision("highest")
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638  # Default value
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False, 
    )
    return model_module



def boltz_hallucination(
    # Required arguments
    boltz_model,
    yaml_path,
    ccd_lib,
    length=100,
    binder_chain='A',
    design_algorithm="3stages",
    recycling_steps=0,
    pre_iteration=20,
    soft_iteration=50, 
    soft_iteration_1=50,
    soft_iteration_2=25,
    temp_iteration=50,
    hard_iteration=10,
    semi_greedy_steps=0,
    learning_rate=0.1,
    learning_rate_pre=0.1,
    inter_chain_cutoff=21.0,
    intra_chain_cutoff=14.0,
    num_inter_contacts=2,
    num_intra_contacts=4,
    e_soft=0.8,
    e_soft_1=0.8,
    e_soft_2=1.0,
    alpha=2.0,
    pre_run=False,
    set_train=True,
    use_temp=False,
    disconnect_feats=False,
    disconnect_pairformer=False,
    mask_ligand=False,
    distogram_only=False,
    input_res_type=False,
    non_protein_target=False,
    increasing_contact_over_itr=False,
    loss_scales=None,
    optimize_contact_per_binder_pos=False,
    pocket_conditioning=False,
    chain_to_number=None,
    msa_max_seqs=4096,
    optimizer_type='SGD',
    save_trajectory=False
):

    predict_args = {
        "recycling_steps": recycling_steps,  # Default value
        "sampling_steps": 200,  # Default value
        "diffusion_samples": 1,  # Default value
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    with yaml_path.open("r") as file:
        data = yaml.safe_load(file)

    data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = 'X'*length
    name = yaml_path.stem
    target = parse_boltz_schema(name, data, ccd_lib)

    fixed_residues_info = getattr(target, 'fixed_residues', [])
    fixed_binder_residues_map = {}
    if fixed_residues_info:
        # Ensure chain_to_number dict is available (it's a function argument)
        binder_chain_numeric_id_lookup = chain_to_number.get(binder_chain)
        if binder_chain_numeric_id_lookup is not None:
            for fr_info in fixed_residues_info:
                if fr_info['chain_asym_id'] == binder_chain_numeric_id_lookup: # Compare with the numeric ID from chain_to_number
                    fixed_binder_residues_map[fr_info['res_idx_in_entity']] = fr_info['aa']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    boltz_model.train() if set_train else boltz_model.eval()
    print(f"set in {'train' if set_train else 'eval'} mode")

    def get_batch(target, max_seqs=0, length=100, pocket_conditioning=False, keep_record=False):
        target_id = target.record.id
        structure = target.structure

        structure = Structure(
                atoms=structure.atoms,
                bonds=structure.bonds,
                residues=structure.residues,
                chains=structure.chains,
                connections=structure.connections.astype(Connection),
                interfaces=structure.interfaces,
                mask=structure.mask,
            )

        msas = {}
        for chain in target.record.chains:
            msa_id = chain.msa_id
            if msa_id != -1:
                msa = np.load(msa_id)
                msas[chain.chain_id] = MSA(**msa)

        input = Input(structure, msas) 
        
        tokenizer = BoltzTokenizer()
        tokenized = tokenizer.tokenize(input)
        featurizer = BoltzFeaturizer()

        if pocket_conditioning:
            options = target.record.inference_options
            binders, pocket = options.binders, options.pocket  
            batch = featurizer.process(
                        tokenized,
                        training=False,
                        max_atoms=None,
                        max_tokens=None,
                        max_seqs=max_seqs,
                        pad_to_max_seqs=False,
                        symmetries={},
                        compute_symmetries=False,
                        inference_binder=binders,
                        inference_pocket=pocket,
                    )
        else:
            batch = featurizer.process(
                        tokenized,
                        training=False,
                        max_atoms=None,
                        max_tokens=None,
                        max_seqs=max_seqs,
                        pad_to_max_seqs=False,
                        symmetries={},
                        compute_symmetries=False,
                        inference_binder=None,
                        inference_pocket=None,
                    )

        if keep_record:
            batch['record'] = target.record

        return batch, structure
    
    batch, structure = get_batch(target, max_seqs=msa_max_seqs, length=length, pocket_conditioning=pocket_conditioning)
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
    
    ## initialize res_type_logits
    if pre_run:
        batch['res_type_logits'] = batch['res_type'].clone().detach().to(device).float()
        batch['res_type_logits'][batch['entity_id']==chain_to_number[binder_chain],:] = torch.softmax(torch.distributions.Gumbel(0, 1).sample(batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:].shape).to(device) - torch.sum(torch.eye(batch['res_type'].shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0).to(device)*(1e10), dim=-1)

    else:
        batch['res_type_logits'] = torch.from_numpy(input_res_type).to(device)

    # Default masks in case binder chain info is not found or no fixed residues
    # Determine a fallback length if needed, e.g., from the 'length' param or first chain if possible
    # For now, let's assume 'length' param is a safe fallback for binder length if more specific info isn't available yet.
    # However, it's better to derive from actual batch data.

    binder_chain_numeric_id = chain_to_number.get(binder_chain)
    actual_binder_length = 0 # Default to 0, will be updated

    # Determine actual_binder_length safely
    if binder_chain_numeric_id is not None and 'entity_id' in batch and 'res_type_logits' in batch :
        is_binder_chain_mask_flat_for_len = (batch['entity_id'].squeeze(0) == binder_chain_numeric_id)
        if torch.any(is_binder_chain_mask_flat_for_len):
            actual_binder_length = batch['res_type_logits'][0, is_binder_chain_mask_flat_for_len, :].shape[0]
        else:
            actual_binder_length = length # Fallback to 'length' parameter
            # print(f"Warning: Boltzdesign utils - Binder chain {binder_chain} not found in batch for length calculation, falling back to param length: {length}.")
    else:
        actual_binder_length = length # Fallback to 'length' parameter
        # print(f"Warning: Boltzdesign utils - Binder chain ID or batch components missing for length calculation, falling back to param length: {length}.")

    fixed_mask_1d = torch.zeros(actual_binder_length, device=device, dtype=torch.bool)
    variable_mask_1d = torch.ones(actual_binder_length, device=device, dtype=torch.bool)

    if fixed_binder_residues_map and binder_chain_numeric_id is not None and 'entity_id' in batch and 'res_type_logits' in batch:
        is_binder_chain_mask_flat = (batch['entity_id'].squeeze(0) == binder_chain_numeric_id)

        if torch.any(is_binder_chain_mask_flat):
            binder_logits_view = batch['res_type_logits'][0, is_binder_chain_mask_flat, :]
            # This might be a clone already if the next line is uncommented, or a view.
            # To be safe, operate on a clone if modifications are extensive, then assign back.
            cloned_binder_logits = binder_logits_view.clone()

            current_actual_binder_length = cloned_binder_logits.shape[0]
            # Re-initialize masks if actual_binder_length was different from initial estimate
            if current_actual_binder_length != actual_binder_length:
                 # print(f"Info: Boltzdesign utils - Adjusting binder mask length from {actual_binder_length} to {current_actual_binder_length}.")
                 actual_binder_length = current_actual_binder_length
                 fixed_mask_1d = torch.zeros(actual_binder_length, device=device, dtype=torch.bool)
                 variable_mask_1d = torch.ones(actual_binder_length, device=device, dtype=torch.bool)

            for res_idx_in_binder, aa_char in fixed_binder_residues_map.items():
                if 0 <= res_idx_in_binder < actual_binder_length:
                    # Ensure const.prot_char_to_token_map is available or define it.
                    # Assuming it's like {'A': 'ALA', 'C': 'CYS', ...}
                    aa_3_letter = const.prot_char_to_token_map.get(aa_char.upper())
                    if aa_3_letter and aa_3_letter in tokens:
                        token_idx = tokens.index(aa_3_letter)
                        cloned_binder_logits[res_idx_in_binder, :] = -10.0
                        cloned_binder_logits[res_idx_in_binder, token_idx] = 10.0

                        fixed_mask_1d[res_idx_in_binder] = True
                        variable_mask_1d[res_idx_in_binder] = False
                    else:
                        print(f"Warning: Boltzdesign utils - Could not map fixed aa {aa_char} (3-letter: {aa_3_letter}) to token index.")

            batch['res_type_logits'][0, is_binder_chain_mask_flat, :] = cloned_binder_logits
        else:
            # This else implies binder chain ID was valid but not found in entity_id tensor.
            # Mask will remain default (all variable).
            print(f"Warning: Boltzdesign utils - Binder chain {binder_chain} (ID: {binder_chain_numeric_id}) not found in batch['entity_id'] after check. Masks will be default for length {actual_binder_length}.")
    # else:
        # This else implies fixed_binder_residues_map is empty, or binder_chain_numeric_id is None, or batch is incomplete.
        # Masks will remain default.
        # print(f"Info: Boltzdesign utils - No fixed residues to apply or binder chain ID invalid. Masks are default for length {actual_binder_length}.")

    batch['fixed_mask_binder_1d'] = fixed_mask_1d
    batch['variable_mask_binder_1d'] = variable_mask_1d

    if  non_protein_target:
        batch['msa'] = batch['res_type_logits'].unsqueeze(0).to(device)
        batch['msa_paired'] = torch.ones(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
        batch['deletion_value'] = torch.zeros(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
        batch['has_deletion'] = torch.full((batch['res_type'].shape[0], 1, batch['res_type'].shape[1]), False).to(device)  
        batch['msa_mask'] = torch.ones(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
        batch['profile'] = batch['msa'].float().mean(dim=0).to(device)
        batch['deletion_mean'] = torch.zeros(batch['deletion_mean'].shape).to(device)
        batch['res_type'] = batch['res_type'].float()

    batch['res_type_logits'].requires_grad = True
    optimizer = torch.optim.AdamW([batch['res_type_logits']], lr=learning_rate_pre if pre_run else learning_rate) if optimizer_type == 'AdamW' else torch.optim.SGD([batch['res_type_logits']], lr=learning_rate_pre if pre_run else learning_rate)

    def norm_seq_grad(grad, chain_mask):
        chain_mask = chain_mask.bool()
        masked_grad = grad[:, chain_mask.squeeze(0), :] 
        eff_L = (masked_grad.pow(2).sum(-1, keepdim=True) > 0).sum(-2, keepdim=True)
        gn = masked_grad.norm(dim=(-1, -2), keepdim=True) 
        return grad * torch.sqrt(torch.tensor(eff_L)) / (gn + 1e-7)

    alphabet = list('XXARNDCQEGHILKMFPSTWYV-')
    best_loss = float('inf')  
    min_loss = float('inf') 
    best_batch = None     
    first_step_best_batch=None

    plots = []
    distogram_history = []
    sequence_history = []
    loss_history = []
    lr_history = []
    con_loss_history = []
    i_con_loss_history = []
    plddt_loss_history = []

    mask = torch.ones_like(batch['res_type_logits'])
    mask[batch['entity_id']!=chain_to_number[binder_chain], :] = 0
    chain_mask = (batch['entity_id'] == chain_to_number[binder_chain]).int()
    mid_points = torch.linspace(2, 22, 64).to(device) 

    def design(batch, 
               iters = None,
                soft=0.0, e_soft=None,
                step=1.0, e_step=None,
                temp=1.0, e_temp=None,
                hard=0.0, e_hard=None,
                num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=1,
                learning_rate=1.0,
                inter_chain_cutoff=21.0,
                intra_chain_cutoff=14.0,
                mask=None,
                chain_mask=None,
                length=100,
                plots=None,
                loss_history=None,
                i_con_loss_history=None,
                con_loss_history=None,
                plddt_loss_history=None,
                distogram_history=None,
                sequence_history=None,
                pre_run=False,
                mask_ligand=False,
                distogram_only=False,
                predict_args=None,
                alpha=2.0,
                loss_scales=None,
                binder_chain='A',
                non_protein_target=False,
                increasing_contact_over_itr=False,
                optimize_contact_per_binder_pos=False,
                num_inter_contacts=2,
                num_intra_contacts=4,
                save_trajectory=False,
                ):

        prev_sequence=""
        def get_model_loss(batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, pre_run=False, mask_ligand=False, distogram_only=False, predict_args=None, loss_scales=None, binder_chain='A', increasing_contact_over_itr=False, optimize_contact_per_binder_pos=False, num_inter_contacts=2, num_intra_contacts=4,  num_optimizing_binder_pos =1, inter_chain_cutoff=21.0, intra_chain_cutoff=14.0, save_trajectory=False, variable_mask_binder=None, chain_to_number=None, device=None):
            traj_coords = None
            traj_plddt = None

            # Device should be reliably passed or obtained before this function.
            # If device is None here, it's an issue with the calling context.
            # For safety, one might default or raise an error:
            if device is None:
                if batch.get('res_type_logits') is not None:
                    device = batch['res_type_logits'].device
                else:
                    # This is a fallback if no tensor is available in batch to get device from.
                    # It's better if device is always passed.
                    # Consider raising an error if device cannot be determined.
                    print("Warning: get_model_loss - device is None and could not be inferred from batch. Defaulting to CPU, but this may be incorrect.")
                    device = torch.device("cpu")

            # Prepare chain masks
            binder_chain_numeric_id = chain_to_number.get(binder_chain)
            if binder_chain_numeric_id is None:
                print(f"Error: get_model_loss - binder_chain {binder_chain} not in chain_to_number.")
                chain_mask_bool = torch.zeros(batch['entity_id'].shape[1], dtype=torch.bool, device=device)
            else:
                chain_mask_bool = (batch['entity_id'].squeeze(0) == binder_chain_numeric_id)

            effective_mask_1d_binder_variable = chain_mask_bool # Default to all binder residues if no variable_mask_binder
            if variable_mask_binder is not None:
                num_binder_residues_in_batch = chain_mask_bool.sum().item()
                if num_binder_residues_in_batch == variable_mask_binder.shape[0]:
                    temp_variable_mask_total_L = torch.zeros_like(chain_mask_bool, dtype=torch.bool, device=device)
                    temp_variable_mask_total_L[chain_mask_bool] = variable_mask_binder
                    effective_mask_1d_binder_variable = temp_variable_mask_total_L
                else:
                    print(f"Warning: get_model_loss - Length mismatch for binder chain. Variable mask not applied effectively. Binder in batch: {num_binder_residues_in_batch}, var_mask: {variable_mask_binder.shape[0]}")

            mask_for_pae_other_chains = ~chain_mask_bool # For PAE interactions with non-binder parts

            # Handle masking first if needed (original pre_run and mask_ligand logic)
            if pre_run and mask_ligand and chain_to_number is not None: # ensure chain_to_number is available
                binder_id_for_masking = chain_to_number.get(binder_chain)
                if binder_id_for_masking is not None:
                    batch['token_pad_mask'][batch['entity_id']!=binder_id_for_masking]=0 # Ensure comparison is with numeric ID
                    # The following masking logic for token_to_rep_atom and atom_pad_mask seems complex
                    # and might need careful review if it's causing issues.
                    # For now, assuming it's correct as per original intent if binder_id_for_masking is valid.
                    if 'token_to_rep_atom' in batch and 'atom_pad_mask' in batch :
                        masked_token_to_rep = torch.ones_like(batch['token_to_rep_atom'])
                        masked_token_to_rep[batch['entity_id']==binder_id_for_masking,:] = 0
                        # Ensure that the multiplication and nonzero operation are safe
                        # This can lead to issues if shapes are not broadcastable or if result is empty
                        # It might be safer to directly create masks based on entity_id
                        if batch['token_to_rep_atom'].shape == masked_token_to_rep.shape :
                             product = batch['token_to_rep_atom'] * masked_token_to_rep
                             if product.ndim >=3 : # Expecting at least 3 dims (batch, seq, atom_type)
                                masked_token_to_rep_index_tuple = torch.nonzero(product, as_tuple=True)
                                if len(masked_token_to_rep_index_tuple) == product.ndim and masked_token_to_rep_index_tuple[2].numel() > 0:
                                     masked_token_to_rep_index = masked_token_to_rep_index_tuple[2]
                                     batch['atom_pad_mask'][:, masked_token_to_rep_index] = 0
                                # else: print("Warning: masked_token_to_rep_index was empty or product had unexpected dimensions.")
                        # else: print("Warning: Shape mismatch for token_to_rep_atom masking.")
                    # else: print("Warning: token_to_rep_atom or atom_pad_mask missing from batch for pre_run masking.")


            # Common arguments for get_distogram_confidence
            confidence_args = {
                'recycling_steps': predict_args["recycling_steps"],
                'num_sampling_steps': predict_args["sampling_steps"],
                'multiplicity_diffusion_train': 1,
                'diffusion_samples': predict_args["diffusion_samples"],
                'run_confidence_sequentially': True,
                'disconnect_feats': disconnect_feats,
                'disconnect_pairformer': disconnect_pairformer
            }

            if save_trajectory:
                # Get model output with trajectory info
                dict_out = boltz_model.get_distogram_confidence(batch, **confidence_args)
                traj_coords = dict_out['sample_atom_coords'][0].detach().cpu().numpy()
                traj_plddt = dict_out['plddt'][0].detach().cpu().numpy()
            else:
                # Get model output without trajectory
                if pre_run or distogram_only:
                    dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
                else:
                    dict_out = boltz_model.get_distogram_confidence(batch, **confidence_args)


            pdist = dict_out['pdistogram']
            mid_pts = get_mid_points(pdist).to(device) # device should be set by now

            # Calculate contact losses using effective_mask_1d_binder_variable
            con_loss = get_con_loss(pdist, mid_pts,
                                num=num_intra_contacts, seqsep=9, cutoff=intra_chain_cutoff,
                                binary=False,
                                mask_1d=effective_mask_1d_binder_variable,
                                mask_1b=effective_mask_1d_binder_variable)

            if optimize_contact_per_binder_pos:
                if increasing_contact_over_itr:
                    num_optimizing_binder_pos = 0 if pre_run else num_optimizing_binder_pos
                # Contacts between binder's variable part and other chains
                i_con_loss = get_con_loss(pdist, mid_pts,
                                        num=num_inter_contacts, seqsep=0, num_pos=num_optimizing_binder_pos,
                                        cutoff=inter_chain_cutoff, binary=False,
                                        mask_1d=effective_mask_1d_binder_variable, # variable part of binder
                                        mask_1b=mask_for_pae_other_chains)      # other chains
            else:
                # Contacts between other chains and binder's variable part
                i_con_loss = get_con_loss(pdist, mid_pts,
                                        num=num_inter_contacts, seqsep=0, 
                                        cutoff=inter_chain_cutoff, binary=False, 
                                        mask_1d=mask_for_pae_other_chains,      # other chains
                                        mask_1b=effective_mask_1d_binder_variable) # variable part of binder

            # Helix loss for variable parts of the binder
            mask_2d_helix = effective_mask_1d_binder_variable.unsqueeze(1) * effective_mask_1d_binder_variable.unsqueeze(0)
            helix_loss = _get_helix_loss(pdist, mid_pts,
                                    offset=None, mask_2d=mask_2d_helix, binary=True)


            if pre_run and mask_ligand:
                    losses = {
                        'con_loss': con_loss,
                        'helix_loss': helix_loss
                    }
            else:
                    losses = {
                        'con_loss': con_loss,
                        'i_con_loss': i_con_loss,
                        'helix_loss': helix_loss
                    }           

            if not pre_run and not distogram_only:
                plddt_values = dict_out['plddt'].squeeze(0) # Shape (total_L)
                plddt_loss = get_plddt_loss(plddt_values, mask_1d=effective_mask_1d_binder_variable)

                pae_ symmetrized = (dict_out['pae'] + dict_out['pae'].transpose(-2,-1))/2 # pae shape (1, total_L, total_L)
                squeezed_pae = pae_symmetrized.squeeze(0) # Shape (total_L, total_L)

                # pae_loss for binder's variable residues interacting with other variable residues of binder
                pae_loss = get_pae_loss(squeezed_pae,
                                        mask_1d=effective_mask_1d_binder_variable,
                                        mask_1b=effective_mask_1d_binder_variable)

                # i_pae_loss for binder's variable residues interacting with other chains
                i_pae_loss = get_pae_loss(squeezed_pae,
                                          mask_1d=mask_for_pae_other_chains,
                                          mask_1b=effective_mask_1d_binder_variable)

                rg_loss, rg = add_rg_loss(dict_out['sample_atom_coords'], batch, length, binder_chain=binder_chain) # Remains unchanged

                losses.update({
                    'plddt_loss': plddt_loss,
                    'i_pae_loss': i_pae_loss,
                    'pae_loss': pae_loss,
                    'rg_loss': rg_loss
                })
                
                plddt_loss_history.append(plddt_loss.item())

            bins = mid_points < 8.0
            px = torch.sum(torch.softmax(dict_out['pdistogram'], dim=-1)[:,:,:,bins], dim=-1)

            if loss_scales is None:
                loss_scales = {
                    'con_loss': 1.0,
                    'i_con_loss': 1.0, 
                    'helix_loss': random.uniform(-0.4, 0.0),
                    'plddt_loss': 0.1,
                    'pae_loss': 0.4,
                    'i_pae_loss': 0.1,
                    'rg_loss': 0.0,
                }

            # Calculate total loss and print individual losses
            total_loss = sum(loss * loss_scales[name] for name, loss in losses.items())
            loss_str = [f"{k}:{v.item():.2f}" for k,v in losses.items()]
            plots.append(px[0].detach().cpu().numpy())
            loss_history.append(total_loss.item())
            i_con_loss_history.append(i_con_loss.item())
            con_loss_history.append(con_loss.item())
            # distogram_history.append(torch.softmax(dict_out['pdistogram'], dim=-1)[0].detach().cpu().numpy())
            distogram_history.append(px[0].detach().cpu().numpy())
            sequence_history.append(batch['res_type'][0, :, 2:22].detach().cpu().numpy())

            return total_loss, plots, loss_history, i_con_loss_history, con_loss_history, distogram_history, sequence_history, plddt_loss_history, loss_str, traj_coords, traj_plddt
        
        def update_sequence(opt, batch, mask, alpha=2.0, non_protein_target=False, binder_chain='A', variable_mask_binder=None, chain_to_number=None, device=None):
            # Ensure device is available, if not passed, try to get from batch tensors
            if device is None and batch.get('res_type_logits') is not None:
                device = batch['res_type_logits'].device
            elif device is None:
                # Fallback or raise error if device cannot be determined
                # For now, assuming it will be available or error handling is elsewhere.
                pass

            batch["logits"] = alpha*batch['res_type_logits']
            # The line for X needs device context if not already applied to sum part
            forbidden_sum_term = torch.sum(torch.eye(batch['logits'].shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0)
            if device:
                forbidden_sum_term = forbidden_sum_term.to(device)
            X =  batch['logits']- forbidden_sum_term
            batch['soft'] = torch.softmax(X/opt["temp"],dim=-1)
            batch['hard'] =  torch.zeros_like(batch['soft']).scatter_(-1, batch['soft'].max(dim=-1, keepdim=True)[1], 1.0)
            batch['hard'] =  (batch['hard'] - batch['soft']).detach() + batch['soft']
            batch['pseudo'] =  opt["soft"] * batch["soft"] + (1-opt["soft"]) * batch["res_type_logits"]
            batch['pseudo'] = opt["hard"] * batch["hard"] + (1-opt["hard"]) * batch["pseudo"]

            if variable_mask_binder is not None and chain_to_number is not None:
                binder_chain_numeric_id = chain_to_number.get(binder_chain)
                if binder_chain_numeric_id is not None and 'entity_id' in batch:
                    is_binder_chain_flat = (batch['entity_id'].squeeze(0) == binder_chain_numeric_id)
                    num_binder_residues_in_batch = is_binder_chain_flat.sum().item()

                    if num_binder_residues_in_batch == variable_mask_binder.shape[0]:
                        full_variable_mask_for_logits_flat = torch.zeros_like(is_binder_chain_flat, dtype=torch.bool, device=device)
                        full_variable_mask_for_logits_flat[is_binder_chain_flat] = variable_mask_binder
                        binder_variable_locations_3d = full_variable_mask_for_logits_flat.unsqueeze(0).unsqueeze(-1).expand_as(batch['pseudo'])

                        term_variable_binder = batch['pseudo'] * mask * binder_variable_locations_3d
                        term_fixed_binder = batch['res_type_logits'] * mask * (~binder_variable_locations_3d)
                        term_non_binder = batch['res_type_logits'] * (1 - mask)
                        batch['res_type'] = term_variable_binder + term_fixed_binder + term_non_binder
                    else:
                        print(f"Warning: update_sequence - Length mismatch for binder chain. Using original res_type update logic. Binder in batch: {num_binder_residues_in_batch}, var_mask: {variable_mask_binder.shape[0]}")
                        batch['res_type'] = batch['pseudo'] * mask + batch['res_type_logits'] * (1 - mask)
                else:
                    # print(f"Warning: update_sequence - Binder chain '{binder_chain}' not processed correctly for variable mask. Using original logic.")
                    batch['res_type'] = batch['pseudo'] * mask + batch['res_type_logits'] * (1 - mask)
            else:
                batch['res_type'] = batch['pseudo'] * mask + batch['res_type_logits'] * (1 - mask)
        
            if non_protein_target:
                batch['msa'] = batch['res_type'].unsqueeze(0).to(device).detach()
                batch['profile'] = batch['msa'].float().mean(dim=0).to(device).detach()
            else:
                # Ensure chain_to_number and binder_chain are available for this part
                if chain_to_number is not None: # Added check
                    binder_entity_id_val = chain_to_number.get(binder_chain)
                    if binder_entity_id_val is not None: # Added check
                         batch['msa'][:,0,:,:] = batch['res_type'].to(device).detach() # This line was fine
                         # The profile update should only be for the binder chain if entity_id matches
                         binder_profile_mask = (batch['entity_id']==binder_entity_id_val)[0] # entity_id is (1,L), so squeeze
                         if torch.any(binder_profile_mask): # Check if binder chain exists in batch
                            batch['profile'][binder_profile_mask,:] = batch['msa'][:, 0, binder_profile_mask,:].float().mean(dim=1).to(device).detach()

            return batch
        
        m = {"soft":[soft,e_soft],"temp":[temp,e_temp],"hard":[hard,e_hard], "step":[step,e_step], 'num_optimizing_binder_pos':[num_optimizing_binder_pos, e_num_optimizing_binder_pos]}
        m = {k:[s,(s if e is None else e)] for k,(s,e) in m.items()}

        opt = {}
        traj_coords_list = []
        traj_plddt_list = []
        for i in range(iters):
            for k,(s,e) in m.items():
                if k == "temp":
                    opt[k] = (e+(s-e)*(1-(i)/iters)**2)
                else:
                    v = (s+(e-s)*((i)/iters))
                    if k == "step": step = v
                    opt[k] = v
                
            lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            num_optimizing_binder_pos = int(opt["num_optimizing_binder_pos"])

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale

            opt["lr_rate"] = learning_rate * lr_scale

            # Pass chain_to_number and device to update_sequence
            batch = update_sequence(opt, batch, mask,
                                    variable_mask_binder=batch.get('variable_mask_binder_1d'),
                                    non_protein_target=non_protein_target,
                                    binder_chain=binder_chain,
                                    chain_to_number=chain_to_number,
                                    device=device)

            total_loss, plots, loss_history, i_con_loss_history, con_loss_history, distogram_history, sequence_history, plddt_loss_history, loss_str, traj_coords, traj_plddt = get_model_loss(
                batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history,
                pre_run, mask_ligand, distogram_only, predict_args, loss_scales, binder_chain,
                increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos,
                num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts,
                num_optimizing_binder_pos=num_optimizing_binder_pos,
                inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff,
                save_trajectory = save_trajectory,
                variable_mask_binder=batch.get('variable_mask_binder_1d'), # Pass to get_model_loss
                chain_to_number=chain_to_number, device=device # Pass to get_model_loss
            )
            traj_coords_list.append(traj_coords)
            traj_plddt_list.append(traj_plddt)
            current_sequence = ''.join([alphabet[i] for i in torch.argmax(batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
            if prev_sequence is not None:
                diff_count = sum(1 for a, b in zip(current_sequence, prev_sequence) if a != b)
                diff_percentage = (diff_count / length) * 100
            prev_sequence = current_sequence
            total_loss.backward()

            if batch['res_type_logits'].grad is not None:
                # 1. Zero out grads for non-binder chains
                batch['res_type_logits'].grad[batch['entity_id'] != chain_to_number[binder_chain], :] = 0.0

                # 2. Zero out grads for fixed positions within the binder chain
                fixed_mask_binder_1d = batch.get('fixed_mask_binder_1d')
                if fixed_mask_binder_1d is not None:
                    binder_chain_numeric_id = chain_to_number.get(binder_chain)
                    if binder_chain_numeric_id is not None:
                        is_binder_chain_mask_flat = (batch['entity_id'].squeeze(0) == binder_chain_numeric_id)
                        if torch.any(is_binder_chain_mask_flat):
                            grad_binder_view = batch['res_type_logits'].grad[0, is_binder_chain_mask_flat, :]
                            if grad_binder_view.shape[0] == fixed_mask_binder_1d.shape[0]:
                                grad_binder_view[fixed_mask_binder_1d, :] = 0.0
                                batch['res_type_logits'].grad[0, is_binder_chain_mask_flat, :] = grad_binder_view
                            else:
                                print(f"Warning: design - Mismatch between grad binder part length {grad_binder_view.shape[0]} and fixed_mask_binder_1d length {fixed_mask_binder_1d.shape[0]}. Gradient not masked for fixed residues.")

                # 3. Zero out grads for forbidden tokens
                # Using the existing list: [0,1,6,22,23,24,25,26,27,28,29,30,31,32]
                # Note: Index 6 is CYS. If CYS can be a fixed residue, its grad is already zeroed by step 2.
                # If CYS is truly forbidden for variable positions, this handles it.
                forbidden_token_indices = [0,1,6,22,23,24,25,26,27,28,29,30,31,32]
                batch['res_type_logits'].grad[..., forbidden_token_indices] = 0.0

                # 4. Normalize binder chain grads
                # chain_mask is (batch_size, total_length)
                # norm_seq_grad expects chain_mask to identify the part of the grad to normalize.
                current_chain_mask_for_norm = (batch['entity_id'] == chain_to_number[binder_chain]).int()
                batch['res_type_logits'].grad = norm_seq_grad(batch['res_type_logits'].grad, current_chain_mask_for_norm)

                optimizer.step()
                optimizer.zero_grad()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {i}: lr: {current_lr:.3f}, soft: {opt['soft']:.2f}, hard: {opt['hard']:.2f}, temp: {opt['temp']:.2f}, total loss: {total_loss.item():.2f}, {loss_str}")
        
        return batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list

    if pre_run:
        batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list = design(batch, iters=pre_iteration, soft=1.0, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, mask_ligand=mask_ligand, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
    else:
        if design_algorithm == "3stages":
            print('-'*100)
            print(f"logits to softmax(T={e_soft})")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list1, traj_plddt_list1 = design(batch, iters=soft_iteration, e_soft=e_soft, num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=8, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("softmax(T=1) to softmax(T=0.01)")
            print('-'*100)
            print("set res_type_logits to logits")
            new_logits = (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            batch['res_type_logits'] = new_logits
            optimizer = torch.optim.SGD([batch['res_type_logits']], lr=learning_rate)
            batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history, traj_coords_list2, traj_plddt_list2 = design(batch, iters=temp_iteration, soft=1.0, temp = 1.0,e_temp=0.01, num_optimizing_binder_pos=8, e_num_optimizing_binder_pos=12,  mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("hard")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list3, traj_plddt_list3 = design(batch, iters=hard_iteration, soft=1.0, hard = 1.0,temp=0.01, num_optimizing_binder_pos=12, e_num_optimizing_binder_pos=16, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            traj_coords_list = traj_coords_list1 + traj_coords_list2 + traj_coords_list3 if save_trajectory else []
            traj_plddt_list = traj_plddt_list1 + traj_plddt_list2 + traj_plddt_list3 if save_trajectory else []

        elif design_algorithm == "3stages_extra":
            print('-'*100)
            print(f"logits to softmax(T={e_soft_1})")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list1, traj_plddt_list1 = design(batch, iters=soft_iteration_1, e_soft=e_soft_1, num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=8, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory) 
            print('-'*100)
            print(f"logits to softmax(T={e_soft_2})")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list2, traj_plddt_list2 = design(batch, iters=soft_iteration_2, e_soft=e_soft_2, num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=8, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("softmax(T=1) to softmax(T=0.01)")
            print('-'*100)
            print("set res_type_logits to logits")
            new_logits = (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            batch['res_type_logits'] = new_logits
            optimizer = torch.optim.SGD([batch['res_type_logits']], lr=learning_rate)
            batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history, traj_coords_list3, traj_plddt_list3 = design(batch, iters=temp_iteration, soft=1.0, temp = 1.0,e_temp=0.01, num_optimizing_binder_pos=8, e_num_optimizing_binder_pos=12,  mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)
            print('-'*100)
            print("hard")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list4, traj_plddt_list4 = design(batch, iters=hard_iteration, soft=1.0, hard = 1.0,temp=0.01, num_optimizing_binder_pos=12, e_num_optimizing_binder_pos=16, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)

            traj_coords_list = traj_coords_list1 + traj_coords_list2 + traj_coords_list3 + traj_coords_list4 if save_trajectory else []
            traj_plddt_list = traj_plddt_list1 + traj_plddt_list2 + traj_plddt_list3 + traj_plddt_list4 if save_trajectory else []
                
        elif design_algorithm == "logits":
            print('-'*100)
            print("logits")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list= design(batch, iters=soft_iteration, soft = 0.0, e_soft=0.0, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, increasing_contact_over_itr=increasing_contact_over_itr, optimize_contact_per_binder_pos=optimize_contact_per_binder_pos, non_protein_target=non_protein_target, inter_chain_cutoff=inter_chain_cutoff, intra_chain_cutoff=intra_chain_cutoff, num_inter_contacts=num_inter_contacts, num_intra_contacts=num_intra_contacts, save_trajectory=save_trajectory)

    def _run_model(boltz_model, batch, predict_args):
        boltz_model.predict_args = predict_args
        return boltz_model.predict_step(batch, batch_idx=0, dataloader_idx=0)

    def visualize_results(plots):
        # Plot distogram predictions
        if plots:
            num_plots = len(plots)
            num_rows = (num_plots + 5) // 6
            fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))
            
            if num_rows == 1:
                axs = axs.reshape(1, -1)

            for i, plot_data in enumerate(plots):
                row, col = i // 6, i % 6
                axs[row, col].imshow(plot_data)
                axs[row, col].set_title(f'Epoch {i + 1}')
                axs[row, col].axis('off')

            # Hide unused subplots
            for j in range(num_plots, num_rows * 6):
                axs[j // 6, j % 6].axis('off')

            plt.tight_layout()
            plt.show()
            plots.clear()

    # visualize_results(plots)

    if pre_run:
        predict_args = {
        "recycling_steps": 3,  # Default value
        "sampling_steps": 200,  # Default value
        "diffusion_samples": 1,  # Default value
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
        }

        best_logits = batch['res_type_logits']
        best_seq = ''.join([alphabet[i] for i in torch.argmax(batch['res_type'][batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
        data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = best_seq
        return batch['res_type'].detach().cpu().numpy(), plots, loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list

    boltz_model.eval()

    if best_batch is None:
        if first_step_best_batch is not None:
            best_batch = first_step_best_batch
        else:
            best_batch = batch  

    predict_args = {
    "recycling_steps": 3,  # Default value
    "sampling_steps": 200,  # Default value
    "diffusion_samples": 1,  # Default value
    "write_confidence_summary": True,
    "write_full_pae": True,
    "write_full_pde": False,
    }

    # _mutate is defined within boltz_hallucination, so it has access to length, alphabet, device from outer scope
    def _mutate(current_sequence_str, binder_specific_logits, probability_per_position, variable_mask_for_binder=None):
        # length, alphabet, device are from the outer scope of boltz_hallucination
        binder_length = len(current_sequence_str)
        mutated_sequence_list = list(current_sequence_str)

        chosen_idx_in_binder = -1

        if variable_mask_for_binder is not None:
            # Ensure variable_mask_for_binder is a numpy array for boolean indexing
            if torch.is_tensor(variable_mask_for_binder):
                variable_mask_np = variable_mask_for_binder.cpu().numpy().astype(bool)
            else: # Assuming it's already a numpy array or compatible
                variable_mask_np = np.array(variable_mask_for_binder, dtype=bool)

            # Ensure variable_mask_np has the same length as the binder sequence
            if variable_mask_np.shape[0] != binder_length:
                print(f"Warning: _mutate - Mismatch between variable_mask_np length {variable_mask_np.shape[0]} and binder_length {binder_length}. Defaulting to original mutation behavior.")
                variable_mask_np = np.ones(binder_length, dtype=bool) # Fallback to all variable

            possible_indices = np.arange(binder_length)[variable_mask_np]
            if possible_indices.size == 0:
                # All residues are fixed, no mutation possible
                return "".join(mutated_sequence_list)

            # Filter probability_per_position to only include variable positions
            prob_variable_part = probability_per_position[variable_mask_np]

            prob_sum = prob_variable_part.sum()
            if prob_sum == 0 or np.isnan(prob_sum) or np.isinf(prob_sum): # Avoid division by zero/nan/inf
                # Fallback: uniform probability over variable positions
                normalized_prob_variable = np.ones_like(prob_variable_part) / float(prob_variable_part.size) if prob_variable_part.size > 0 else []
            else:
                normalized_prob_variable = prob_variable_part / prob_sum

            if possible_indices.size > 0 and normalized_prob_variable.size > 0 :
                 # Ensure normalized_prob_variable sums to 1 for np.random.choice
                if not np.isclose(normalized_prob_variable.sum(), 1.0):
                    normalized_prob_variable = normalized_prob_variable / normalized_prob_variable.sum() # Re-normalize if not close
                chosen_idx_in_binder = np.random.choice(possible_indices, p=normalized_prob_variable)
            else: # No variable positions to choose from after filtering
                 return "".join(mutated_sequence_list)

        else:
            # Original behavior: choose from any position if no variable_mask is provided
            prob_sum = probability_per_position.sum()
            if prob_sum == 0 or np.isnan(prob_sum) or np.isinf(prob_sum):
                 normalized_prob = np.ones_like(probability_per_position) / float(probability_per_position.size) if probability_per_position.size > 0 else []
            else:
                normalized_prob = probability_per_position / prob_sum

            if probability_per_position.size > 0 and normalized_prob.size > 0:
                if not np.isclose(normalized_prob.sum(), 1.0):
                     normalized_prob = normalized_prob / normalized_prob.sum()
                chosen_idx_in_binder = np.random.choice(np.arange(binder_length), p=normalized_prob)
            else: # Cannot make a choice
                return "".join(mutated_sequence_list)

        if chosen_idx_in_binder == -1 : # Should not happen if logic above is correct
            return "".join(mutated_sequence_list)

        # binder_specific_logits is already sliced: (1, binder_length, num_tokens)
        idx_logits = binder_specific_logits[0, chosen_idx_in_binder, :]
        idx_logits = idx_logits - torch.max(idx_logits) # For numerical stability

        # Use the forbidden token sum from the outer scope, ensure it's correctly defined and on device
        # This was: i_X = i_logits - (torch.sum(torch.eye(i_logits.shape[-1])[[0,1,6,22,...]],dim=0)*(1e10)).to(device)
        # Assuming 'forbidden_logit_bias' is predefined in outer scope correctly (shape: num_tokens)
        # Example: forbidden_logit_bias = torch.sum(torch.eye(num_tokens_total)[[0,1,6,...]], dim=0).to(device) * 1e10
        # For now, re-creating it here for clarity, but ideally it's passed or from scope.
        # num_tokens must match idx_logits.shape[-1]
        num_tokens_for_mutation = idx_logits.shape[-1]
        forbidden_indices_for_mutation = [0,1,6,22,23,24,25,26,27,28,29,30,31,32] # Ensure this list is correct for 'tokens'
        # Filter indices that are out of bounds for num_tokens_for_mutation
        valid_forbidden_indices = [idx for idx in forbidden_indices_for_mutation if idx < num_tokens_for_mutation]

        forbidden_bias_term = torch.zeros(num_tokens_for_mutation, device=device)
        if valid_forbidden_indices: # only if list is not empty
            forbidden_bias_term.scatter_(0, torch.tensor(valid_forbidden_indices, device=device), 1.0)
        forbidden_bias_term *= 1e10

        idx_X = idx_logits - forbidden_bias_term

        # TEMP_MUTATE could be 1.0 (implicit in original) or a parameter
        idx_probs = torch.softmax(idx_X / 1.0, dim=-1)

        if torch.all(torch.isnan(idx_probs)) or idx_probs.sum() == 0: # Avoid error with multinomial if all probs are 0/nan
            # Fallback: sample uniformly from allowed tokens if softmax fails
            allowed_indices = [i for i in range(num_tokens_for_mutation) if i not in valid_forbidden_indices]
            if not allowed_indices: return "".join(mutated_sequence_list) # No allowed tokens
            chosen_aa_token_idx = random.choice(allowed_indices)
        else:
            chosen_aa_token_idx = torch.multinomial(idx_probs, 1).item()

        mutated_sequence_list[chosen_idx_in_binder] = alphabet[chosen_aa_token_idx] # alphabet from outer scope
        return "".join(mutated_sequence_list)

    # Initial best_logits (for all chains) and best_seq (binder only)
    initial_best_logits_all_chains = best_batch['res_type_logits']
    initial_best_seq_binder = ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][best_batch['entity_id']==chain_to_number[binder_chain],:], dim=-1).detach().cpu().numpy()])
    data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = initial_best_seq_binder

    data_apo = copy.deepcopy(data)
    data_apo.pop('constraints', None)
    # Ensure only the binder chain sequence is in data_apo if that's the intent
    # This assumes chain_to_number[binder_chain] gives the correct index in the original list
    binder_idx_in_yaml = chain_to_number.get(binder_chain)
    if binder_idx_in_yaml is not None and binder_idx_in_yaml < len(data['sequences']):
         # This was data_apo['sequences'] = [data_apo['sequences'][chain_to_number[binder_chain]]]
         # but data_apo is a deepcopy of data. So we refer to data's structure.
        data_apo['sequences'] = [data['sequences'][binder_idx_in_yaml]]
    else:
        # Fallback or error if binder_chain not found or index out of bounds
        print(f"Warning: Could not isolate binder chain {binder_chain} for apo data. Check chain_to_number mapping and YAML structure.")
        # Decide on fallback: use all sequences, or error out, or use first sequence if only one expected
        # For now, if it fails, it might carry over other chains.
        pass # Or data_apo['sequences'] = [] to make it clearly problematic if not handled

    def _update_batches(data, data_apo):
        target = parse_boltz_schema(name, data, ccd_lib)
        target_apo = parse_boltz_schema(name, data_apo, ccd_lib)
        best_batch, best_structure = get_batch(target, msa_max_seqs, length, keep_record=True)
        best_batch_apo, best_structure_apo = get_batch(target_apo, msa_max_seqs, length, keep_record=True)
        best_batch = {key: value.unsqueeze(0).to(device) if key != 'record' else value for key, value in best_batch.items()}
        best_batch_apo = {key: value.unsqueeze(0).to(device) if key != 'record' else value for key, value in best_batch_apo.items()}
        return best_batch, best_batch_apo, best_structure, best_structure_apo

    best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(data, data_apo)
    output = _run_model(boltz_model, best_batch, predict_args)
    output_apo = _run_model(boltz_model, best_batch_apo, predict_args)

    # Use initial_best_seq_binder for prev_sequence, and initial_best_logits_all_chains for best_logits
    # This ensures that mutations start from the sequence generated by the main design loop.
    prev_sequence = initial_best_seq_binder
    # best_logits for mutation should be from the *current* best_batch state if we want iterative refinement from the previous step's output.
    # However, the original code used `best_logits` from *before* the semi-greedy loop.
    # Let's stick to the original intent: best_logits refers to the output of the main design stages.
    # So, best_logits_for_mutation_loop = initial_best_logits_all_chains

    prev_iptm = output['iptm'].detach().cpu().numpy()
    print("best design iptm", prev_iptm)
    print("Semi-greedy steps", semi_greedy_steps)

    for step in range(semi_greedy_steps):
        confidence_score = []
        mutated_sequence_ls = []
        
        # best_batch is updated in each semi-greedy step by _update_batches.
        # So, plddt and sequence should be from the current best_batch.
        current_binder_chain_id = chain_to_number.get(binder_chain)
        if current_binder_chain_id is None: # Should not happen
            print(f"Error: semi-greedy - binder chain {binder_chain} not found in chain_to_number. Skipping mutation.")
            break

        binder_mask_for_plddt = (best_batch['entity_id'].squeeze(0) == current_binder_chain_id)

        # Ensure plddt_binder is correctly shaped and indexed
        if output['plddt'].ndim > 1 and output['plddt'].shape[0] == 1: # Expected shape (1, total_L)
            plddt_total_L = output['plddt'].squeeze(0)
            if binder_mask_for_plddt.shape[0] == plddt_total_L.shape[0]:
                 plddt_binder = plddt_total_L[binder_mask_for_plddt]
            else:
                print(f"Warning: semi-greedy - Mismatch plddt ({plddt_total_L.shape[0]}) and binder mask ({binder_mask_for_plddt.shape[0]}) lengths. Using uniform i_prob.")
                plddt_binder = torch.empty(0) # Will trigger fallback for i_prob_np
        else: # Fallback if plddt shape is unexpected
            print(f"Warning: semi-greedy - Unexpected plddt shape: {output['plddt'].shape}. Using uniform i_prob.")
            plddt_binder = torch.empty(0)


        # i_prob should have length equal to binder length (length is outer scope binder_length)
        if plddt_binder.numel() == 0 :
            i_prob_np = np.ones(length) / float(length) if length > 0 else np.array([])
            # print(f"Warning: semi-greedy - plddt for binder {binder_chain} is empty or length mismatch. Using uniform i_prob.")
        else:
            i_prob_tensor = torch.maximum(1.0 - plddt_binder, torch.tensor(0.0, device=plddt_binder.device))
            i_prob_np = i_prob_tensor.detach().cpu().numpy()
            if i_prob_np.shape[0] != length: # Double check length consistency after slicing
                print(f"Warning: semi-greedy - i_prob_np length {i_prob_np.shape[0]} differs from target binder length {length}. Using uniform i_prob.")
                i_prob_np = np.ones(length) / float(length) if length > 0 else np.array([])


        current_sequence_str = ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][0, binder_mask_for_plddt,:], dim=-1).detach().cpu().numpy()])

        # Prepare binder-specific logits for _mutate
        # initial_best_logits_all_chains is (1, total_L, n_tokens)
        # batch['entity_id'] is (1, total_L)
        if initial_best_logits_all_chains.shape[1] == batch['entity_id'].shape[1] : # total_L should match
            initial_binder_mask_flat = (batch['entity_id'].squeeze(0) == current_binder_chain_id)
            logits_for_binder_mutation = initial_best_logits_all_chains[:, initial_binder_mask_flat, :]
        else:
            print(f"Warning: semi-greedy - Logits shape mismatch (total_L: {initial_best_logits_all_chains.shape[1]} vs entity_id_L: {batch['entity_id'].shape[1]}). Cannot reliably slice binder logits for mutation.")
            # Fallback: skip mutation trial for this step or use a dummy tensor that _mutate can handle
            logits_for_binder_mutation = torch.empty((1, length, initial_best_logits_all_chains.shape[2]), device=device) # Dummy


        variable_mask_for_mutation = best_batch.get('variable_mask_binder_1d') # (binder_L,)

        for t in range(10): # Try 10 mutations per semi-greedy step
            if i_prob_np.size == 0: # Cannot make a choice for mutation position
                # print(f"Debug: semi-greedy trial {t} - i_prob_np is empty, skipping mutation.")
                mutated_sequence = current_sequence_str # No mutation possible
            elif logits_for_binder_mutation.shape[1] != length : # Check if binder logits length is correct
                print(f"Warning: semi-greedy trial {t} - Mismatch between logits_for_binder_mutation length ({logits_for_binder_mutation.shape[1]}) and binder length ({length}). Skipping mutation.")
                mutated_sequence = current_sequence_str
            else:
                mutated_sequence = _mutate(current_sequence_str, logits_for_binder_mutation, i_prob_np, variable_mask_for_mutation)

            data['sequences'][current_binder_chain_id]['protein']['sequence'] = mutated_sequence

            current_eval_batch, current_eval_batch_apo, _, _ = _update_batches(data, data_apo)
            current_output = _run_model(boltz_model, current_eval_batch, predict_args)
            # current_output_apo = _run_model(boltz_model, current_eval_batch_apo, predict_args) # if needed

            iptm = current_output['iptm'].detach().cpu().numpy()
            confidence_score.append(iptm)
            mutated_sequence_ls.append(mutated_sequence)
            print(f"Step {step}, Mutation Trial {t}, iptm {iptm.item():.3f}")

        best_trial_id = np.argmax(confidence_score)
        best_iptm_this_step = confidence_score[best_trial_id]

        if best_iptm_this_step > prev_iptm:
            best_seq_this_step = mutated_sequence_ls[best_trial_id]
            # Update prev_sequence and prev_iptm for the next semi-greedy step
            prev_sequence = best_seq_this_step
            prev_iptm = best_iptm_this_step

            # Update the main 'data' and 'data_apo' that will be used for the *next* semi-greedy step's _update_batches
            # or for the final output if this is the last step.
            data['sequences'][current_binder_chain_id]['protein']['sequence'] = best_seq_this_step
            if binder_idx_in_yaml is not None and binder_idx_in_yaml < len(data_apo['sequences']): # Should be always 1 sequence in data_apo now
                 data_apo['sequences'][0]['protein']['sequence'] = best_seq_this_step # data_apo has only binder

            print(f"Step {step}, Best Trial {best_trial_id}, Accepted new sequence. IPTM: {best_iptm_this_step:.3f} (prev: {output['iptm'].item():.3f})") # output['iptm'] is from previous step
            print(f"Updated sequence to: {best_seq_this_step}")

            # Update 'output' to reflect the best of this step for the next iteration's plddt source
            # This requires re-running _update_batches and _run_model with the accepted best_seq_this_step
            best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(data, data_apo)
            output = _run_model(boltz_model, best_batch, predict_args) # output_apo also if needed by plddt source
            # output_apo = _run_model(boltz_model, best_batch_apo, predict_args) # if apo plddt was used
        else:
            print(f"Step {step}, No improvement over IPTM {prev_iptm:.3f}. Max this step: {best_iptm_this_step:.3f}. Keeping previous sequence.")
            # Revert 'data' and 'data_apo' to `prev_sequence` if mutations were not accepted.
            data['sequences'][current_binder_chain_id]['protein']['sequence'] = prev_sequence
            if binder_idx_in_yaml is not None and binder_idx_in_yaml < len(data_apo['sequences']):
                 data_apo['sequences'][0]['protein']['sequence'] = prev_sequence
            # best_batch and output remain from the previous step or the start of this step.
            # No need to call _update_batches if sequence didn't change.

    # After all semi-greedy steps, ensure final best_batch, output etc. correspond to prev_sequence
    # This is implicitly handled if 'data' always reflects the accepted sequence.
    # Final _update_batches and _run_model if the sequence changed in the last step.
    # If no change in last step, current best_batch and output are already correct.
    # To be safe, always run _update_batches and _run_model for the final state based on final `data`.
    best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(data, data_apo)
    output = _run_model(boltz_model, best_batch, predict_args)
    output_apo = _run_model(boltz_model, best_batch_apo, predict_args)

    return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure
            
            iptm = output['iptm'].detach().cpu().numpy()
            confidence_score.append(iptm)
            mutated_sequence_ls.append(mutated_sequence)
            print(f"Step {step}, Epoch {t}, iptm {iptm[0]:.3f}")

        best_id = np.argmax(confidence_score)
        best_iptm = confidence_score[best_id]
        
        if best_iptm > prev_iptm:
            best_seq = mutated_sequence_ls[best_id]
            for seq_data in [data, data_apo]:
                seq_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = best_seq
            print(f"Step {step}, Epoch {best_id}, Update sequence, iptm {best_iptm}, previous iptm {prev_iptm}")
            print(f"Update sequence {best_seq}")
            prev_iptm = best_iptm
            prev_sequence = best_seq
        else:
            for seq_data in [data, data_apo]:
                seq_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence'] = prev_sequence

        best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(data, data_apo)

        if step == semi_greedy_steps - 1:
            output = _run_model(boltz_model, best_batch, predict_args)
            output_apo = _run_model(boltz_model, best_batch_apo, predict_args)

    return output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list, traj_plddt_list, structure


def run_boltz_design(
    boltz_path,
    main_dir,
    yaml_dir,
    boltz_model,
    ccd_path,
    design_samples =1,
    version_name=None,
    config=None,
    loss_scales=None,
    num_workers=1,
    show_animation=False,
    save_trajectory=False,
    redo_boltz_predict=True,
):
    """
    Run Boltz protein design pipeline.
    
    Args:
        main_dir (str): Main directory path
        yaml_dir (str): Directory containing input yaml files
        version_name (str): Name for version subdirectory
        config (dict): Configuration parameters. If None, uses defaults.
    """
    if config is None:
        config = {
            'recycling_steps': 0,
            'pre_iteration': 30,
            'soft_iteration': 75, 
            'soft_iteration_1': 50,
            'soft_iteration_2': 25,
            'temp_iteration': 45,
            'hard_iteration': 5,
            'semi_greedy_steps': 0,
            'learning_rate_pre': 0.2,
            'learning_rate': 0.1,
            'inter_chain_cutoff': 21.0,
            'intra_chain_cutoff': 14.0,
            'num_inter_contacts': 2,
            'num_intra_contacts': 4,
            'e_soft': 0.8,
            'e_soft_1': 0.8,
            'e_soft_2': 1.0,
            'design_algorithm': '3stages',
            'set_train': True,
            'use_temp': True,
            'disconnect_feats': True,
            'disconnect_pairformer': False,
            'distogram_only': True,
            'binder_chain': 'A',
            'non_protein_target': False,
            'increasing_contact_over_itr': False,
            'mask_ligand': False,
            'optimize_contact_per_binder_pos':False,
            'pocket_conditioning': False,
            'msa_max_seqs':4096,
            'length_min': 95,
            'length_max': 160,
            'helix_loss_min': -0.6,
            'helix_loss_max': -0.2,
            'optimizer_type': 'SGD',
        }


    version_dir = os.path.join(main_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)

    with open(os.path.expanduser(ccd_path), 'rb') as f:
        ccd_lib = pickle.load(f)
    
    results_final_dir = os.path.join(version_dir, 'results_final')
    results_yaml_dir = os.path.join(version_dir, 'results_yaml')
    results_final_dir_apo = os.path.join(version_dir, 'results_final_apo')
    results_yaml_dir_apo = os.path.join(version_dir, 'results_yaml_apo')
    loss_dir = os.path.join(version_dir, 'loss')
    animation_save_dir = os.path.join(version_dir, 'animation')

    for directory in [results_yaml_dir, results_final_dir, results_yaml_dir_apo, results_final_dir_apo, loss_dir, animation_save_dir]:
        os.makedirs(directory, exist_ok=True)

    # Save config
    config_path = os.path.join(results_final_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    alphabet = list('XXARNDCQEGHILKMFPSTWYV-')
    rmsd_csv_path = os.path.join(results_final_dir, 'rmsd_results.csv')
    csv_exists = os.path.exists(rmsd_csv_path)
    filtered_config = {k: v for k, v in config.items() 
                if k not in ['helix_loss_min', 'helix_loss_max', 'length_min', 'length_max']}
    for yaml_path in Path(yaml_dir).glob('*.yaml'):
        if yaml_path.name.endswith('.yaml'):
                target_binder_input = yaml_path.stem
                for itr in range(design_samples):
                    config['length'] = random.randint(config['length_min'],config['length_max'])
                    loss_scales['helix_loss'] = random.uniform(config['helix_loss_min'], config['helix_loss_max'])
                    filtered_config['length'] = config['length']

                    print('pre-run warm up')
                    input_res_type, plots, loss_history, distogram_history, sequence_history, traj_coords_list, traj_plddt_list = boltz_hallucination(
                        boltz_model,
                        yaml_path,
                        ccd_lib,
                        **filtered_config,
                        pre_run=True,
                        input_res_type=False,
                        loss_scales=loss_scales,
                        chain_to_number=chain_to_number,
                        save_trajectory=save_trajectory
                    )
                    print('warm up done')     
                    output, output_apo, best_batch, best_batch_apo, best_structure, best_structure_apo ,distogram_history_2, sequence_history_2, loss_history_2, con_loss_history, i_con_loss_history, plddt_loss_history, traj_coords_list_2, traj_plddt_list_2, structure = boltz_hallucination(
                        boltz_model,
                        yaml_path,
                        ccd_lib,
                        **filtered_config,
                        pre_run=False,
                        input_res_type=input_res_type,
                        loss_scales=loss_scales,
                        chain_to_number=chain_to_number,
                        save_trajectory=save_trajectory
                    )
                    loss_history.extend(loss_history_2)
                    distogram_history.extend(distogram_history_2) 
                    sequence_history.extend(sequence_history_2)
                    traj_coords_list.extend(traj_coords_list_2)
                    traj_plddt_list.extend(traj_plddt_list_2)

                    if save_trajectory:
                        from logmd import LogMD
                        logmd = LogMD() 
                        logmd.notebook()
                        print(logmd.url) 
                        atoms = structure.atoms
                        ref_coords = traj_coords_list[-1][:atoms['coords'].shape[0], :]
                        for i in range(len(traj_coords_list)):
                            current_coords = traj_coords_list[i][:atoms['coords'].shape[0], :]
                            aligned_coords = align_points(current_coords, ref_coords)
                            structure.atoms['coords'] = aligned_coords
                            structure.atoms["is_present"] = True
                            pdb_str = to_pdb(structure, plddts=traj_plddt_list[i])
                            pdb_str = "\n".join([line for line in pdb_str.split("\n") if line.startswith("ATOM") or line.startswith("HETATM")])
                            logmd(pdb_str)

                    print('-' * 100)
                    print(f"Holo Protein PLDDT: {output['plddt'][:config['length']].mean():.3f}")
                    print(f"Apo Protein PLDDT: {output_apo['plddt'][:config['length']].mean():.3f}")
                    print('-' * 100)
                    print(f"Holo Complex PLDDT: {float(output['complex_plddt'].detach().cpu().numpy()):.3f}")
                    print(f"Apo Complex PLDDT: {float(output_apo['complex_plddt'].detach().cpu().numpy()):.3f}")
                    print('-' * 100)

                    ca_coords = get_ca_coords(output['coords'], best_batch, binder_chain=config['binder_chain']).detach().cpu().numpy()
                    ca_coords_apo = get_ca_coords(output_apo['coords'], best_batch_apo, binder_chain='A').detach().cpu().numpy()

                    rmsd = np_rmsd(ca_coords, ca_coords_apo)
                    print('-' * 100)
                    print("rmsd", rmsd)
                    print('-' * 100) 

                    if loss_dir:
                        os.makedirs(loss_dir, exist_ok=True)
                    # Plot loss history
                    try:
                        # Create figure with a dark background style
                        plt.style.use('dark_background')
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
                        fig.patch.set_facecolor('#1C1C1C')
                        
                        # Custom colors for each plot
                        colors = ['#00ff99', '#ff3366', '#3366ff']
                        
                        # Plot 1: Training Loss
                        ax1.plot(loss_history, color=colors[0], linewidth=2)
                        ax1.set_xlabel('Epochs', fontsize=12)
                        ax1.set_ylabel('Total Loss', fontsize=12)
                        ax1.set_title('Total Loss History', fontsize=14, pad=15)
                        ax1.grid(True, linestyle='--', alpha=0.3)
                        
                        # Plot 2: Con Loss
                        ax2.plot(con_loss_history, color=colors[1], linewidth=2)
                        ax2.set_xlabel('Epochs', fontsize=12)
                        ax2.set_ylabel('Intra-Contact Loss', fontsize=12)
                        ax2.set_title('Intra-Contact Loss History', fontsize=14, pad=15)
                        ax2.grid(True, linestyle='--', alpha=0.3)
                        
                        # Plot 3: iCon Loss
                        ax3.plot(i_con_loss_history, color=colors[2], linewidth=2)
                        ax3.set_xlabel('Epochs', fontsize=12)
                        ax3.set_ylabel('Inter-Contact Loss', fontsize=12)
                        ax3.set_title('Inter-Contact Loss History', fontsize=14, pad=15)
                        ax3.grid(True, linestyle='--', alpha=0.3)
                        
                        # Adjust layout and add spacing between subplots
                        plt.tight_layout(pad=3.0)
                        
                        if loss_dir:
                            plt.savefig(os.path.join(loss_dir, f'{target_binder_input}_loss_history_itr{itr + 1}_length{config["length"]}.png'),
                                      facecolor='#1C1C1C', edgecolor='none', bbox_inches='tight', dpi=300)
                        plt.show()
                        distogram_ani, sequence_ani = visualize_training_history(best_batch,loss_history, sequence_history, distogram_history, config["length"], binder_chain =config['binder_chain'], save_dir=animation_save_dir, save_filename=f"{target_binder_input}_itr{itr + 1}_length{config['length']}")
                        if show_animation:
                            display(HTML(f"<div style='display:flex;gap:10px'><div style='flex:0.4'>{distogram_ani.to_jshtml()}</div><div style='flex:0.6'>{sequence_ani.to_jshtml()}</div></div>"))

                    except Exception as e:
                        print(f"Error plotting loss history: {str(e)}")
                        continue

                    with open(rmsd_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not csv_exists:
                            writer.writerow(['target', 'length', 'iteration', 'apo_holo_rmsd', 'complex_plddt', 'iptm',  'helix_loss'])
                            csv_exists = True
                        writer.writerow([target_binder_input, config['length'], itr + 1, rmsd, output['complex_plddt'].item(), output['iptm'].item(), loss_scales['helix_loss']])

                    result_yaml = os.path.join(results_yaml_dir, f'{target_binder_input}_results_itr{itr + 1}_length{config["length"]}.yaml')
                    result_yaml_apo = os.path.join(results_yaml_dir_apo, f'{target_binder_input}_results_itr{itr + 1}_length{config["length"]}.yaml')
                    best_batch_cpu = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in best_batch.items()}
                    best_sequence = ''.join([alphabet[i] for i in np.argmax(best_batch_cpu['res_type'][best_batch_cpu['entity_id']==chain_to_number[config['binder_chain']],:], axis=-1)])
                    print("best_sequence", best_sequence)
                    

                    shutil.copy2(yaml_path, result_yaml)
                    with open(result_yaml, 'r') as f:
                        data = yaml.safe_load(f)
                    chain_num = chain_to_number[config['binder_chain']]
                    data['sequences'][chain_num]['protein']['sequence'] = best_sequence
                    data.pop('constraints', None)

                    # Convert any MSA files from npz to a3m format
                    for seq in data['sequences']:
                        if 'protein' in seq and 'msa' in seq['protein'] and seq['protein']['msa']:
                            seq['protein']['msa'] = seq['protein']['msa'].replace('.npz', '.a3m')

                    with open(result_yaml, 'w') as f:
                        yaml.dump(data, f)

                    shutil.copy2(result_yaml, result_yaml_apo)
                    with open(result_yaml_apo, 'r') as f:
                        data_apo = yaml.safe_load(f)
                    data_apo['sequences'] = [data_apo['sequences'][chain_to_number[config['binder_chain']]]]
                    data_apo.pop('constraints', None)   

                    with open(result_yaml_apo, 'w') as f:
                        yaml.dump(data_apo, f)

                    if redo_boltz_predict:
                        subprocess.run([boltz_path, 'predict', str(result_yaml), '--out_dir', str(results_final_dir), '--write_full_pae'])                     
                        subprocess.run([boltz_path, 'predict', str(result_yaml_apo), '--out_dir', str(results_final_dir_apo), '--write_full_pae'])
                    else:
                        save_confidence_scores(results_final_dir, output, best_structure, f"{target_binder_input}_results_itr{itr + 1}_length{config['length']}", 0)
                        save_confidence_scores(results_final_dir_apo, output_apo, best_structure_apo, f"{target_binder_input}_results_itr{itr + 1}_length{config['length']}", 0)
                    gc.collect()
                    torch.cuda.empty_cache()
