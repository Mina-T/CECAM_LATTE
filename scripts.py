import json
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint


def convert_multiple_json_to_extxyz(examples_dir, output_name):
    """
    Convert multiple JSON files to a single extxyz file
    
    Parameters:
    - examples_dir: path to example files
    - output_name: name for saving the output file
    """   
    example_files =  [f for f in os.listdir(examples_dir) if f.endswith('.example')]    
    if not example_files:
        print("No example files found:")
        return
    
    print(f"Found {len(example_files)} example files to convert")
    dataset = np.array([], dtype=object)   
    with open(output_name, 'w') as outfile:
        for exp in example_files:
            with open(exp, 'r') as infile:
                data = json.load(infile)
                
                atoms = data.get("atoms", [])
                energy_value, energy_unit = data.get("energy", [0.0, "unknown"])
                lattice_vectors = data.get("lattice_vectors", [[0,0,0], [0,0,0], [0,0,0]])
    
                outfile.write(f"{len(atoms)}\n")
                
                lattice_str = " ".join([f"{vec[0]} {vec[1]} {vec[2]}" for vec in lattice_vectors])
                outfile.write(f'Lattice="{lattice_str}" energy={energy_value} unit={energy_unit} pbc="T T T" \n')
                
                for atom in atoms:
                    _, element, position, atomic_forces = atom
                    x, y, z = position 
                    fx, fy, fz = atomic_forces
                    outfile.write(f"{element} {x:.10f} {y:.10f} {z:.10f} {fx:.10f} {fy:.10f} {fz:.10f} \n")
                
            outfile.write("\n")
            
    print(f"\nCombined file saved as {output_name}")
    print(f"Total example files: {len(example_files)}")


def Force_MAE(path, epoch):
    _file = f'epoch_{epoch}_step_{100*epoch}_forces.dat'
    df = read_dat(path, _file)
    Ex = abs(df['fx_nn'] - df['fx_ref'])
    Ey = abs(df['fy_nn'] - df['fy_ref'])
    Ez = abs(df['fz_nn'] - df['fz_ref'])
    Err_tot = sum(Ex + Ey + Ez)
    mae = Err_tot /(len(df)*3)
    return mae

def Energy_MAE(path,epoch):
    _file = f'epoch_{epoch}_step_{100*epoch}.dat'
    df = read_dat(path, _file)
    mae = abs((df['e_ref']/df['n_atoms']) - (df['e_nn']/df['n_atoms']))
    mae = sum(mae)/len(mae)
    return mae

def read_metrics(path):
    '''
    Given a path, converts the metrics.dat into pd.df
    '''
    df= pd.read_csv(path + '/metrics.dat',  sep=r'\s+')
    df = df.groupby(['#epoch'], as_index=False).last()
    return df

def read_dat(path, _file):
    '''
    Given a path, converts the *.dat into pd.df
    '''
    if path[-1] != '/':
        path = path + '/'

    full_path = path + _file

    if not os.path.exists(full_path):
        print(f"[Warning] File not found: {full_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(full_path, sep=r'\s+')
        if df.empty:
            print(f"[Warning] Empty file: {full_path}")
            return pd.DataFrame()
        return df

    except pd.errors.EmptyDataError:
        print(f"[Warning] Empty or unreadable file: {full_path}")
        return pd.DataFrame()


def plot_training(training_path, axs=None, label=None, color=None, factor=1, validation=True, train=True):
    df = read_metrics(training_path)
    if axs is None:
        raise ValueError("axs must be provided (list/array of 2 axes)")

    if color is None:
        color = f'#{randint(0, 0xFFFFFF):06X}'

    x = df['#epoch'] * factor

    val_cols = [
        ('VMAE/at', 'VMAEF'),
        ('Val_MAE/at', 'Val_MAEF')
    ]

    train_cols = ('MAE/at', 'MAEF')

    if validation:
        for c1, c2 in val_cols:
            if c1 in df.columns and c2 in df.columns:
                axs[0].plot(x, df[c1], color=color, alpha=0.7,
                            linewidth=1.5, label=label)
                axs[1].plot(x, df[c2], color=color, alpha=0.7,
                            linewidth=1.5)
                break  

    if train:
        if all(c in df.columns for c in train_cols):
            axs[0].plot(x, df[train_cols[0]], color=color,
                        alpha=0.4, linewidth=1.0)
            axs[1].plot(x, df[train_cols[1]], color=color,
                        alpha=0.4, linewidth=1.0, label=label)


def plot_setup(fig, axs, log=True, lim=True,
               xlim=100,
               ylim=0.1, fylim=0.2,
               title=None):

    axs[0].set_title('Energy', fontsize=18)
    axs[1].set_title('Force', fontsize=18)

    axs[0].set_ylabel('MAE (eV)')
    axs[1].set_ylabel('MAE (eV/Å)') 
    fig.supxlabel('Epochs (log scale, 100 steps each)')

    if lim:
        axs[0].set(xlim=(1, xlim), ylim=(0, ylim))
        axs[1].set(xlim=(1, xlim), ylim=(0, fylim))

    if log:
        for ax in axs:
            ax.set_xscale('log')

    axs[0].set_yticks(np.arange(1, ylim, 0.005), minor=True)
    axs[1].set_yticks(np.arange(1, fylim, 0.005), minor=True)

    for ax in axs:
        ax.grid(which='major', axis='y', alpha=0.7)
        ax.grid(which='minor', axis='y', alpha=0.2)

    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        if unique:
            leg = ax.legend(unique.values(), unique.keys(),
                            loc='upper right', fontsize=12)

            for line in leg.get_lines():
                line.set_linewidth(3.0)

    if title is not None:
        fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()


def find_project_root(target="CECAM_LATTE"):
    cwd = Path.cwd()

    for parent in [cwd] + list(cwd.parents):
        if parent.name == target:
            return parent
            
    for p in cwd.rglob(target):
        if p.is_dir():
            return p

    raise RuntimeError(f"{target} not found anywhere")




