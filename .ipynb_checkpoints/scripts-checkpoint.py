import json
import glob
import os

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





def F_MAE(df):
        Ex = abs(df['fx_nn'] - df['fx_ref'])
        Ey = abs(df['fy_nn'] - df['fy_ref'])
        Ez = abs(df['fz_nn'] - df['fz_ref'])
        Err_tot = sum(Ex + Ey + Ez)
        mae = Err_tot /(len(df)*3)
        return mae



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


def plot_df(df, axs=None ,label=None, color=None, factor = 1, validation = True, train = True):
    try:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            try:
                axs[0].plot(df['#epoch'] * factor, df['VMAE/at'], color= color, alpha = 0.7 , ms = 6.0, label = label)
                axs[1].plot(df['#epoch'] * factor, df['VMAEF'], color= color, alpha = 0.7, ms = 6.0) #+  ' V'
            except:
                axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.7 , ms = 6.0, label = label)
                axs[1].plot(df['#epoch'] * factor, df['Val_MAEF'], color= color, alpha = 0.7, ms = 6.0)
        if train == True:
                axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.4, ms = 0.08)
                axs[1].plot(df['#epoch']* factor, df['MAEF'], color= color , alpha = 0.4, ms = 0.08, label = label)
    except:
        if not color:
            color = ('#%06X' % randint(0, 0xFFFFFF))
        if validation == True:
            axs[0].plot(df['#epoch'] * factor, df['Val_MAE/at'], color= color, alpha = 0.7 , ms = 6.0, label = label)
            axs[1].plot(df['#epoch'] * factor, df['Val_MAEF'], color= color, alpha = 0.7, ms = 6.0) #+  ' V'
        if train == True:
            axs[0].plot(df['#epoch']* factor, df['MAE/at'], color = color , alpha = 0.4, ms = 0.08)
            axs[1].plot(df['#epoch']* factor, df['MAEF'], color= color , alpha = 0.4, ms = 0.08, label = label)


def plot_setup(fig, axs, log = True, lim = True, x0lim = 10,  xlim = 1000, y0lim = 0.0, ylim = 0.3, fylim =0.55, title = None ):
    axs[0].legend(loc = 'upper right', prop={'size': 12})
    axs[1].legend(loc = 'upper right', prop={'size': 12})
    for ax in [axs[0], axs[1]]:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        leg = ax.legend(unique.values(), unique.keys(), loc='upper right')
        for line in leg.get_lines():
            line.set_linewidth(6.0)

    axs[0].set_title('Energy', fontsize= 18)
    axs[1].set_title('Force', fontsize = 18)
    axs[0].set_ylabel('MAE (eV)')
    axs[1].set_ylabel('MAE (eV/A)')
    fig.supxlabel('log Epochs(1000 steps each)')
    if lim:
        axs[0].set_xlim(x0lim,xlim)
        axs[0].set_ylim(y0lim, ylim)
        axs[1].set_xlim(x0lim,xlim)
        axs[1].set_ylim(y0lim, fylim)

    minor_ticks = np.arange(0, ylim, 0.005)
    axs[0].set_yticks(minor_ticks, minor=True)
    axs[0].grid(which='minor', axis = 'y', alpha=0.2)
    axs[0].grid(which='major', axis = 'y', alpha=0.7)

    minor_ticks = np.arange(0, fylim, 0.005)
    axs[1].set_yticks(minor_ticks, minor=True)
    axs[1].grid(which='minor', axis = 'y', alpha=0.2)
    axs[1].grid(which='major', axis = 'y', alpha=0.7)
    if log:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
    if title:
        fig.suptitle(title, fontsize=20)
    plt.show()





p = '/leonardo/pub/userexternal/mtaleblo/Carbon_trainings/1_similar_dataset/ID_test/'
f = 'epoch_250_step_250000_forces.dat'
df = read_dat(p, f)
mae = F_MAE(df)
print(mae)



p = '/leonardo/pub/userexternal/mtaleblo/Carbon_trainings'
dirs = ['dia_100','div_100','dia_1k', 'div_1k', 'dia_5k', 'div_5k'][2:4]
fig, axs = plt.subplots(1, 2, figsize = (15, 5))
for dir in dirs:
    plot_df(read_metrics(os.path.join(p , dir , 'training')), axs, dir, validation = True, train = True)
plot_setup(fig, axs, lim= True)




p = '/leonardo/pub/userexternal/mtaleblo/Carbon_trainings/'

x = [100, 1000, 5000]
dia_dirs = ['dia_100', 'dia_1k', 'dia_5k']
dia_epochs = [150, 250, 1000]
dia_dia_errors = []
dia_div_errors = []

for d, e in zip(dia_dirs, dia_epochs):
    path_aa = os.path.join(p, d, 'test_dia')
    aa = F_MAE(path_aa, f'epoch_{e}_step_{e*1000}_forces.dat')
    dia_dia_errors.append(aa)
    path_av = os.path.join(p, d, 'test_div')
    av = F_MAE(path_av, f'epoch_{e}_step_{e*1000}_forces.dat')
    dia_div_errors.append(av)

div_dirs = ['div_100', 'div_1k', 'div_5k']
div_epochs = [50, 50, 1000]
div_div_errors = []
div_dia_errors = []


for dv, ev in zip(div_dirs, div_epochs):
    path_vv = os.path.join(p, dv, 'test_div')
    vv = F_MAE(path_vv, f'epoch_{ev}_step_{ev*1000}_forces.dat')
    div_div_errors.append(aa)
    path_va = os.path.join(p, dv, 'test_dia')
    va = F_MAE(path_va, f'epoch_{ev}_step_{ev*1000}_forces.dat')
    div_dia_errors.append(av)


fig, axs = plt.subplots(1, 2, figsize = (12, 4))
axs[0].plot(x, dia_dia_errors, '*b', markersize = 12, label = 'tested on diamond')
axs[0].plot(x, dia_div_errors, '*r', markersize = 12, label = 'tested on diverse')
axs[1].plot(x, div_div_errors, '*b', markersize = 12, label = 'diamond')
axs[1].plot(x, div_dia_errors, '*r', markersize = 12, label = 'diverse')
axs[0].set_title('Trained on Diamond data')
axs[1].set_title('Trained on Diverse data')
axs[0].set_xlabel('Training set size')
axs[0].set_ylabel('MAE (eV/ A)')
axs[1].set_xlabel('Training set size')
axs[0].legend()
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid()
plt.show()



