import h5py
import numpy as np
import pandas as pd
import os

def mm_eos_directory_to_h5(
        directory_path, indices_to_use, h5_outpath, 
        eos_per_mm=15,
        mm_template = lambda index: f"EoSNewRestricted-{index}",
        eos_indices = None,
        cherry_pick = False):
    h5file = h5py.File(h5_outpath, "w")
    # ns = h5file.create_group("ns")
    eos_group = h5file.create_group("eos")
    mm_ids  = []
    counter = 0
    print(f"Saving {np.size(indices_to_use)} EoSs")

    ### Allowing for cherry picking EoSs
    if cherry_pick:
        for i in range(np.size(indices_to_use)):
            mm_index = indices_to_use[i]
            eos_index = eos_indices[i]
    
            sample_id = eos_index - eos_per_mm * mm_index
        
            try:
                print(os.path.join(directory_path, mm_template(mm_index),
                                               f"eos-draw-{sample_id:04d}.csv"))
                try:
                    eos = pd.read_csv(os.path.join(directory_path, mm_template(mm_index),
                                                    f"eos-draw-{sample_id:04d}.csv"))
                except:
                    print("EOS read in failure.")

                print(type(eos.to_records()))

                eos_group[f"eos_{eos_index:06d}"] =  eos.to_records()
                mm_ids.append(mm_index)
            except FileNotFoundError:
                print("metamodel id", mm_index, "extension #", sample_id, "not found" )
    else:
        for index in indices_to_use:
            for sample_id in range(eos_per_mm):
                try:
                    print(os.path.join(directory_path, mm_template(index),
                                                   f"eos-draw-{sample_id:04d}.csv"))
                    try:
                        eos = pd.read_csv(os.path.join(directory_path, mm_template(index),
                                                        f"eos-draw-{sample_id:04d}.csv"))
                    except:
                        print("EOS read in failure.")
    
                    # try:
                    #     macro = pd.read_csv(os.path.join(directory_path, mm_template(index),
                    #                                 f"macro-eos-draw-{sample_id:04d}.csv"))
                    # except:
                    #     print("Macro read in failure.")
                    print(type(eos.to_records()))
    
                    # ns[f"eos_{counter:06d}"] = macro.to_records()
                    eos_group[f"eos_{counter:06d}"] =  eos.to_records()
                    mm_ids.append(index)
                    counter += 1
                except FileNotFoundError:
                    print("metamodel id", index, "extension #", sample_id, "not found" )
    eos_id = h5file.create_dataset("id", data=np.arange(counter))
    mm_id = h5file.create_dataset("mm_id", data=np.array(mm_ids))
    h5file.close()

# ### Function to save a subset of EoSs in a larger h5 file to a smaller h5 file
# def h5_subset_to_h5(directory_path, indices_to_use, h5_in_name, h5_out_name, mm_template = lambda index: f"EoSNewRestricted-{index}"):
#     # Reading data from larger h5 file
#     h5_inpath = f"{directory_path}/{h5_in_name}.h5"
#     h5_infile = h5py.File(h5_inpath, "r+")
#     eos_data = h5_infile["eos"]

#     # Creating smaller h5 file to write data to
#     h5_outpath = f"{directory_path}/{h5_out_name}.h5"
#     h5_outfile = h5py.File(h5_outpath, "w")
#     eos_group = h5_outfile.create_group("eos")
#     mm_ids  = []
#     for index in indices_to_use:
#         try:
#             # eos = eos_data[f"eos_{index:06d}"]

#             sample_id = index - 15 * mm_id
#             eos = pd.read_csv(os.path.join(directory_path, mm_template(index), f"eos-draw-{sample_id:04d}.csv"))
#         except:
#             print("EOS read in failure.")

#         print(type(pd.DataFrame(eos[:])))

#         eos_group[f"eos_{index:06d}"] = pd.DataFrame(eos[:])
#         mm_ids.append(index)
#     eos_id = h5_outfile.create_dataset("id", data=np.array(indices_to_use))
#     mm_id = h5_outfile.create_dataset("mm_id", data=np.array(mm_ids))
#     h5_outfile.close()

if __name__ == "__main__":

    project_path = "/home/ryan.krismer/CSUF_EoS_project/ns_dense_matter/conformal_limit"
    eos_path = f"{project_path}/all_eoss"
    # num_eos = int(len(os.listdir(eos_path)))
    output_file = "conformal.h5"

    ### Only using conformal EoSs
    eos_indices = pd.read_csv(f"{project_path}/astro_likelihoods/conformal_weights.csv")["eos_index"].to_numpy()
    mm_indices = eos_indices // 15
    
    mm_eos_directory_to_h5(eos_path,
                           indices_to_use = mm_indices,
                           h5_outpath=f"{project_path}/{output_file}",
                           eos_indices = eos_indices,
                           cherry_pick = True
                          )