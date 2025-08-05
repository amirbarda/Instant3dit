import os
import json

mask_root = 'training_data/masks'
output_dir = 'training_data'
# save as json file with
num_masks = 0
masks_paths = {'root' : mask_root}
for mask_type in os.listdir(mask_root):
    for prefix in os.listdir(os.path.join(mask_root, mask_type)):
        for uid in os.listdir(os.path.join(mask_root, mask_type, prefix)):
            full_path = os.path.join(mask_root, mask_type, prefix, uid)
            mask_list = []
            for mask_folder in os.listdir(full_path):
                mask_list.append(mask_folder.split('_')[1])
                mask_list = sorted(mask_list, key= lambda x : int(x))
            if os.path.join(prefix, uid) not in masks_paths:
                masks_paths[os.path.join(prefix, uid)] = [({'mask_type': mask_type , 'masks' : mask_list})]
            else:       
                masks_paths[os.path.join(prefix, uid)].append({'mask_type': mask_type , 'masks' : mask_list})
            num_masks += 1


print (num_masks)
with open(os.path.join(output_dir,'mask_data.json'), 'w') as f:
    json.dump(masks_paths, f)
pass