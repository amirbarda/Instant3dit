python inference.py --prompt "a dog wearing a jacket" --mask_dilation 0 --azimuth_offset 135 --seed 9900 --config configs/inpainting_3d_sdxl.yaml \
--input_path "experiment_data/dog_jacket/mesh.obj" --mask_path "experiment_data/dog_jacket/mask.obj" --output_type "mv_grid_image" --output_path "experiments_just_mv"

python inference.py --prompt "a panda wearing a denim jacket" --mask_dilation 0 --azimuth_offset 20 --seed 10000 \
--input_path "experiment_data/panda_jacket/mesh.obj" --mask_path "experiment_data/panda_jacket/mask.obj" --output_type "mv_grid_image" --output_path "experiments_just_mv"

python inference.py --prompt "a cow wearing a cowboy hat" --mask_dilation 0 --flip_image --seed 42000 \
--input_path "experiment_data/cow_with_cowboy_hat/mesh.obj" --mask_path "experiment_data/cow_with_cowboy_hat/mask.obj" --output_type "mv_grid_image" --output_path "experiments_just_mv"

python inference.py --prompt "astronaut riding wearing a blue uniform riding a rocking horse" --mask_dilation 0.05 --seed 5000 \
--input_path "experiment_data/rocking_horse/mesh.obj" --mask_path "experiment_data/rocking_horse/mask.obj" --output_type "mv_grid_image" --output_path "experiments_just_mv"

python inference.py --prompt "a roman soldier riding a rocking horse" --mask_dilation 0.05 --seed 9900 \
--input_path "experiment_data/rocking_horse/mesh.obj" --mask_path "experiment_data/rocking_horse/mask.obj" --output_type "mv_grid_image" --output_path "experiments_just_mv"

python inference.py --prompt "a whale riding a pink surfboard" --mask_dilation 0.00 --azimuth_offset 20 --seed 5000 \
--input_path "experiment_data/whale_surfboard/mesh.obj" --mask_path "experiment_data/whale_surfboard/mask.obj" --output_type "mv_grid_image" --output_path "experiments_just_mv"

python inference.py --prompt "a tanuki riding a beagle" --mask_dilation 0.00 --azimuth_offset -40 --seed 9900 \
--input_path "experiment_data/dog_tanuki/mesh.obj" --mask_path "experiment_data/dog_tanuki/mask.obj" --output_type "mv_grid_image" --output_path "experiments_just_mv"

python inference.py --prompt "a scarecrow riding a rocking horse" --mask_dilation 0.05 --seed 45345 \
--input_path "experiment_data/rocking_horse/mesh.obj" --mask_path "experiment_data/rocking_horse/mask.obj" --output_type "mesh" --output_path "experiments_just_mv"

