# Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects (CVPR 2025)

[Project Page](https://amirbarda.github.io/Instant3dit.github.io/) | [Paper](https://arxiv.org/pdf/2412.00518) |

This is the official implementation of Instant3dit. We provide the weights and inference code for the multiview 3d inpainting network that allows fast editing of 3d objects, by reconstruction to various representations using corresponding LRMs (Large Reconstruction Models).

![alt text](https://github.com/amirbarda/Instant3dit/blob/main/assets/overview.png?raw=true)

## Multiview Edited Image Generation
The code has been tested on python 3.8 and 3.10 with pytorch 2.1.2 and 2.7.0, both with cuda 11.8, but should work for all versions in between.
1. run `pip install requirements.txt` to install dependencies
2. download the multiview inpainting [SDXL weights](https://drive.google.com/drive/folders/1yLdhgEqv0FBD19r4RPBsBzpa3congkDv?usp=sharing)
3. replace Path/to/Instant3dit_model in the default argument with the path to the SDXL multiview inpainting checkpoint folder downloaded in the previous step.

**Note:** We use the diffusers library, so you must have a Huggingface [access token](https://huggingface.co/docs/hub/en/security-tokens), in a file called TOKEN, at the root of the project.

## Reconstructing using LRMs (Large Reconstruction Models)
**Disclaimer:** The results in the paper were obatined used internal Adobe LRMs for reconstruction to various 3d representations (NeRF, meshes and 3DGS). 
We substitute this with the best open source offerings we could find. Currently, these are not on par with the Adobe models. Newer and more powerful open source LRMs can be integrated in the future (PRs welcome).

We allow for using these LRMS seamlessly in our inference code.

### mesh LRM
We use [InstantMesh](https://github.com/TencentARC/InstantMesh) for mesh reconstruction, all the required dependencies are already in `requirements.txt`. \
locally clone InstantMesh: `git clone git@github.com:TencentARC/InstantMesh.git`
replace Path/to/InstantMesh in the default argument for instantmesh_path with the path to the InstantMesh folder

### 3D Gaussian Splats LRM
We use [geoLRM](https://github.com/alibaba-yuanjing-aigclab/GeoLRM) for 3DGS reconstruction, To install, after installing all the dependencies in requirements.txt, run: \
`pip install flash-attn --no-build-isolation` \
`pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git` \
`pip install git+https://github.com/Stability-AI/generative-models.git` \
(**Note**: installing flash-attn may take a while) \
locally clone geoLRM: `git clone git@github.com:alibaba-yuanjing-aigclab/GeoLRM.git`
replace Path/to/geoLRM in the default argument for geoLRM_path with the path to the geoLRM folder

## TODO:
- [ ] adaptive remeshing pipeline
- [ ] texturing pipeline
- [ ] training code
- [ ] training dataset