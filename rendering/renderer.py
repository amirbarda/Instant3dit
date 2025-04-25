from rendering.background import SolidColorBackground
from rendering.material import DiffuseWithPointLightMaterial
from easydict import EasyDict as edict

def get_renderer(renderer_type, data_path, device):
    
    background = SolidColorBackground()
    background.configure()
    material = DiffuseWithPointLightMaterial()
    material.configure()

    if renderer_type == "mesh":
        from rendering.mesh_renderer import NVDiffRasterizer
        from ROAR.mesh import Mesh
        renderer = NVDiffRasterizer()
        geometry = Mesh(geom_args = edict({'init_mesh_mode': 'initial_mesh', 'init_mesh_info' : {'mesh_path': data_path, 'normalize': False}}))
        renderer.configure(geometry,material, background, device)
    else:
        raise NotImplementedError('ERROR: renderer of type {} does not exist!'.format(renderer_type))

    return renderer