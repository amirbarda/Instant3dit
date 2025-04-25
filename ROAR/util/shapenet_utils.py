import requests
from pathlib import Path
import numpy as np
from util.func import normalize_vertices, fix_mesh, save_mesh_properly, load_mesh, save_pc
from sampling.triangle_soup_to_pointcloud import triangle_soup_to_point_cloud_random
import json
# todo: add class_id to class_name utility function

def shapenetcore_preprocess(root_folder, output_folder, renderer, mesh_class="airplane", n_meshes=400, n_points=100000,
                            selection_mode='random', folder_per_instance=False):
    """
    preprocess shapenet meshes into output folder.
    mesh_class must be valid shapenetcore class
    n is number of meshes to extract from the class (randomly or sequentially chosen), if n==-1 than extract all meshes in class
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    url = "https://gist.githubusercontent.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e/raw/5064af3603d509b79229f6931998d4e197575ad3/shapenet_synset_list"
    response = requests.get(url)
    data = response.content.decode("utf-8").split()
    mesh_classes = data[1::2]
    mesh_classes_id = data[::2]
    my_mesh_id = mesh_classes_id[mesh_classes.index(mesh_class)]
    actual_class_dir = Path(root_folder, my_mesh_id)
    instances = np.array([x for x in sorted(actual_class_dir.glob("**/*.obj"))])
    if n_meshes == -1:
        n_meshes = instances.shape[0]
    np.random.seed(43)
    if selection_mode == 'random':
        selections = np.random.choice(instances, n_meshes, replace=False)
    elif selection_mode == 'ordered':
        selections = instances[:n_meshes]
    else:
        raise NotImplementedError('invalid selection mode for shapenet meshes')
    selection_names = []
    for i, instance in enumerate(selections):
        print("{} / {}".format(i, n_meshes))
        instance_name = instance.parent.parent.name
        if folder_per_instance:
            instance_dir = Path(output_folder, instance_name)
            if instance_dir.exists() and Path(instance_dir, "{}_original.obj".format(instance_name)).exists():
                continue
            instance_dir.mkdir(parents=True, exist_ok=True)
        else:
            instance_dir = Path(output_folder)
        selection_names.append(instance_name)
        mesh_vertices, mesh_faces, valid_mesh = load_mesh(instance, repair_dup_faces=True)
        assert valid_mesh
        mesh_vertices = normalize_vertices(mesh_vertices)
        save_mesh_properly(mesh_vertices, mesh_faces, str(Path(instance_dir, "{}_original.obj".format(instance_name))))
        mesh_vertices, mesh_faces = fix_mesh(renderer, mesh_vertices, mesh_faces)
        pc_vertices, pc_normals = triangle_soup_to_point_cloud_random(mesh_vertices, mesh_faces, n_points)
        save_pc(Path(instance_dir, "{}.ply".format(instance_name)), pc_vertices, pc_normals)
        save_mesh_properly(mesh_vertices, mesh_faces, str(Path(instance_dir, "{}.obj".format(instance_name))))
    np.savetxt(Path(output_folder, "selections.txt"), selection_names, fmt="%s")


def get_shapenet_map(shapenet_root, top_level_only = False):
    """
    :param shapenet_root: path to shapenet root
    :param top_level_only: get only the 55 base classes
    :return: a map (class id, class name)
    """
    shapenet_taxonomy_file = Path(shapenet_root, "taxonomy.json")
    with open(shapenet_taxonomy_file, "r") as f:
        data = json.load(f)
    mesh_classses_ids = []
    mesh_classes = []
    ids_to_ignore = []
    for class_dict in data:
        if top_level_only:
            if class_dict["synsetId"] in ids_to_ignore:
                ids_to_ignore.extend(class_dict["children"])
                continue
        mesh_classses_ids.append(class_dict["synsetId"])
        mesh_classes.append(class_dict["name"].split(",")[0])
        if top_level_only:
            ids_to_ignore.extend(class_dict["children"])
    return mesh_classes, mesh_classses_ids


def get_shapenet_class_id_from_name(shapenet_root, class_name):
    mesh_classes, mesh_classes_id = get_shapenet_map(shapenet_root)
    class_id = mesh_classes_id[mesh_classes.index(class_name)]
    return class_id

def get_shapenet_name_from_class_id(shapenet_root, class_id):
    mesh_classes, mesh_classes_id = get_shapenet_map(shapenet_root)
    class_name = mesh_classes[mesh_classes_id.index(class_id)]
    return class_name

