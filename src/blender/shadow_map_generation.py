import os
import json
import math

from tqdm import tqdm
import numpy as np

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix
except ImportError:
    print("This script must be run from Blender")
    quit()


DEBUG = os.getenv('DEBUG', '0') == "1"
COMPUTE_DEVICE_TYPE = 'OPTIX'
NUM_SAMPLES_CYCLES = 4096
DEPTH_MODEL_NAME = 'depthanythingv2_relative_matched'
BGMESH_DIR = os.path.join(os.getenv('SCRATCH'), 'GT_emission_envmap_depthanythingv2_relative_obj')
OBJECTS_DIR = os.path.join(os.getenv('SCRATCH'), 'objects_out')

RENDER_ENGINE = os.getenv('RENDER_ENGINE', 'BLENDER_EEVEE') # BLENDER_EEVEE or CYCLES

LIGHT_ENERGY = 1000.0


def setup_render_properties(num_samples_cycles=NUM_SAMPLES_CYCLES):
    bpy.context.scene.render.engine = RENDER_ENGINE
    if RENDER_ENGINE == 'CYCLES':
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = COMPUTE_DEVICE_TYPE
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # 
        found_right_device = False
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            print(d.name, d.type, d.use)
            if d.type == COMPUTE_DEVICE_TYPE:
                # on my local Windows machine with GTX 1060, works better with only the GPU, not GPU+CPU
                found_right_device = True
                print("Using", d.name, "for rendering")
                d.use = True
            else:
                d.use = False
    # 
        if not found_right_device:
            raise Exception("No suitable device found")

        # bpy.context.scene.cycles.use_adaptive_sampling = True
        # bpy.context.scene.cycles.adaptive_threshold = noise_threshold_cycles  # mesh rendering seems to require 0.01, while envmap, only 0.1
        bpy.context.scene.cycles.samples = num_samples_cycles  # TODO: 4096 for eval

        bpy.context.scene.cycles.use_denoising = True
            
        # # OPENIMAGEDENOISE doesn't have Optix's problem with rendering shadow catcher with high exposure
        # bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        # bpy.context.scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    else:
        bpy.context.scene.eevee.taa_render_samples = 64
        bpy.context.scene.eevee.shadow_cube_size = '4096'
        bpy.context.scene.eevee.shadow_cascade_size = '4096'
        bpy.context.scene.eevee.use_shadow_high_bitdepth = True
        bpy.context.scene.eevee.use_soft_shadows = True

# 

    # Avoid rendering the environment map as background, only use it for IBL
    bpy.context.scene.render.film_transparent = True


def import_object(object_absolute_path):
    if object_absolute_path.endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=object_absolute_path)
    elif object_absolute_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=object_absolute_path)
    elif object_absolute_path.endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=object_absolute_path)
    else:
        raise Exception("Unsupported object format", object_absolute_path)

    # swtich to object mode
    # bpy.context.view_layer.update()
    imported_obj = bpy.context.selected_objects[0]
    # object_name = [ob.name for ob in bpy.context.scene.objects if ob.data is not None and ob.name not in ['Camera', 'Plane']][0]
    return imported_obj


def move_object_to_collection(obj, col):
    # Move object to collection
    for other_col in obj.users_collection:
        other_col.objects.unlink(obj)
    if obj.name not in col.objects:
        col.objects.link(obj)


def find_mesh_object(obj):
    if obj.type == 'MESH':
        return obj
    for child in obj.children:
        mesh_obj = find_mesh_object(child)
        if mesh_obj:
            return mesh_obj
    return None

def move_pivot(obj):
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select the object and set it as the active object
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Enter edit mode to access the mesh data
    bpy.ops.object.mode_set(mode='EDIT')

    # Create a BMesh representation of the mesh
    bm = bmesh.from_edit_mesh(obj.data)

    # Get the world matrix of the object
    world_matrix = obj.matrix_world

    # Calculate the bounding box min and max
    bbox_min = Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))

    for vert in bm.verts:
        co = world_matrix @ vert.co
        bbox_min.x = min(bbox_min.x, co.x)
        bbox_min.y = min(bbox_min.y, co.y)
        bbox_min.z = min(bbox_min.z, co.z)
        bbox_max.x = max(bbox_max.x, co.x)
        bbox_max.y = max(bbox_max.y, co.y)
        bbox_max.z = max(bbox_max.z, co.z)

    # Calculate the center bottom of the bounding box
    center_bottom = Vector(((bbox_min.x + bbox_max.x) / 2, (bbox_min.y + bbox_max.y) / 2, bbox_min.z))

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Set the origin to the calculated center bottom point
    bpy.context.scene.cursor.location = center_bottom
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    # Clear the cursor's location
    obj.location = (0, 0, 0)
    bpy.context.scene.cursor.location = (0, 0, 0)


def get_bbox(obj):
    # Get the bounding box of the background mesh, and uniformly sample points within it
    obj_to_world = np.array(obj.matrix_world)
    # Convert the bounding box to homogeneous coordinates numpy array, and to world space
    mesh_obj = find_mesh_object(obj)
    obj_bbox_local = np.array(mesh_obj.bound_box)
    obj_bbox_local = np.hstack([obj_bbox_local, np.ones((8, 1))])
    obj_bbox = (obj_to_world @ obj_bbox_local.T).T
    obj_bbox = obj_bbox[:, :3]
    return obj_bbox

def render_scene(name, shadow_map_name, blender_dto_path, output_dir, light_position_obj_space, *, light_radius=0.0, hide_object=False):
    
    # DEBUG = True
    # Load blender file
    blender_file = 'src/blender/shadowmap.blend'
    bpy.ops.wm.open_mainfile(filepath=blender_file)

    setup_render_properties()


    # name = '9C4A1774-12d053d172_06_crop_B07QFB1TMP'
    bg_mesh_path = os.path.join(BGMESH_DIR, name, 'obj', f'{name}_{DEPTH_MODEL_NAME}.obj')
    # output_dir = os.path.join(OUTPUT_MAIN_DIR, name, 'shadowmap')
    # output_scene_dir = os.path.join(OUTPUT_SCENE_DIR, name, 'shadowmap')
    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(output_scene_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Remove the objects
    for comp_obj in bpy.context.scene.objects:
        if comp_obj.name not in ['Camera', 'Point']:
            bpy.data.objects.remove(comp_obj)

    # blender_dto_path = os.path.join(folder, f'{name}_blender_dto.json')
    with open(blender_dto_path, 'r') as f:
        blender_dto = json.load(f)

    # Extract the camera and object properties
    camera_rotation = blender_dto['camera']['rotation_world_space']
    obj_relative_path = blender_dto['object']['relative_path']
    obj_position = blender_dto['object']['position']
    obj_rotation = blender_dto['object']['rotation_world_space']
    obj_scale = blender_dto['object']['scale']

    # Set the camera properties
    camera = bpy.context.scene.objects['Camera']
    camera.rotation_euler = camera_rotation

    # Load the background mesh
    bg_obj = import_object(bg_mesh_path)
    bg_obj.name = 'BgMesh'
    # Set the background mesh properties
    bg_obj.location = (0, 0, 0)
    bg_obj.rotation_mode = 'XYZ'
    bg_obj.rotation_euler = (camera_rotation[0], math.pi, camera_rotation[2] + math.pi)
    bg_obj_mesh = find_mesh_object(bg_obj)
    if RENDER_ENGINE == 'CYCLES':
        bg_obj_mesh.is_shadow_catcher = True
        bg_obj_mesh.visible_camera = True
        bg_obj_mesh.visible_diffuse = False
        bg_obj_mesh.visible_glossy = False
        bg_obj_mesh.visible_transmission = False
        bg_obj_mesh.visible_volume_scatter = False
        bg_obj_mesh.visible_shadow = False
    else:
        # add material that doesn't cast shadow
        mat = bpy.data.materials.new(name="NoShadow")
        # use nodes
        mat.use_nodes = False
        mat.diffuse_color = (1, 1, 1, 1)
        mat.shadow_method = 'NONE'
        mat.blend_method = 'BLEND'
        # link to the bg
        bg_obj.data.materials.append(mat)


    # Load the object
    object_path = os.path.join(OBJECTS_DIR, obj_relative_path)
    comp_obj = import_object(object_path)
    comp_obj.name = 'ObjMesh'
    # Set the object pivot to the bottom center
    comp_obj_mesh = find_mesh_object(comp_obj)
    move_pivot(comp_obj_mesh)
    comp_obj.location = obj_position
    # Set the object properties
    comp_obj.rotation_mode = 'XYZ'
    comp_obj.rotation_euler = obj_rotation
    comp_obj.scale = (obj_scale, obj_scale, obj_scale)
    comp_obj_mesh = find_mesh_object(comp_obj)

    # Set holdout material
    mat_holdout = bpy.data.materials.new(name="HoldoutMaterial")
    mat_holdout.use_nodes = True
    # clear default nodes
    mat_holdout.node_tree.nodes.clear()
    holdout_node = mat_holdout.node_tree.nodes.new('ShaderNodeHoldout')
    material_output = mat_holdout.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat_holdout.node_tree.links.new(holdout_node.outputs['Holdout'], material_output.inputs['Surface'])
    comp_obj_mesh.data.materials.clear()
    comp_obj_mesh.data.materials.append(mat_holdout)

    # EEVEE only - modify shadow mode and hide render behavior
    if RENDER_ENGINE == 'BLENDER_EEVEE':
        if hide_object:
            mat_holdout.shadow_method = 'NONE'
        else:
            mat_holdout.shadow_method = 'OPAQUE'
    else:
        assert not hide_object

    # If don't add this, the later code cannot get the newest location of the object
    bpy.context.view_layer.update()

    # Render settings
    render_settings = bpy.context.scene.render
    render_settings.filepath = os.path.abspath(output_dir)

    # Set the point light position, and render the scene
    point_light = bpy.context.scene.objects['Point']

    obj_center = np.mean(get_bbox(comp_obj), axis=0)

    # point_light.location = obj_center + light_position_obj_space

    matrix_cancel_camera_rot_x = Matrix([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    point_light.location = obj_center + camera.matrix_world @ matrix_cancel_camera_rot_x @ Vector(light_position_obj_space)
    point_light.data.shadow_soft_size = light_radius
    point_light.data.energy = LIGHT_ENERGY

    if RENDER_ENGINE == 'BLENDER_EEVEE':
        # Fixes missing shadows close to the object
        # The bias is not needed, since we don't have self-shadowing in our setup (the object casts shadows only on the background)
        point_light.data.shadow_buffer_bias = 0.0
        point_light.data.shadow_buffer_clip_start = 0.01 # TODO: optimize for precision and non-clipping

    # Output the shadow map
    nodes = bpy.context.scene.node_tree.nodes
    file_output_node = nodes['FileOutput']
    file_output_node.base_path = os.path.join(os.path.abspath(output_dir))
    file_output_node.file_slots.clear()
    file_name = f'ShadowMap_{shadow_map_name}_'
    file_output_node.file_slots.new(file_name)
    file_output_node.format.file_format = 'OPEN_EXR'
    render_layers_node = nodes['Render Layers']
    if RENDER_ENGINE == 'CYCLES':
        bpy.context.scene.node_tree.links.new(render_layers_node.outputs['Alpha'], file_output_node.inputs[file_name])
    else:
        bpy.context.scene.node_tree.links.new(render_layers_node.outputs['Image'], file_output_node.inputs[file_name])

    bpy.context.view_layer.update()

    bpy.ops.render.render()

    # Save the blender file
    if DEBUG:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(output_dir, f'shadowmap_{name}.blend'))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Render shadow maps for a scene.")
    # parser.add_argument("scene_path", type=str, help="Path to the scene JSON file.")
    # parser.add_argument("light_position", type=float, nargs=3, help="3D position of the light source.")
    # args = parser.parse_args()

    # scene_path = 'GT_emission_envmap_depthanythingv2_relative_obj/'
    # scene_path = r'../compositing-dataset-creation-blender\outputs\singleimagelightpredictions\rendered_crops\GT_emission_envmap'
    name = os.getenv('SCENE_NAME')
    shadow_map_name = os.getenv('SHADOW_MAP_NAME')
    blender_dto_path = os.path.join(os.getenv('DATASET_DIR'), name, f'{name}_blender_dto.json')
    
    # output_dir = os.path.join(OUTPUT_MAIN_DIR, name, 'shadowmap')
    output_dir = os.getenv('OUTPUT_DIR')
    hide_object = os.getenv('HIDE_OBJECT', 'False') == 'True'
    light_position = np.array([float(os.getenv('LIGHT_X')), float(os.getenv('LIGHT_Y')), float(os.getenv('LIGHT_Z'))])
    render_scene(name, shadow_map_name, blender_dto_path, output_dir, 
                 light_position_obj_space=light_position, light_radius=float(os.getenv('LIGHT_RADIUS', 0.0)), hide_object=hide_object)
