# The following API functions are defined:
#  3DCompat       - 3DCompat api class that loads 3DCompat annotation file and prepare data structures.
#  load_stylized_3d - load stylized 3d shapes with the specified ids.

_ALL_CLASSES = []
_ALL_PARTS = []


class 3DCompat:
    def __init__(self, meta_file=None, data_folder=None):
        """
        Constructor of 3DCompat helper class for reading and visualizing annotations.
        :param meta_file (str): location of meta file
        :param data_folder (str): location to the folder that hosts data.
        :return:
        """
        model.csv
        label.json
        part_material_map.json

	def load_raw_models(self, shape_id):

	def show_raw_models(self, shape_id):

	def load_stylized_3d(self, shape_id, style_id):
		gltf_path = os.path.join(shape_id, '_', style_id)
		mesh = trimesh.load(gltf_path)

	    return mesh

	def show_stylized_3d(self, stylized_3d):


	def load_stylized_2d(self, shape_id, style_id, view_id):


	def show_stylized_2d(self, stylized_2d):




