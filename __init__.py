from .AnyText.utils import is_folder_exist
import folder_paths
import os

#加载插件前先检查是否在os.listdir里存在自定义目录，没有则自动创建，防止加载节点失败，官方目录可无视。
fonts_path = os.path.join(folder_paths.models_dir, 'fonts')
translator_path = os.path.join(folder_paths.models_dir, 'prompt_generator')
comfy_temp_dir = folder_paths.get_temp_directory()
if not is_folder_exist(fonts_path):
    os.makedirs(fonts_path)
if not is_folder_exist(translator_path):
    os.makedirs(translator_path)
if not is_folder_exist(comfy_temp_dir):
    os.makedirs(comfy_temp_dir)
    
# only import if running as a custom node
try:
	pass
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}
 
 	# AnyText
	from .AnyText.nodes import NODE_CLASS_MAPPINGS as AnyText_Nodes
	NODE_CLASS_MAPPINGS.update(AnyText_Nodes)
 
	# AnyText_utils
	from .AnyText.utils import NODE_CLASS_MAPPINGS as AnyText_loader_Nodes
	NODE_CLASS_MAPPINGS.update(AnyText_loader_Nodes)
 
	NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
