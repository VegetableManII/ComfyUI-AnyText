import os
import folder_paths
import re
import cv2
import numpy as np
from .utils import is_module_imported, pil2tensor, get_device_by_name, comfy_tensor_Image2np_Image

comfy_temp_dir = folder_paths.get_temp_directory()
Random_Gen_Mask_path = os.path.join(comfy_temp_dir,  "AnyText_random_mask_pos_img.png")
tmp_pose_img_path = os.path.join(comfy_temp_dir, "AnyText_manual_mask_pos_img.png")
tmp_ori_img_path = os.path.join(comfy_temp_dir, "AnyText_ori_img.png")

class AnyText:
  
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AnyText_Loader": ("AnyText_Loader", {"forceInput": True}),
                "prompt": ("STRING", {"default": "A raccoon stands in front of the blackboard with the words \"你好呀~Hello!\" written on it.", "multiline": True}),
                "a_prompt": ("STRING", {"default": "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks", "multiline": True}),
                "n_prompt": ("STRING", {"default": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture", "multiline": True}),
                "mode": (['text-generation', 'text-editing'],{"default": 'text-generation'}),  
                "sort_radio": (["↕", "↔"],{"default": "↔"}), 
                "revise_pos": ("BOOLEAN", {"default": False}),
                "img_count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "ddim_steps": ("INT", {"default": 20, "min": 2, "max": 100}),
                "seed": ("INT", {"default": 9999, "min": -1, "max": 99999999}),
                "nonEdit_random_gen_width": ("INT", {"default": 512, "min": 128, "max": 1920, "step": 64}),
                "nonEdit_random_gen_height": ("INT", {"default": 512, "min": 128, "max": 1920, "step": 64}),
                # "width": ("INT", {"forceInput": True}),
                # "height": ("INT", {"forceInput": True}),
                "Random_Gen": ("BOOLEAN", {"default": False}),
                "strength": ("FLOAT", {
                    "default": 1.00,
                    "min": -999999,
                    "max": 9999999,
                    "step": 0.01
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 9,
                    "min": 1,
                    "max": 99,
                    "step": 0.1
                }),
                "eta": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 0.1
                }),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
                "fp16": ("BOOLEAN", {"default": True}),
                "cpu_offload": ("BOOLEAN", {"default": False, "label_on": "model_to_cpu", "label_off": "unload_model"}),
                "all_to_device": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                        "ori_image": ("IMAGE", {"forceInput": True}),
                        "pos_image": ("IMAGE", {"forceInput": True}),
                        # "show_debug": ("BOOLEAN", {"default": False}),
                        },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "anytext_process"
    TITLE = "AnyText Geneation"

    def anytext_process(self,
        mode,
        AnyText_Loader,
        ori_image,
        pos_image,
        sort_radio,
        revise_pos,
        Random_Gen,
        prompt, 
        cpu_offload,
        # show_debug, 
        img_count, 
        fp16,
        device,
        all_to_device,
        ddim_steps=20, 
        strength=1, 
        cfg_scale=9, 
        seed="", 
        eta=0.0, 
        a_prompt="", 
        n_prompt="", 
        nonEdit_random_gen_width=512, 
        nonEdit_random_gen_height=512,
    ):
        def prompt_replace(prompt):
            prompt = prompt.replace('“', '"')
            prompt = prompt.replace('”', '"')
            p = '"(.*?)"'
            strs = re.findall(p, prompt)
            if len(strs) == 0:
                strs = [' ']
            else:
                for s in strs:
                    prompt = prompt.replace(f'"{s}"', f' * ', 1)
            return prompt
        
        def check_overlap_polygon(rect_pts1, rect_pts2):
            poly1 = cv2.convexHull(rect_pts1)
            poly2 = cv2.convexHull(rect_pts2)
            rect1 = cv2.boundingRect(poly1)
            rect2 = cv2.boundingRect(poly2)
            if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
                return True
            return False
        
        def count_lines(prompt):
            prompt = prompt.replace('“', '"')
            prompt = prompt.replace('”', '"')
            p = '"(.*?)"'
            strs = re.findall(p, prompt)
            if len(strs) == 0:
                strs = [' ']
            return len(strs)
        
        def generate_rectangles(w, h, n, max_trys=200):
            img = np.zeros((h, w, 1), dtype=np.uint8)
            rectangles = []
            attempts = 0
            n_pass = 0
            low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
            while attempts < max_trys:
                rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
                ratio = np.random.uniform(4, 10)
                rect_h = max(low_edge, int(rect_w/ratio))
                rect_h = min(rect_h, int(h*0.8))
                # gen rotate angle
                rotation_angle = 0
                rand_value = np.random.rand()
                if rand_value < 0.7:
                    pass
                elif rand_value < 0.8:
                    rotation_angle = np.random.randint(0, 40)
                elif rand_value < 0.9:
                    rotation_angle = np.random.randint(140, 180)
                else:
                    rotation_angle = np.random.randint(85, 95)
                # rand position
                x = np.random.randint(0, w - rect_w)
                y = np.random.randint(0, h - rect_h)
                # get vertex
                rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
                rect_pts = np.int32(rect_pts)
                # move
                rect_pts += (x, y)
                # check boarder
                if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
                    attempts += 1
                    continue
                # check overlap
                if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles): # type: ignore
                    attempts += 1
                    continue
                n_pass += 1
                img = cv2.fillPoly(img, [rect_pts], 255)
                cv2.imwrite(Random_Gen_Mask_path, 255-img[..., ::-1])
                rectangles.append(rect_pts)
                if n_pass == n:
                    break
                print("attempts:", attempts)
            if len(rectangles) != n:
                raise Exception(f'Failed in auto generate positions after {attempts} attempts, try again!')
            return img
        
        if not is_module_imported('AnyText_Pipeline'):
            from .AnyText_scripts.AnyText_pipeline import AnyText_Pipeline
        
        #check if prompt is chinese to decide whether to load translator，检测是否为中文提示词，否则不适用翻译。
        prompt_modify = prompt_replace(prompt)
        bool_is_chinese = AnyText_Pipeline.is_chinese(self, prompt_modify)
        
        device = get_device_by_name(device)
        loader_out = AnyText_Loader.split("|")
        
        if bool_is_chinese == False:
            use_translator = False
        else:
            use_translator = True
            if 'damo/nlp_csanmt_translation_zh2en' in loader_out[3]:
                if not os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "nlp_csanmt_translation_zh2en", "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
                    if not is_module_imported('snapshot_download'):
                        from modelscope.hub.snapshot_download import snapshot_download
                    snapshot_download('damo/nlp_csanmt_translation_zh2en')
            else:
                if not os.access(os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_small_1024", "model.safetensors"), os.F_OK):
                    if not is_module_imported('hg_snapshot_download'):
                        from huggingface_hub import snapshot_download as hg_snapshot_download
                    hg_snapshot_download(repo_id="utrobinmv/t5_translate_en_ru_zh_small_1024")
        
        pipe = AnyText_Pipeline(ckpt_path=loader_out[1], clip_path=loader_out[2], translator_path=loader_out[3], cfg_path=loader_out[4], use_translator=use_translator, device=device, use_fp16=fp16, all_to_device=all_to_device, loaded_model_tensor=self.model)
        
        # tensor图片转换为numpy图片
        pos_image = comfy_tensor_Image2np_Image(self, pos_image)
        ori_image = comfy_tensor_Image2np_Image(self, ori_image)
        # 保存转换后的numpy图片到ComfyUI临时文件夹
        pos_image.save(tmp_pose_img_path)
        ori_image.save(tmp_ori_img_path)
        
        ori = tmp_ori_img_path
        pos = tmp_pose_img_path
        
        if mode == "text-generation":
            ori_image = None
            revise_pos = revise_pos
        else:
            revise_pos = False
            ori_image = ori
            
        n_lines = count_lines(prompt)
        if Random_Gen == True:
            generate_rectangles(nonEdit_random_gen_width, nonEdit_random_gen_height, n_lines, max_trys=500)
            pos_img = Random_Gen_Mask_path
        else:
            pos_img = pos
            
        # lora_path = r"D:\AI\ComfyUI_windows_portable\ComfyUI\models\loras\ys艺术\sd15_mw_bpch_扁平风格插画v1d1.safetensors"
        # lora_ratio = 1
        # lora_path_ratio = str(lora_path)+ " " + str(lora_ratio)
        # print("\033[93m", lora_path_ratio, "\033[0m")
        
        params = {
            "mode": mode,
            "use_fp16": fp16,
            "Random_Gen": Random_Gen,
            "sort_priority": sort_radio,
            "revise_pos": revise_pos,
            # "show_debug": show_debug,
            "image_count": img_count,
            "ddim_steps": ddim_steps - 1,
            "image_width": nonEdit_random_gen_width,
            "image_height": nonEdit_random_gen_height,
            "strength": strength,
            "cfg_scale": cfg_scale,
            "eta": eta,
            "a_prompt": a_prompt,
            "n_prompt": n_prompt,
            # "lora_path_ratio": lora_path_ratio,
            }
        input_data = {
                "prompt": prompt,
                "seed": seed,
                "draw_pos": pos_img,
                "ori_image": ori_image,
                }
        # if show_debug ==True:
        #     print(f'\033[93mloader from .util(从.util输入的loader): {AnyText_Loader}, \033[0m\n \
        #             \033[93mloader_out split form loader(分割loader得到4个参数): {loader_out}, \033[0m\n \
        #             \033[93mFont(字体)--loader_out[0]: {loader_out[0]}, \033[0m\n \
        #             \033[93mAnyText Model(AnyText模型)--loader_out[1]: {loader_out[1]}, \033[0m\n \
        #             \033[93mclip model(clip模型)--loader_out[2]: {loader_out[2]}, \033[0m\n \
        #             \033[93mTranslator(翻译模型)--loader_out[3]: {loader_out[3]}, \033[0m\n \
        #             \033[93myaml_file(yaml配置文件): {loader_out[4]}, \033[0m\n) \
        #             \033[93mIs Chinese Input(是否中文输入): {use_translator}, \033[0m\n \
        #             \033[93mNumber of text-content to generate(需要生成的文本数量): {n_lines}, \033[0m\n \
        #             \033[93mpos_image location(遮罩图位置): {pos}, \033[0m\n \
        #             \033[93mori_image location(原图位置): {ori}, \033[0m\n \
        #             \033[93mSort Position(文本生成位置排序): {sort_radio}, \033[0m\n \
        #             \033[93mEnable revise_pos(启用位置修正): {revise_pos}, \033[0m')
        x_samples, results, rtn_code, rtn_warning, debug_info, self.model = pipe(input_data, font_path=loader_out[0], cpu_offload=cpu_offload, **params)
        if rtn_code < 0:
            raise Exception(f"Error in AnyText pipeline: {rtn_warning}")
        output = pil2tensor(x_samples)
        print("\n", debug_info)
        return(output)
        
# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyText": AnyText,
}
