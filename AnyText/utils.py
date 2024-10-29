import os
import folder_paths
import torch
import numpy as np
import time
from PIL import Image

current_directory = os.path.dirname(os.path.abspath(__file__))
comfyui_models_dir = folder_paths.models_dir
comfy_temp_dir = folder_paths.get_temp_directory()
temp_txt_path = os.path.join(comfy_temp_dir, "AnyText_temp.txt")

class AnyText_loader:
    @classmethod
    def INPUT_TYPES(cls):
        font_list = os.listdir(os.path.join(comfyui_models_dir, "fonts"))
        checkpoints_list = folder_paths.get_filename_list("checkpoints")
        clip_list = os.listdir(os.path.join(comfyui_models_dir, "clip"))
        font_list.insert(0, "Auto_DownLoad")
        checkpoints_list.insert(0, "Auto_DownLoad")
        clip_list.insert(0, "Auto_DownLoad")

        return {
            "required": {
                "font": (font_list, ),
                "ckpt_name": (checkpoints_list, ),
                "clip": (clip_list, ),
                "translator": (["utrobinmv/t5_translate_en_ru_zh_small_1024", "damo/nlp_csanmt_translation_zh2en"],{"default": "utrobinmv/t5_translate_en_ru_zh_small_1024"}), 
                # "show_debug": ("BOOLEAN", {"default": False}),
                }
            }

    RETURN_TYPES = ("AnyText_Loader", )
    RETURN_NAMES = ("AnyText_Loader", )
    FUNCTION = "AnyText_loader_fn"
    CATEGORY = "ExtraModels/AnyText"
    TITLE = "AnyText Loader"

    def AnyText_loader_fn(self, 
                          font, 
                          ckpt_name, 
                          clip, 
                          translator, 
                        #   show_debug
                          ):
        font_path = os.path.join(comfyui_models_dir, "fonts", font)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_path = os.path.join(current_directory, 'models_yaml', 'anytext_sd15.yaml')
        if clip != 'Auto_DownLoad':
                clip_path = os.path.join(comfyui_models_dir, "clip", clip)
        else:
                clip_path = clip
        if translator != 'Auto_DownLoad':
                translator_path = os.path.join(comfyui_models_dir, "prompt_generator", translator)
        else:
                translator_path = translator
        
        #将输入参数合并到一个参数里面传递到.nodes
        loader = (font_path + "|" + str(ckpt_path) + "|" + clip_path + "|" + translator_path + "|" + cfg_path)
        
        # if show_debug == True:
        #     print(f'\033[93mloader(合并后的4个输入参数，传递给nodes): {loader} \033[0m\n \
        #             \033[93mfont_path(字体): {font_path} \033[0m\n \
        #             \033[93mckpt_path(AnyText模型): {ckpt_path} \033[0m\n \
        #             \033[93mclip_path(clip模型): {clip_path} \033[0m\n \
        #             \033[93mtranslator_path(翻译模型): {translator_path} \033[0m\n \
        #             \033[93myaml_file(yaml配置文件): {cfg_path} \033[0m\n')
        return (loader, )

class AnyText_translator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  (["utrobinmv/t5_translate_en_ru_zh_small_1024", "damo/nlp_csanmt_translation_zh2en"],{"default": "utrobinmv/t5_translate_en_ru_zh_small_1024"}), 
                "prompt": ("STRING", {"default": "这里是单批次翻译文本输入。\n声明补充说，沃伦的同事都深感震惊，并且希望他能够投案自首。\n尽量输入单句文本，如果是多句长文本建议人工分句，否则可能出现漏译或未译等情况！！！\n使用换行，效果可能更佳。", "multiline": True}),
                "Batch_prompt": ("STRING", {"default": "这里是多批次翻译文本输入，使用换行进行分割。\n天上掉馅饼啦，快去看超人！！！\n飞流直下三千尺，疑似银河落九天。\n启用Batch_Newline表示输出的翻译会按换行输入进行二次换行,否则是用空格合并起来的整篇文本。", "multiline": True}),
                "t5_Target_Language": (["en", "zh", "ru", ],{"default": "en"}), 
                "if_Batch": ("BOOLEAN", {"default": False}),
                "Batch_Newline" :("BOOLEAN", {"default": True}),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"],{"default": "auto"}), 
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    CATEGORY = "ExtraModels/AnyText"
    FUNCTION = "AnyText_translator"
    TITLE = "AnyText Translator"

    def AnyText_translator(self, prompt, model, Batch_prompt, if_Batch, device, Batch_Newline, t5_Target_Language):
        device = get_device_by_name(device)
        # 使用换行(\n)作为分隔符
        Batch_prompt = Batch_prompt.split("\n")  
        input_sequence = prompt
        if model == 'damo/nlp_csanmt_translation_zh2en':
            sttime = time.time()
            if if_Batch == True:
                input_sequence = Batch_prompt
                # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
                input_sequence = '<SENT_SPLIT>'.join(input_sequence)
            if os.access(os.path.join(comfyui_models_dir, "prompt_generator", "nlp_csanmt_translation_zh2en", "tf_ckpts", "ckpt-0.data-00000-of-00001"), os.F_OK):
                zh2en_path = os.path.join(comfyui_models_dir, 'prompt_generator', 'nlp_csanmt_translation_zh2en')
            else:
                zh2en_path = "damo/nlp_csanmt_translation_zh2en"
            
            if not is_module_imported('pipeline'):
                from modelscope.pipelines import pipeline
            if not is_module_imported('Tasks'):
                from modelscope.utils.constant import Tasks
            if device == 'cuda':
                pipeline_ins = pipeline(task=Tasks.translation, model=zh2en_path, device='gpu')
            outputs = pipeline_ins(input=input_sequence)
            if if_Batch == True:
                results = outputs['translation'].split('<SENT_SPLIT>')
                if Batch_Newline == True:
                    results = '\n\n'.join(results)
                else:
                    results = ' '.join(results)
            else:
                results = outputs['translation']
            endtime = time.time()
            print("\033[93mTime for translating(翻译耗时): ", endtime - sttime, "\033[0m")
            del pipeline_ins
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            if if_Batch == True:
                input_sequence = Batch_prompt
                # 用特定的连接符<SENT_SPLIT>，将多个句子进行串联
                input_sequence = '|'.join(input_sequence)
            self.zh2en_path = os.path.join(folder_paths.models_dir, "prompt_generator", "models--utrobinmv--t5_translate_en_ru_zh_small_1024")
            if not os.access(os.path.join(self.zh2en_path, "model.safetensors"), os.F_OK):
                self.zh2en_path = "utrobinmv/t5_translate_en_ru_zh_small_1024"
            outputs = t5_translate_en_ru_zh(t5_Target_Language, input_sequence, self.zh2en_path, device)[0]
            if if_Batch == True:
                results = outputs.split('| ')
                if Batch_Newline == True:
                    results = '\n\n'.join(results)
                else:
                    results = ' '.join(results)
            else:
                results = outputs
        
        with open(temp_txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(results)
        return (results, )

def is_module_imported(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def is_folder_exist(folder_path):
    result = os.path.exists(folder_path)
    return result

def get_device_by_name(device):
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            elif torch.xpu.is_available():
                device = "xpu"
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device

# Node class and display name mappings
NODE_CLASS_MAPPINGS = {
    "AnyText_loader": AnyText_loader,
    "AnyText_translator": AnyText_translator,
}

def t5_translate_en_ru_zh(Target_Language, prompt, model_path, device):
    
    # prefix = 'translate to en: '
    sttime = time.time()
    if not is_module_imported('T5ForConditionalGeneration'):
        from transformers import T5ForConditionalGeneration
    if not is_module_imported('T5Tokenizer'):
        from transformers import T5Tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path,)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    if Target_Language == 'zh':
        prefix = 'translate to zh: '
    elif Target_Language == 'en':
        prefix = 'translate to en: '
    else:
        prefix = 'translate to ru: '
    src_text = prefix + prompt
    input_ids = tokenizer(src_text, return_tensors="pt")
    generated_tokens = model.generate(**input_ids).to(device, torch.float32)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    model.to('cpu')
    endtime = time.time()
    print("\033[93mTime for translating(翻译耗时): ", endtime - sttime, "\033[0m")
    return result

def comfy_tensor_Image2np_Image(self,comfy_tensor_image):
    comfyimage = comfy_tensor_image.numpy()[0] * 255
    image_np = comfyimage.astype(np.uint8)
    image = Image.fromarray(image_np)
    return image