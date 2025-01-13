# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import io
import os
import random
import numpy as np
import torch
import torchaudio
import gc
import folder_paths
import platform
import subprocess
import soundfile as sf
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from .inference_lite import change_gpt_weights, change_sovits_weights, get_tts_wav,i18n,lazy_change
from .GPT_SoVITS.feature_extractor import cnhubert

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

# make weights dir 建立模型基准目录
weigths_GPT_SoVITS_current_path = os.path.join(folder_paths.models_dir, "GPT_SoVITS")
if not os.path.exists(weigths_GPT_SoVITS_current_path):
    os.makedirs(weigths_GPT_SoVITS_current_path)

folder_paths.add_model_folder_path("GPT_SoVITS", weigths_GPT_SoVITS_current_path)

weigths_GPT_current_path = os.path.join(weigths_GPT_SoVITS_current_path, "GPT_weights")
if not os.path.exists(weigths_GPT_current_path):
    os.makedirs(weigths_GPT_current_path)

weigths_SoVITS_current_path = os.path.join(weigths_GPT_SoVITS_current_path, "SoVITS_weights")
if not os.path.exists(weigths_SoVITS_current_path):
    os.makedirs(weigths_SoVITS_current_path)

bert_path=os.path.join(weigths_GPT_SoVITS_current_path,"chinese-roberta-wwm-ext-large")
if not os.path.exists(bert_path):
     os.makedirs(bert_path)
     
cnhubert_base_path=os.path.join(weigths_GPT_SoVITS_current_path,"chinese-hubert-base")
if not os.path.exists(cnhubert_base_path):
    os.makedirs(cnhubert_base_path)

dict_language_v1 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("粤语"): "all_yue",#全部按中文识别
    i18n("韩文"): "all_ko",#全部按韩文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("粤英混合"): "yue",#按粤英混合识别####不变
    i18n("韩英混合"): "ko",#按韩英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
}

# ffmpeg
ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/python_embeded/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"
    

# *****************mian***************

class GPT_SoVITS_LoadModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        sovits_weigths_list = [i for i in folder_paths.get_filename_list("GPT_SoVITS") if "sovits_weights" in i.lower()]
        gpt_weigths_list = [i for i in folder_paths.get_filename_list("GPT_SoVITS") if "gpt_weights" in i.lower()]
        return {
            "required": {
                "sovits_weigths": (["none"] + sovits_weigths_list,),
                "gpt_weigths": (["none"] + gpt_weigths_list,),
                "refer_languages": (
                ["chinese", "english", "japenese", "cn_en", "en_jp", "mix_languages", "v2_cantonese", "v2_korea",
                 "v2_cantonese_en", "v2_en_ko", "v2_mix_cantonese"],),
                "infer_languages": (
                ["chinese", "english", "japenese", "cn_en", "en_jp", "mix_languages", "v2_cantonese", "v2_korea",
                 "v2_cantonese_en", "v2_en_ko", "v2_mix_cantonese"],),
                "version": (["V2", "V1", ],),
                "is_half": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("MODEL_GPTSOVITS", )
    RETURN_NAMES = ("model", )
    FUNCTION = "main_loader"
    CATEGORY = "GPT_SoVITS_Lite"
    
    def main_loader(self, sovits_weigths, gpt_weigths, refer_languages,infer_languages, version,is_half):
        refer_languages=lazy_change(refer_languages)
        infer_languages=lazy_change(infer_languages)
        
        sovits_path=folder_paths.get_full_path("GPT_SoVITS",sovits_weigths)
        gpt_path = folder_paths.get_full_path("GPT_SoVITS", gpt_weigths)
        
        for i in ["config.json","tokenizer.json","pytorch_model.bin"]:
            if not os.path.isfile(os.path.join(bert_path,i)):
                hf_hub_download(
                    repo_id="lj1995/GPT-SoVITS",
                    subfolder="chinese-roberta-wwm-ext-large",
                    filename=i,
                    local_dir=weigths_GPT_SoVITS_current_path,
                )
            
        # init model
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half == True:
            bert_model = bert_model.half().to(device)
        else:
            bert_model = bert_model.to(device)
        
        
        for i in ["config.json", "preprocessor_config.json", "pytorch_model.bin"]:
            if not os.path.isfile(os.path.join(cnhubert_base_path, i)):
                hf_hub_download(
                    repo_id="lj1995/GPT-SoVITS",
                    subfolder="chinese-hubert-base",
                    filename=i,
                    local_dir=weigths_GPT_SoVITS_current_path,
                )
            
        cnhubert.cnhubert_base_path = cnhubert_base_path
        
        ssl_model = cnhubert.get_model()
        if is_half == True:
            ssl_model = ssl_model.half().to(device)
        else:
            ssl_model = ssl_model.to(device)
        
        gpt_hz,gpt_max_sec,gpt_t2s_model,gpt_config=change_gpt_weights(gpt_path=gpt_path,version=version,is_half=is_half)
        
        vq_model,hps,update_dict,update_choice_dict,prompt_text_update, prompt_language_update, text_update, text_language_update,dict_language\
            =change_sovits_weights(sovits_path=sovits_path,version=version,is_half=is_half,dict_language_v1=dict_language_v1,dict_language_v2=dict_language_v2,prompt_language=refer_languages,text_language=infer_languages)
        
        gc.collect()
        torch.cuda.empty_cache()
        model = {"gpt_hz": gpt_hz,"gpt_max_sec":gpt_max_sec,"gpt_t2s_model":gpt_t2s_model,"gpt_config":gpt_config,"vq_model":vq_model,"hps":hps,"update_dict":update_dict,
                 "dict_language":dict_language,"bert_model":bert_model,"tokenizer":tokenizer,"ssl_model":ssl_model,"is_half":is_half,
                 "update_choice_dict":update_choice_dict,"prompt_text_update":prompt_text_update,"prompt_language_update":prompt_language_update,"text_update":text_update,"text_language_update":text_language_update,"version":version}
        return (model,)


class GPT_SoVITS_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_GPTSOVITS",),
                "refer_audio": ("AUDIO",),
                "refer_texts": ("STRING", {"multiline": True, "default": "调制解酒药的方法，也很简单，只要把材料均匀混合在一起，就可以啦。"}),
                "split_infer_text":(["four_sentences", "none", "fifty_words", "chinese_period", "english_period", "four_punctuation"],),
                "infer_text": ("STRING", {"multiline": True, "default": "你今天吃的是什么，我吃的是红烧肉。"}),
                "top_k": ("INT", {"default": 5, "min": 1, "max":100}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 11.0, "step": 0.01, "round": 0.001}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "save_wav":("BOOLEAN", {"default": True},),
            }
        }
    
    RETURN_TYPES = ("AUDIO","STRING",)
    RETURN_NAMES = ("audio","file_path",)
    FUNCTION = "main"
    CATEGORY = "GPT_SoVITS_Lite"
    
    def main(self, model, refer_audio, refer_texts, split_infer_text, infer_text, top_k, top_p, temperature, speed,save_wav):
        #get items
        version= model.get("version")
        gpt_hz = model.get("gpt_hz")
        gpt_max_sec = model.get("gpt_max_sec")
        gpt_t2s_model = model.get("gpt_t2s_model")
        vq_model = model.get("vq_model")
        hps = model.get("hps")
        dict_language = model.get("dict_language")
        prompt_language_update = model.get("prompt_language_update")
        text_language_update = model.get("text_language_update")
        tokenizer=model.get("tokenizer")
        bert_model=model.get("bert_model")
        ssl_model=model.get("ssl_model")
        is_half=model.get("is_half")
        
        #pre data
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_file = os.path.join(folder_paths.get_temp_directory(), f"audio_refer_temp{audio_file_prefix}.wav")
        buff = io.BytesIO() # 减少音频数据传递导致的不必要文件存储
        torchaudio.save(buff, refer_audio["waveform"].squeeze(0), refer_audio["sample_rate"],format="FLAC")
        with open(audio_file, 'wb') as f:
            f.write(buff.getbuffer())
        
        dtype = torch.float16 if is_half == True else torch.float32
        
        gen=get_tts_wav(ssl_model,vq_model,gpt_t2s_model,audio_file, refer_texts, prompt_language_update, infer_text, text_language_update,version,tokenizer,bert_model,is_half,dtype,dict_language,
                          gpt_hz, gpt_max_sec,hps, how_to_cut=i18n(lazy_change(split_infer_text)), top_k=top_k, top_p=top_p, temperature=temperature, ref_free=False,speed=speed,if_freeze=False,inp_refs=None)
        for value in gen:
            print( "audio length is :", value[1].shape[0]/value[0]," seconds")
        
        if save_wav:
            last_sampling_rate, last_audio_data = value[0],value[1]
            file_path = os.path.join(folder_paths.get_output_directory(), f"infer_{audio_file_prefix}.wav")
            sf.write(file_path, last_audio_data, last_sampling_rate)
            print(f"Audio saved to {file_path}")
        else:
            file_path=folder_paths.get_output_directory()
        waveform = torch.from_numpy(value[1]).unsqueeze(0)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": value[0]}
        gc.collect()
        torch.cuda.empty_cache()
        
        return (audio,file_path,)


NODE_CLASS_MAPPINGS = {
    "GPT_SoVITS_LoadModel": GPT_SoVITS_LoadModel,
    "GPT_SoVITS_Sampler": GPT_SoVITS_Sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT_SoVITS_LoadModel": "GPT_SoVITS_LoadModel",
    "GPT_SoVITS_Sampler": "GPT_SoVITS_Sampler",
}
