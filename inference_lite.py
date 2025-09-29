# !/usr/bin/env python
# -*- coding: UTF-8 -*-

# https://github.com/RVC-Boss/GPT-SoVITS
import traceback
import LangSegment, os, re, sys, json
import torch
import numpy as np
import librosa
from .GPT_SoVITS.module.models import SynthesizerTrn
from .GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from .GPT_SoVITS.text import cleaned_text_to_sequence
from .GPT_SoVITS.text.cleaner import clean_text
from time import time as ttime
from .GPT_SoVITS.module.mel_processing import spectrogram_torch
from .tools.my_utils import load_audio
from .tools.i18n.i18n import I18nAuto, scan_language_list
from .GPT_SoVITS.text import chinese
current_u_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

import os.path
import sys
sys.path.append(f'{current_u_path}')

current_u_json=os.path.join(current_u_path,"weight.json")

if os.path.exists(current_u_json):
    pass
else:
    with open(current_u_json, 'w', encoding="utf-8") as file:json.dump({'GPT':{},'SoVITS':{}},file)

language_=os.environ.get("language","Auto")
language_=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language_
i18n=I18nAuto(language=language_)

punctuation = set(['!', '?', '…', ',', '.', '-'," "])

def lazy_change(text):
    if text=="chinese":
        output="中文"
    elif text=="english":
        output="英文"
    elif text=="japenese":
        output="日文"
    elif text=="cn_en":
        output="中英混合"
    elif text == "en_jp":
        output = "日英混合"
    elif text == "mix_languages":
        output = "多语种混合"
    elif text == "v2_cantonese":
        output = "粤语"
    elif text == "v2_korea":
        output = "韩文"
    elif text == "v2_cantonese_en":
        output = "粤英混合"
    elif text == "v2_en_ko":
        output = "韩英混合"
    elif text =="v2_mix_cantonese":
        output = "多语种混合(粤语)"
    elif text =="none":
        output = "不切"
    elif text =="fifty_words":
        output = "凑50字一切"
    elif text =="four_sentences":
        output = "凑四句一切"
    elif text =="chinese_period":
        output = "按中文句号。切"
    elif text =="english_period":
        output = "按英文句号.切"
    elif text =="four_punctuation":
        output = "按标点符号切"
    else:
        raise "Unsupport text"
   
    return output



def get_bert_feature(text, word2ph,tokenizer,bert_model):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")



def change_sovits_weights(sovits_path,version,is_half,dict_language_v1,dict_language_v2,prompt_language=None,text_language=None):
    #global vq_model, hps, version, dict_language

    import utils
    from .ulits import HParams
    original_hparams = getattr(utils, 'HParams', None)
    utils.HParams = HParams
    try:
        dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
    finally:
        if original_hparams is not None:
            utils.HParams = original_hparams
        elif hasattr(utils, 'HParams'):
            delattr(utils, 'HParams')

    #dict_s2 = torch.load(sovits_path, map_location="cpu",weights_only=False)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    #version = hps.model.version
    # print("sovits版本:",hps.model.version)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False)) #千万不能注释掉
    dict_language = dict_language_v1 if version =='v1' else dict_language_v2
    with open(current_u_json)as f:
        data=f.read()
        data=json.loads(data)
        data["SoVITS"][version]=sovits_path
    with open(current_u_json,"w")as f:f.write(json.dumps(data))
    # if prompt_language is not None and text_language is not None:
    if prompt_language in list(dict_language.keys()):
        prompt_text_update, prompt_language_update = {'__type__':'update'},  {'__type__':'update', 'value':prompt_language}
    else:
        prompt_text_update = {'__type__':'update', 'value':''}
        prompt_language_update = {'__type__':'update', 'value':i18n("中文")}
    if text_language in list(dict_language.keys()):
        text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
    else:
        text_update = {'__type__':'update', 'value':''}
        text_language_update = {'__type__':'update', 'value':i18n("中文")}
    return vq_model,hps, {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update,dict_language



#change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path,version,is_half):
    hz = 50
    import utils
    from .ulits import HParams
    original_hparams = getattr(utils, 'HParams', None)
    utils.HParams = HParams
    try:
        dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    finally:
        # 清理临时添加的 HParams
        if original_hparams is not None:
            utils.HParams = original_hparams
        elif hasattr(utils, 'HParams'):
            delattr(utils, 'HParams')
    #dict_s1 = torch.load(gpt_path, map_location="cpu",weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open(current_u_json)as f:
        data=f.read()
        data=json.loads(data)
        data["GPT"][version]=gpt_path
    with open(current_u_json,"w")as f:f.write(json.dumps(data))
    
    return hz,max_sec,t2s_model,config


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
   # audio, sr = filename["waveform"].squeeze(0).squeeze(0), filename["sample_rate"]  # e([1, 236928])
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

#dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language,tokenizer,bert_model,is_half):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph,tokenizer,bert_model).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text,language,version,tokenizer,bert_model,is_half,final=False):
    dtype = torch.float16 if is_half == True else torch.float32
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version,tokenizer,bert_model,is_half)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph,tokenizer,bert_model).to(device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version,tokenizer,bert_model,is_half)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])

        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang,tokenizer,bert_model,is_half)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,tokenizer,bert_model,is_half,final=True)
    
    return phones,bert.to(dtype),norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

##ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
cache= {}
def get_tts_wav(ssl_model,vq_model,t2s_model,ref_wav_path, prompt_text, prompt_language, text, text_language,version,tokenizer,bert_model,is_half,dtype,dict_language,hz,max_sec,hps,
                how_to_cut, top_k=20, top_p=0.6, temperature=0.6, ref_free
    =False,speed=1,if_freeze=False,inp_refs=None):
    global cache
    if ref_wav_path:pass
    else:raise "need audio"
    if text:pass
    else:raise "need texts"
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    
    prompt_language = dict_language[prompt_language["value"]]
    text_language = dict_language[text_language["value"]]
    
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
    
    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    if not ref_free:
        with torch.no_grad():
            #wav16k, sr=ref_wav_path["waveform"].squeeze(0).squeeze(0),ref_wav_path["sample_rate"] #e([ 236928])
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                print("参考音频在3~10秒范围外，请更换！")
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            
            if is_half:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1-t0)
    
    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts,i18n)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language, version,tokenizer,bert_model,is_half)

    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        #print(i18n("实际输入的目标文本(每句):"), text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version,tokenizer,bert_model,is_half)
        #print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text]=pred_semantic
        t3 = ttime()
        refers=[]
        if(inp_refs):
            for path in inp_refs:
                try:
                    refer = get_spepc(hps, path.name).to(dtype).to(device)
                    refers.append(refer)
                except:
                    traceback.print_exc()
        if(len(refers)==0):refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
        audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        t.extend([t2 - t1,t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % 
           (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
           )
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return  "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def process_text(texts,i18n):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

