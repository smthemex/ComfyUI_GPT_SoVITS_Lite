
# ComfyUI_GPT_SoVITS_Lite 
[GPT_SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) infer only for ComfyUI users.纯推理版本，基于最新的GPT_SoVITS。

# 注意事项
* 本插件优先适配了中英文，日文，韩文等并未列入（可去掉requirements.txt里对应注释），如果你嫌安装麻烦,下载麻烦，请下载GPT_SoVITS官方的各种一键整合包。
* 本插件的测试环境，torch2.51 ，cuda124,python311

1.Installation  
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_GPT_SoVITS_Lite.git
```
2.requirements  
----
```
pip install -r requirements.txt
```
* 整合包安装requirements.txt的标准流程：1、复制插件的requirements.txt到你的python_embeded文件目录下；2右键桌面空白位置，打开CMD，3，输入以下命令：
```
python -m pip install -r requirements.txt
# 或者走清华的加速路线
python -m pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple   
```
* 特别需要做的步骤（这个是微调模型时带来的缺陷，在comfyUI目前只能注意操作，当然，安装特定的库似乎也可以，但是带来更多的不兼容，所以用这个办法解决）：
复制插件目录里的HParams.py文件，到ComfyUI\utils目录下，然后打开‘ComfyUI\utils\__init__.py’ 文件，一般是空的，加入如下两行代码： 
```
from .extra_config import *
from .HParams import *
```

3.checkpoints 
----
* 3.1 从[lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) 下载chinese-hubert-base和chinese-roberta-wwm-ext-large2个文件夹的所有文件，放在ComfyUI\models\GPT_SoVITS下
* 3.2 [这里](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip) 下载g2pW的汉字转拼音文件，解压后，放在示例的G2PWModel目录下
* 3.3 [这里](https://www.icloud.com/iclouddrive/079Bx3QbEosu8XIDkjim_ixPw#nltk_data) 下载nltk_data ，解压后，如果是整合包，放在python_embeded目录下
 ```
├── ComfyUI/models/GPT_SoVITS/
|     ├── chinese-hubert-base  #(内含1个模型2个config文件)
|          ├──config.json
|          ├──preprocessor_config.json
|          ├──pytorch_model.bin
|     ├── chinese-roberta-wwm-ext-large2  #(内含1个模型2个config文件)
|          ├──tokenizer.json
|          ├──config.json
|          ├──pytorch_model.bin
|     ├── GPT_weights 
|          ├──这个文件夹下放你用来推理的GPT_weights 模型 （别人训练的，自己去找吧）*.ckpt
|     ├── SoVITS_weights 
|          ├──这个文件夹下放你用来推理的SoVITS_weights 模型  （别人训练的，自己去找吧）*.pth
|     ├── G2PWModel #
|          ├──bopomofo_to_pinyin_wo_tune_dict.json
|          ├──char_bopomofo_dict.json
|          ├──config.py
|          ├──g2pW.onnx
|          ├──MONOPHONIC_CHARS.txt
|          ├──POLYPHONIC_CHARS.txt
├── 你的comfyUI便携包名/python_embeded
|     ├── nltk_data
|          ├── corpora （内部还有文件）
|          ├──taggers （内部还有文件）
  ```

