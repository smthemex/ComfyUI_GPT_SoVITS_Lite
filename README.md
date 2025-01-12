
# ComfyUI_GPT_SoVITS_Lite 
[GPT_SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) infer only for ComfyUI users.纯推理版本，基于最新的GPT_SoVITS。

# 注意事项
* 本插件优先适配了中英文，日文，韩文等并未列入（可去掉requirements.txt里对应注释），如果你嫌安装麻烦,下载麻烦，请下载GPT_SoVITS官方的各种一键整合包。
* 本插件的测试环境，win11 torch2.51 ，cuda124,python311 comfyUI便携包 / win11 torch2.2.0 cuda121,python311 comfyUI便携包

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
* 2.1 整合包安装requirements.txt的标准流程：1、复制插件的requirements.txt到你的python_embeded文件目录下；2右键桌面空白位置，打开CMD，3，输入以下命令：
  ```
  python -m pip install -r requirements.txt
  python -m pip install jieba-fast
  # 或者走清华的加速路线
  python -m pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
  python -m pip install jieba-fast -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
* 2.2 特别需要做的步骤 [jieba-fast](https://github.com/deepcs233/jieba_fast) windows可能不好安装，如果你安装不上，尤其是整合包，请按照以下方法来安装：
    直接从 [jieba-fast](https://github.com/deepcs233/jieba_fast)下载库文件的压缩包，解压后，复制jieba目录下的windows/python3文件夹里的2个文件夹（jieba_fast和jieba_fast-0.49.dist-info）将两个文件夹放入python_embeded\Lib\site-packages目录下，然后也复制插件目录下的_jieba_fast_functions_py3.cp311-win_amd64.pyd 文件到python_embeded\Lib\site-packages目录下。
  这里要注意，jieba官方的是py35的太旧了，我在插件里放的是311的，如果你是python310或者312，可能需要重新编译，编译方法如下：
  在解压的jieba-fast目录下，注意要有setup.py文件，打开CMD，运行
  ```
  python setup.py build_ext --inplace
  ```
* 2.3 特别需要做的步骤（这个是微调模型时带来的缺陷，在comfyUI目前只能按此操作，当然，安装特定的库似乎也可以，但是带来更多的不兼容，所以用这个简单办法解决）：
复制插件目录里的HParams.py文件，到ComfyUI\utils目录下，然后打开‘ComfyUI\utils\__init__.py’ 文件，一般是空的，加入如下两行代码： 
  ```
  from .extra_config import *
  from .HParams import *
  ```

3.checkpoints 
----
* 3.1 从[lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) 下载chinese-hubert-base和chinese-roberta-wwm-ext-large2个文件夹的所有文件，放在ComfyUI\models\GPT_SoVITS下
* 3.2 [这里](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip) 下载g2pW的汉字转拼音文件，解压后，放在示例的G2PWModel目录下
* 3.3 [这里](https://www.icloud.com/iclouddrive/079Bx3QbEosu8XIDkjim_ixPw#nltk_data) 下载nltk_data ，解压后，如果是整合包，放在python_embeded目录下，如果是安装包，放在插件所在盘的位置，比如在D盘就放D盘下，当然，放python目录也可以。
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

# 4 Example
![](https://github.com/smthemex/ComfyUI_GPT_SoVITS_Lite/blob/main/example.png)

# 5 Citation
Al code from [GPT_SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 


