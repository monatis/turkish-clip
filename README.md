## Acknowledgement
Google supported this work by providing Google Cloud credit. Thank you Google for supporting the open source! 🎉

## What is this?
This work enables to use [OpenAI CLIP](https://github.com/openai/CLIP)'s `ViT-B/32` image encoder with a text encoder in Turkish. It is composed of a base model and a clip head model. The base model is a finetuned version of [dbmdz/distilbert-base-turkish-cased](https://huggingface.co/dbmdz/distilbert-base-turkish-cased) and published at [HuggingFace's Models Hub](https://huggingface.co/mys/distilbert-base-turkish-cased-clip). It should be used with `clip_head.h5` from this repo.

## Installation
First, you need to install CLIP and its requirements according the prompts in [its repo](https://github.com/openai/CLIP). Then, clone this repo and all other requirements can be installed by using `requirements.txt`:
```shell
git clone https://github.com/monatis/turkish-clip.git
cd turkish-clip
pip install -r requirements.txt
```

## Usage
Once you clone the repo and install the requirements, you can run `inference.py` script for a quick inference demo:

```shell
python inference.py
```

This script loads the base model from HuggingFace's Models Hub and the clip head from this repo. It correctly classifies two sample images with a zero-shot technique.
 
## How it works
`encode_text()` function agregates per-token hidden states outputted by the Distilbert model to produce a single vector per sequence. Then, `clip_head.h5` model projects this vector onto the same vector space as CLIP's text encoder with a single dense layer. First, all the Distilbert layers were frozen an and the head dense layer was trained for a few epochs. Then, freezing was removed and the dense layer was trained with the Distilbert layers for a few more epochs. I created the dataset by machine-translating COCO captions into Turkish. During training, vector representations of English captions outputted by the original CLIP text encoder was used as target values, and MSE between these vectors and `clip_head.h5` outputs were minimized.

## Future work
The dataset and the training notebook will be released soon. I may also consider releasing bigger models finetuned with better datasets as well as more usage examples if the community finds this work useful. This model will also be added to my [ai-aas](https://github.com/monatis/ai-aas) project.