from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
import clip

model_name = "../distilbert-base-turkish-cased-clip"
base_model = TFAutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
head_model = tf.keras.models.load_model("./clip_head.h5")

def encode_text(base_model, tokenizer, head_model, texts):
    tokens = tokenizer(texts, padding=True, return_tensors='tf')
    embs = base_model(**tokens)[0]

    attention_masks = tf.cast(tokens['attention_mask'], tf.float32)
    sample_length = tf.reduce_sum(attention_masks, axis=-1, keepdims=True)
    masked_embs = embs * tf.expand_dims(attention_masks, axis=-1)
    base_embs = tf.reduce_sum(masked_embs, axis=1) / tf.cast(sample_length, tf.float32)
    clip_embs = head_model(base_embs)
    clip_embs /= tf.norm(clip_embs, axis=-1, keepdims=True)
    return clip_embs




demo_images = {
    "bilgisayarda çalışan bir insan": "myspc.jpeg",
    "sahilde bir insan ve bir heykel": "mysdk.jpeg"
    }

clip_model, preprocess = clip.load("ViT-B/32")
images = {key: Image.open(f"images/{value}") for key, value in demo_images.items()}
img_inputs = torch.stack([preprocess(image).to('cpu') for image in images.values()])

with torch.no_grad():
    image_embs = clip_model.encode_image(img_inputs).float().to('cpu')

image_embs /= image_embs.norm(dim=-1, keepdim=True)
image_embs = image_embs.detach().numpy()
text_embs = encode_text(base_model, tokenizer, head_model, list(images.keys())).numpy()
similarities = image_embs @ text_embs.T
logits = tf.nn.softmax(tf.convert_to_tensor(similarities)).numpy()
idxs = np.argmax(logits, axis=-1).tolist()
for i, (key, value) in enumerate(demo_images.items()):
    print("path: ", value, "true label: ", key, "prediction: ", list(demo_images.keys())[idxs[i]], "score: ", logits[i, idxs[i]])
