import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=[
    'bert_base', 
    'vit', 
    't5_small', 
    't5_base', 
    'bert_large',
    'clip',
    'opt_125m',
    'bert_tiny_pesudo'
    ])

parser.add_argument('--pretrained_model', type=str, default=None)
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--seq_length', type=int, default=128)

args = parser.parse_args()

import tensorflow as tf
import shutil
from pathlib import Path
import os
import subprocess
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
tf.get_logger().setLevel('ERROR')

# ## Get Huggingface BERT model
from transformers import TFBertModel
from transformers import TFViTForImageClassification
from transformers import TFT5Model
from transformers import BertConfig
from transformers import TFSwinModel
from transformers import TFSwinForImageClassification
from transformers import SwinConfig
from transformers import TFCLIPModel
from transformers import TFOPTModel
from transformers import TFOPTForCausalLM
from transformers import TFT5ForConditionalGeneration

batch_size = None if args.batch_size == -1 else args.batch_size
seq_length = args.seq_length

class TFBert(TFBertModel):
    @tf.function(input_signature=[{
        "input_ids": tf.TensorSpec((batch_size, seq_length), tf.int32, name="input_ids"),
        "attention_mask": tf.TensorSpec((batch_size, seq_length), tf.int32, name="attention_mask"),
        "token_type_ids": tf.TensorSpec((batch_size, seq_length), tf.int32, name="token_type_ids"),
    }])
    def serving(self, inputs):
        output = self.call(inputs)
        return self.serving_output(output).pooler_output

class TFOPT(TFOPTForCausalLM):
    @tf.function(input_signature=[{
        "attention_mask": tf.TensorSpec((batch_size, 4), tf.int32, name="attention_mask"),
        "input_ids": tf.TensorSpec((batch_size, 4), tf.int32, name="input_ids"),
    }])
    def serving(self, inputs):
        output = self.call(inputs['input_ids'], attention_mask=inputs['attention_mask'], use_cache=False, output_attentions=False,
            past_key_values=None,
            output_hidden_states=False, return_dict=True)
        return tf.math.argmax(output.logits[:, -1], axis=-1)

class TFCLIP(TFCLIPModel):
    @tf.function(input_signature=[{
        "input_ids": tf.TensorSpec((batch_size, seq_length), tf.int32, name="input_ids"),
        "attention_mask": tf.TensorSpec((batch_size, seq_length), tf.int32, name="attention_mask"),
        "pixel_values": tf.TensorSpec((batch_size, 3, 224, 224), tf.float32, name="pixel_values"),
    }])
    def serving(self, inputs):
        output = self.call(inputs)
        return self.serving_output(output).logits_per_image

class TFVit(TFViTForImageClassification):
    @tf.function(input_signature=[{
        "pixel_values": tf.TensorSpec((batch_size, 3, 224, 224), tf.float32, name="pixel_values"),
    }])
    def serving(self, inputs):
        output = self.call(inputs)
        return self.serving_output(output)

class TFT5(TFT5Model):
    @tf.function(input_signature=[{
        "input_ids": tf.TensorSpec((batch_size, 128), tf.int32, name="input_ids"),
        "decoder_input_ids": tf.TensorSpec((batch_size, 128), tf.int32, name="decoder_input_ids"),
    }])
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output).last_hidden_state



def get_model():
    if args.model == 'bert_base':
        return TFBert.from_pretrained(args.pretrained_model)
    if args.model == 'bert_large':
        return TFBert.from_pretrained(args.pretrained_model)
    if args.model == 'vit':
        return TFVit.from_pretrained('google/vit-base-patch16-224')
    if args.model == 'clip':
        return TFCLIP.from_pretrained('openai/clip-vit-base-patch32')
    if args.model == 'opt_125m':
        return TFOPT.from_pretrained('facebook/opt-125m')
    if args.model == 't5_small':
        return TFT5.from_pretrained('t5-small')
    if args.model == 't5_base':
        return TFT5.from_pretrained('t5-base')
    if args.model == "bert_tiny_pesudo":
        model = TFBert.from_config(BertConfig(vocab_size=250368,
                                             hidden_size=384,
                                             num_hidden_layers=6,
                                             num_attention_heads=12,
                                             intermediate_size=384*4,
                                             max_position_embeddings=256))
        # input_ids = np.full((1, 256), 7592, dtype=np.int32)
        # input_ids[:, 0] = 101
        # input_ids[:, -1] = 102
        # attention_mask = np.ones((1, 256), dtype=np.int32)
        # token_type_ids = np.zeros((1, 256), dtype=np.int32)
        # inputs = np.concatenate([input_ids, attention_mask, token_type_ids], axis=0)
        # inputs = tf.convert_to_tensor(inputs)
        # print(model.dummy_inputs)
        output = model(model.dummy_inputs)
        print(output)
        return model

    assert False, f'Unsupported model {args.model}'

model = get_model()

weights = model.trainable_variables
for weight in weights:
    print(weight.name, weight.shape, weight.dtype)

model.save_pretrained(args.model_dir, saved_model=True)

source_dir = os.path.join(args.model_dir, 'saved_model', '1', '*')
dst_dir = args.model_dir

subprocess.call(f'mv {source_dir} {dst_dir}', shell=True)
subprocess.call(f'rm -r {args.model}/saved_model', shell=True)
subprocess.call(f'rm {args.model}/tf_model.h5', shell=True)
