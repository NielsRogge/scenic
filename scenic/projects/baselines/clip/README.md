## CLIP
This directory contains the implementation of CLIP for [learning visual models from natural language supervision](https://arxiv.org/abs/2103.00020).
The code here uses JAX and Flax and follows the [official implementation of CLIP](https://github.com/openai/CLIP).

Note that the current implementation does not yet support training CLIP from
scratch.

This implementation uses the [tokenizer from the original implementation](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py)
and requires downloading and setting the path to the `bpe_simple_vocab_16e6.txt.gz`
in `layer.py`. Besides, path to the local CLIP checkpoints that are downloaded
from the original repo is needed to be set in `model.py`.

## Example usage:
```
from scenic.projects.baselines.clip import model as clip
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer

model_name = 'resnet_50'

model = clip.MODELS[model_name]()
clip_vars = clip.load_model_vars(model_name)
model_bound = model.bind(clip_vars)

tokenizer = clip_tokenizer.build_tokenizer()
text = tokenizer('This is a cat.')
image = jnp.zeros((1, 224, 224, 3))
image = clip.normalize_image(image)

encoded_image, encoded_text = model_bound(image, text)

# Or individually:
encoded_text = model_bound.encode_text(text)
encoded_image = model_bound.encode_image(image)
```

To be loadable, new checkpoints have to be converted from torch:
```
import torch
import numpy as np
import jax

clip = torch.load('/path/to/clip.pt')
params = jax.tree_map(lambda p: p.cpu().numpy(), clip.state_dict())
with open('/path/to/clip.npy', 'wb') as f:
  np.save(f, params)
```

Note that these models run natively on images with resolution 224 and normalized
using `IMAGE_MEAN` and `IMAGE_STD`. The maximum text length is 77.

### Acknowledgment
We would like to thank Ben Poole and Dirk Weissenborn for their contribution to
the CLIP implementation in Scenic.
