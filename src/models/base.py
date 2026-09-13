from typing import Any

import torch

class ImageCaptionBase(torch.nn.Module):
    """Image Caption model base class, required for custom 
    training and inference functions.

    :meth: 
        forward: required for training
        caption_image: required for inference
    """
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        *args
    ):
        super(ImageCaptionBase, self).__init__()

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> Any:
        ...

    def caption_image(self, image, vocabulary, max_length=50, temperature=0.5) -> Any:
        ...
