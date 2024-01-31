from typing import Iterable

from lm_eval.base import BaseLM

class TorchLM(BaseLM):
    def __init__(self,
                 checkpoint="",
                 
                 ):
        super().__init__()

    @property
    def eot_token_id(self):
        pass

    @property
    def max_length(self):
        pass

    @property
    def max_gen_toks(self):
        pass

    @property
    def batch_size(self):
        pass

    @property
    def device(self):
        pass
    
    def tok_encode(self, string: str):
        pass

    def tok_decode(self, tokens: Iterable[int]):
        pass

    def _model_generate(self, context, max_length, eos_token_id):
        pass

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        pass
    