import torch
import torch.nn as nn
from mistral.model import Transformer, SimpleInputMetadata, ModelArgs

import json

class MEEstral(Transformer):
    def __init__(self, args, pipeline_rank = 0, num_pipeline_ranks = 1, path_weights = None, max_batch_size = 1, device = "cuda", dtype = torch.float16):

        if path_weights is not None:
            with open(path_weights + "/params.json", "r") as f:
                args = ModelArgs.from_dict(json.load(f))

        self.args = args
        self.args.max_batch_size = max_batch_size

        super().__init__(self.args, pipeline_rank, num_pipeline_ranks)
        
        if path_weights is not None:
        
            # will I really use this?
            if num_pipeline_ranks > 1:
                pipeline_rank = torch.distributed.get_rank()
            else:
                pipeline_rank = 0

            loaded = torch.load(str(path_weights + "/consolidated.00.pth"), mmap=True)
            self.load_state_dict(loaded, assign=True)        
        
        self.to(device)

        # should be added later
        self.EEs = [16, 24]        
        self.exits = nn.ModuleList([flag_exit(self.args) for _ in range(len(self.EEs))])
        

    def forward_partial(self,
                input_ids,
                seqlens,
                cache = None):
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
        else:
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        freqs_cis = self.freqs_cis[input_metadata.positions]

        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            return self.norm(h)
        

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens,
        cache = None,
    ) -> torch.Tensor:
        h = self.forward_partial(input_ids, seqlens, cache=cache)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            outs = self.output(h)
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        return outs.float()
        


class flag_exit(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.activation = nn.Sigmoid()

        self.flag = nn.Linear(args.dim, 1, bias=True)


    def forward(self, x):
        
        x = self.flag(x)
        x = self.activation(x)
        
        return x