import torch

import sys
sys.path.append('./SALSA-CLRS/')
from baselines.core.models.decoder import output_mask
from baselines.core.utils import stack_hidden   

class EncodeProcessDecode(torch.nn.Module):

    hint_loss_weight = 1.0
    decoder_use_last_hidden = False

    def __init__(self,
                 encoders,
                 decoders,
                 processor,
                 device='cpu'):
        
        super().__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.processor = processor
        self.device = torch.device(device)

    def _stack_hints(self, hints):
        return {k: torch.stack([hint[k] for hint in hints], dim=-1) for k in hints[0]} if hints else {}
    
    def forward(self, data):
        data.to(self.device)

        output = None
        max_len = data.length.max().item()
        hints = []
        x_in = self.encoders[data.task](data)[0]
        
        x = x_in
        for step in range(max_len):
            x_last = x
            x_stacked = stack_hidden(x_in, x, x_last, True)

            x = self.processor(x_stacked,
                               data.edge_index,
                               edge_attr = data.edge_attr,
                               batch=data.batch)
            x_stacked = stack_hidden(x_in, x, x_last, self.decoder_use_last_hidden)
            if self.training and self.hint_loss_weight > 0.0:
                hints.append(self.decoders[data.task](x_stacked, data, 'hints'))
            output_step = self.decoders[data.task](x_stacked, data, 'outputs')
            mask = output_mask(data, step)
            if output is None:
                output = {k: output_step[k]*mask[k] for k in output_step}
            else:
                for k in output_step:
                    output[k][mask[k]] = output_step[k][mask[k]]

        hints = self._stack_hints(hints)

        return output, hints, x