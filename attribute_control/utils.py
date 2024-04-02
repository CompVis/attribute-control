import re
from pydoc import locate
from itertools import product

from typing import Callable, Dict, List, Union, Tuple, Any, Optional
import torch
from jaxtyping import Float
from omegaconf import OmegaConf


OmegaConf.register_new_resolver("locate", lambda name: locate(name))
OmegaConf.register_new_resolver("combinations", lambda combination_string: get_string_combinations(combination_string))
def get_string_combinations(combination_string: str) -> List[str]:
    """Takes a string such as "{,a photo of,a portrait of} a {man,woman}" and transforms it into a list containing all combinations of the comma-separated substrings in braces ("a man", "a woman", "a photo of a man", ...).

    Args:
        combination_string (str): Combination pattern.

    Returns:
        List[str]: List containing all combinations.
    """
    parts = []
    i_last = 0
    for m in re.finditer(r'\{([^\}\{]*)\}', combination_string):
        m_start, m_end = m.span()
        if i_last < m_start:
            parts.append([combination_string[i_last:m_start]])
        parts.append(m.group(1).split(','))
        i_last = m_end
    if i_last < len(combination_string):
        parts.append([combination_string[i_last:]])
    return [''.join(p) for p in product(*parts)]


def reduce_tensors_recursively(*values: Union[torch.Tensor, Tuple, List, Dict], reduction_op: Callable[[Tuple[torch.Tensor]], torch.Tensor]) -> Union[torch.Tensor, Tuple, List, Dict]:
    if len(values) == 0:
        return None
    else:
        if isinstance(values[0], torch.Tensor):
            return reduction_op(values)
        elif isinstance(values[0], tuple):
            return tuple(reduce_tensors_recursively(*vs, reduction_op=reduction_op) for vs in zip(*values, strict=True))
        elif isinstance(values[0], list):
            return [reduce_tensors_recursively(*vs, reduction_op=reduction_op) for vs in zip(*values, strict=True)]
        elif isinstance(values[0], dict):
            return { k: reduce_tensors_recursively([v[k] for v in values], reduction_op=reduction_op) for k in values[0] }
        else:
            return values


def getattr_recursive(obj: Any, path: str) -> Any:
    parts = path.split('.')
    for part in parts:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def broadcast_trailing_dims(tensor: Float[torch.Tensor, '(c)'], reference: Float[torch.Tensor, '(c) ...']) -> torch.Tensor:
    num_trailing = len(reference.shape) - len(tensor.shape)
    for _ in range(num_trailing):
        tensor = tensor.unsqueeze(-1)
    return tensor
