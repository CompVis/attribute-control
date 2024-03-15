import re
from typing import List


def get_mask_regex(prompt: str, pattern_target: str, verbose: bool = True) -> List[bool]:
    characterwise_mask = [False] * len(prompt)
    num_matches = 0
    for m in re.finditer(pattern_target, prompt):
        num_matches += 1
        characterwise_mask[m.span()[0]:m.span()[1]] = [True] * len(m.group(0))
    if verbose and (num_matches > 1 or num_matches == 0):
        print(f'Got {num_matches} matches for pattern "{pattern_target}" in prompt "{prompt}". This might indicate a mistake.')
    return characterwise_mask

def get_mask(prompt: str, target: str, verbose: bool = True) -> List[bool]:
    return get_mask_regex(prompt=prompt, pattern_target=re.escape(target), verbose=verbose)
