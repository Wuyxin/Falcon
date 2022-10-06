import torch.nn as nn_
from nni.nas.pytorch.search_space_zoo import ENASMacroLayer
import nni.retiarii.nn.pytorch as nn
from nni.nas.pytorch import mutables

def phase_space(mutated_stc, hypo_space):
    search_space = {}

    for module in mutated_stc.modules():
        if isinstance(module, nn_.Sequential) or \
            isinstance(module, nn_.ModuleList) :
            for name, child in module.named_modules():
                if isinstance(child, nn.LayerChoice):
                    search_space[child._label] = list([str(i) for i in range(len(child.candidates))])
                elif isinstance(child, mutables.LayerChoice):
                    search_space[name] = list([str(i) for i in range(len(child.names))])
                elif isinstance(child, ENASMacroLayer):
                    for child_name, subchild in child.named_modules():
                        if isinstance(subchild, nn.LayerChoice):
                            search_space[subchild._label] = list([str(i) for i in range(len(subchild.candidates))])
                        elif isinstance(subchild, mutables.LayerChoice):
                            search_space[name] = list([str(i) for i in range(len(subchild.names))])
                if isinstance(child, nn.InputChoice):
                    search_space[child._label] = list(range(child.n_candidates))
                elif isinstance(child, mutables.InputChoice):
                    search_space[name] = list(range(child.n_candidates))
        else:
            child = module
            if isinstance(child, nn.LayerChoice):
                    search_space[child._label] = list([str(i) for i in range(len(child.candidates))])
            elif isinstance(child, mutables.LayerChoice):
                search_space[name] = list([str(i) for i in range(len(child.names))])
            elif isinstance(child, ENASMacroLayer):
                for subchild in child.modules():
                    if isinstance(child, nn.LayerChoice) or isinstance(child, mutables.LayerChoice):
                        search_space[child._label] = list([str(i) for i in range(len(child.candidates))])
            if isinstance(child, nn.InputChoice):
                search_space[child._label] = list(range(child.n_candidates))
            elif isinstance(child, mutables.InputChoice):
                search_space[name] = list(range(child.n_candidates))

        for key, value in hypo_space.items():
            search_space[key] = value
    return search_space
    