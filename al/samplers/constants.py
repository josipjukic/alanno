AL_MAPPING = {}


def get_base_al_mapping():
    from .uncertainty import (
        LeastConfidentSampler,
        MarginSampler,
        EntropySampler,
        MultilabelUncertaintySampler,
    )
    from .random_sampler import RandomSampler
    from .combined_sampler import CombinedEntropyDensitySampler
    from .deep import CoreSet, BADGE

    AL_MAPPING["uniform"] = RandomSampler
    AL_MAPPING["least_conf"] = LeastConfidentSampler
    AL_MAPPING["margin"] = MarginSampler
    AL_MAPPING["entropy"] = EntropySampler
    AL_MAPPING["entropy_density"] = CombinedEntropyDensitySampler
    AL_MAPPING["multilab_uncert"] = MultilabelUncertaintySampler
    AL_MAPPING["core_set"] = CoreSet
    AL_MAPPING["badge"] = BADGE


get_base_al_mapping()


def get_al_sampler(name):
    if name in AL_MAPPING and name != "mixture_of_samplers":
        return AL_MAPPING[name]
    raise NotImplementedError("The specified sampler(%s) is not available." % (name))
