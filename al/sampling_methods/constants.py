from functools import partial

AL_MAPPING = {}


def get_base_al_mapping():
    from .margin_AL import MarginAL
    from .uniform_sampling import UniformSampling
    from .informative_diverse import InformativeClusterDiverseSampler

    AL_MAPPING["margin"] = MarginAL
    AL_MAPPING["uniform"] = UniformSampling
    AL_MAPPING["inf_div"] = InformativeClusterDiverseSampler


get_base_al_mapping()


def get_al_sampler(name):
    if name in AL_MAPPING and name != "mixture_of_samplers":
        return AL_MAPPING[name]
    raise NotImplementedError("The specified sampler(%s) is not available." % (name))
