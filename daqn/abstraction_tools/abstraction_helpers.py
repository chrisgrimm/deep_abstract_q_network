from daqn.abstraction_tools.hero_abstraction import HeroAbstraction
from daqn.abstraction_tools.mr_abstraction_ram import MRAbstraction
from daqn.abstraction_tools.venture_abstraction import VentureAbstraction


def get_abstraction_function(func_id, environment):
    abs = None

    if func_id == 'montezuma_revenge':
        abs = MRAbstraction(environment, use_sectors=True)
    elif func_id == 'venture':
        abs = VentureAbstraction(environment, use_sectors=True)
    elif func_id == 'hero':
        abs = HeroAbstraction(environment, use_sectors=True)

    if abs is None:
        raise Exception('Environment ' + func_id + ' not found')

    abs_func = abs.oo_abstraction_function
    pred_func = abs.predicate_func
    # abs_reset_func = abs.reset

    return abs_func, pred_func