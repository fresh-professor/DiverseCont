from .purified_buffer import PurifiedBuffer
from .delay_buffer import DelayBuffer
from .diverse_buffer import DiverseBuffer

reservoir = {
    'purified': PurifiedBuffer,
    'delay':  DelayBuffer,
    'diverse': DiverseBuffer
}
