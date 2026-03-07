import torch


class Beam(object):
    """ Abstract base class for all beams. """
    def add_token(self, new_token: torch.FloatTensor) -> None:
        raise NotImplementedError
    
    def copy(self) -> "Beam":
        raise NotImplementedError


class System(object):
    """ Abstract base class for all systems. """
    def load_checkpoint(self, dir: str, *args, **kwargs) -> None:
        """ Load checkpoint from a local directory. """
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, *args, **kwargs) -> str:
        raise NotImplementedError
    
    def eval(self):
        pass

    def cuda(self):
        pass

    def get_beam(self, *args, **kwargs):
        raise NotImplementedError
