import inspect

try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except Exception:
    CosineLRScheduler = None

if CosineLRScheduler is not None:
    sig = inspect.signature(CosineLRScheduler.__init__)
    # Your timm uses cycle_mul/cycle_decay -> add backward-compat for SimMIM's t_mul/decay_rate
    if "t_mul" not in sig.parameters and "cycle_mul" in sig.parameters:
        _orig_init = CosineLRScheduler.__init__

        def _init_compat(self, *args, **kwargs):
            if "t_mul" in kwargs and "cycle_mul" not in kwargs:
                kwargs["cycle_mul"] = kwargs.pop("t_mul")
            if "decay_rate" in kwargs and "cycle_decay" not in kwargs:
                kwargs["cycle_decay"] = kwargs.pop("decay_rate")
            return _orig_init(self, *args, **kwargs)

        CosineLRScheduler.__init__ = _init_compat
