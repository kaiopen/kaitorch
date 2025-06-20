from pathlib import Path

from kaitorch.utils.config import Configer


root = Path.cwd()
cfg = Configer()

cfg.merge_(root.joinpath('config.yaml'))
for c in cfg.configs:
    cfg.merge_(root.joinpath(c))

cfg.activate_()
print(cfg)
cfg.freeze_()
cfg.d[0].p = 1
