"""Load SciPy before PaddlePaddle's native runtime in the container.

PaddlePaddle 2.6.2 and the SciPy wheels bundled with PaddleOCR can expose
conflicting native compression symbols when Paddle is imported first. Python
loads this module automatically during interpreter startup, so every backend
process uses the safe import order.
"""

import scipy  # noqa: F401
