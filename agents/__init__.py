import os
import sys

# print('__file__:', __file__)
path = os.path.dirname(os.path.abspath(__file__))
# print('path', path)

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    # print('py:', py)
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    # print('mod:', mod)
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    # print('classes:', classes)
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)