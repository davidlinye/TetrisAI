import numpy as np
import json

# source: https://stackoverflow.com/questions/75475315/python-return-json-dumps-got-error-typeerror-object-of-type-int32-is-not-json

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)