import dotenv
import os
import json

class FileSystem(object):

    def __init__(self):
        dotenv.load_dotenv(dotenv.find_dotenv())
        self._root = os.environ.get('ROOT')


    def _infer_location(self, loc):
        loc = loc.lower()
        if loc == 'figures':
            fpath = os.path.join(self._root, 'data/figs')
        elif loc == 'output':
            fpath = os.path.join(self._root, 'data/output')
        elif loc == 'plnn/raw':
            fpath = os.path.join(self._root, loc)
        else:
            try:
                with open(loc, 'x') as tmpfile:
                    pass
                fpath = loc
            except OSError:
                raise ValueError('location=%s not valid' % (loc))
        return fpath

    def get_path(self, location, fname):
        fpath = os.path.join(self._infer_location(location), fname)
        return fpath

    def read_json(self, location, fname):
        fpath = os.path.join(self._infer_location(location), fname)
        with open(fpath, 'r') as f:
            json_load = json.load(f)
        return json_load

    def write_json(self, location, fname):
        fpath = os.path.join(self._infer_location(location), fname)
        with open(fpath, mode='w', encoding='utf-8') as f:
            json.dump(json_to_write, f)
        return

