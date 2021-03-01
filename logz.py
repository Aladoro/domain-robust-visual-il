"""Logging functionality, extending logging from https://github.com/berkeleydeeprlcourse/homework."""

import json
import os.path as osp, time, atexit, os

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class G:
    output_dir = None
    output_file = None
    output_weights = None
    first_row = True
    save_it = 0
    log_headers = []
    log_current_row = {}


def configure_output_dir(d=None, force=False):
    """Set logging directory if provided or to experiments_data/temp/$current_time."""
    G.output_dir = d or "experiments_data/temp/{}".format(int(time.time()))
    if not force:
        assert not osp.exists(
            G.output_dir), "Log dir %s already exists! Delete it first or use a different dir" % G.output_dir
    G.output_weights = "{}/weights".format(G.output_dir)
    os.makedirs(G.output_weights)
    G.output_file = open(osp.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)
    G.first_row = True
    G.save_it = 0
    G.log_headers.clear()
    G.log_current_row.clear()
    print(colorize("Logging data to %s" % G.output_file.name, 'green', bold=True))


def log_tabular(key, value):
    """Log a value of some diagnostic Call this once for each diagnostic quantity, each iteration"""
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
    assert key not in G.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
    G.log_current_row[key] = value


def dump_tabular():
    """Write all of the diagnostics from the current iteration."""
    vals = []
    key_lens = [len(key) for key in G.log_headers]
    max_key_len = max(15, max(key_lens))
    keystr = '%' + '%d' % max_key_len
    fmt = "| " + keystr + "s | %15s |"
    n_slashes = 22 + max_key_len
    print("-" * n_slashes)
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        if hasattr(val, "__float__"):
            valstr = "%8.3g" % val
        else:
            valstr = val
        print(fmt % (key, valstr))
        vals.append(val)
    print("-" * n_slashes)
    if G.output_file is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
            G.first_row = False
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()


def save_params(params):
    """Save used parameters in JSON file."""
    with open(osp.join(G.output_dir, 'params.json'), 'w') as out:
        out.write(json.dumps(params, indent=2, separators=(',', ': '), sort_keys=True))


def load_params(dir, filename="params.json"):
    """Load parameters from JSON file."""
    with open(osp.join(dir, filename), 'r') as inp:
        data = json.loads(inp.read())
    return data


def save_tf_weights(model):
    """Save checkpoint weights of a trained model."""
    save_dir = osp.join(G.output_weights, '{}.h5'.format(G.save_it))
    print(colorize("Saving model weights to %s" % save_dir, 'green'))
    model.save_weights(save_dir)
    G.save_it += 1

