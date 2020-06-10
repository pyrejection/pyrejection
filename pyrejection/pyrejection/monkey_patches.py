# When this module is imported, joblib's logging function is
# monkey-patched to include a timestamp in each message.

import joblib.parallel
from datetime import datetime
import sys


# https://github.com/joblib/joblib/blob/a41bd5a71a1889a46fb635831fdf2767e2dc89fe/joblib/parallel.py#L837
def joblib_parallel_timestamped_print(self, msg, msg_args):
    """Display the message on stout or stderr depending on verbosity"""
    # XXX: Not using the logger framework: need to
    # learn to use logger better.
    if not self.verbose:
        return
    # Always use stdout, not stderr
    writer = sys.stdout.write
    msg = msg % msg_args
    # Added timestamp to log output.
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    writer('[%s]: %s (%s)\n' % (self, msg, now))


joblib.parallel.Parallel._print = joblib_parallel_timestamped_print
