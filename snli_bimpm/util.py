import os
import sys
import json
import argparse
import datetime
import tempfile
import subprocess


def is_return_code_zero(args):
    """Return true iff the given command's return code is zero.
    All the messages to stdout or stderr are suppressed.
    """
    with open(os.devnull, 'wb') as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


def is_under_git_control():
    """Return true iff the current directory is under git control."""
    return is_return_code_zero(['git', 'rev-parse'])


def prepare_output_dir(args, outdir=None, name=None,
                       argv=None, time_format='%Y%m%dT%H%M%S.%f'):
    """Prepare a directory for outputting training results.

    An output directory, which ends with the current datetime string,
    is created. Then the following infomation is saved into the directory:

        args.txt: command line arguments
        command.txt: command itself
        environ.txt: environmental variables

    Additionally, if the current directory is under git control, the following
    information is saved:

        git-head.txt: result of `git rev-parse HEAD`
        git-status.txt: result of `git status`
        git-log.txt: result of `git log`
        git-diff.txt: result of `git diff`

    Args:
        args (dict or argparse.Namespace): Arguments to save
        outdir (str or None): If str is specified, the output
            directory is created under that path. If not specified, it is
            created as a new temporary directory instead.
        argv (list or None): The list of command line arguments passed to a
            script. If not specified, sys.argv is used instead.
        time_format (str): Format used to represent the current datetime. The
        default format is the basic format of ISO 8601.
    Returns:
        Path of the output directory created by this function (str).
    """
    name_str = datetime.datetime.now().strftime(time_format)
    if name is not None:
        name_str += '_' + name
    if outdir is not None:
        if os.path.exists(outdir):
            if not os.path.isdir(outdir):
                raise RuntimeError(
                    '{} is not a directory'.format(outdir))
        outdir = os.path.join(outdir, name_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=name_str)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    if is_under_git_control():
        # Save `git rev-parse HEAD` (SHA of the current commit)
        with open(os.path.join(outdir, 'git-head.txt'), 'wb') as f:
            f.write(subprocess.check_output('git rev-parse HEAD'.split()))

        # Save `git status`
        with open(os.path.join(outdir, 'git-status.txt'), 'wb') as f:
            f.write(subprocess.check_output('git status'.split()))

        # Save `git log`
        with open(os.path.join(outdir, 'git-log.txt'), 'wb') as f:
            f.write(subprocess.check_output('git log'.split()))

        # Save `git diff`
        with open(os.path.join(outdir, 'git-diff.txt'), 'wb') as f:
            f.write(subprocess.check_output('git diff'.split()))

    return outdir
