
import os
from datetime import datetime
import os.path as osp
from six.moves import cPickle
from six.moves import range


def download_file(file_url, file_save_path):
    r = requests.get(file_url)

    if r.status_code != 200:
        print('Download error')

    with open(file_save_path, 'wb') as f:
        f.write(r.content)

def update_log_dir(log_dir):
    """
    Get unused log dir
    :return:
    """
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
    if osp.isdir(osp.join(log_dir, timestamp)):
        extra = 1
        while osp.isdir(osp.join(log_dir, timestamp + '(' + str(extra) + ')')):
            extra += 1
        ret = osp.join(log_dir, timestamp + '(' + str(extra) + ')')
    else:
        ret = osp.join(log_dir, timestamp)

    return ret


def create_dir(dir_path):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    """
    Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name):
    """
    Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :return: an generator over the un-pickled items.
    """
    in_file = open(file_name, 'rb')
    size = cPickle.load(in_file)

    for _ in range(size):
        yield cPickle.load(in_file)
    in_file.close()



