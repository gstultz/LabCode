import os
import re
import fnmatch
import logbook
import sys
from shutil import copyfile


origin_folder = os.path.join(os.path.expanduser('~'),
                             'Dropbox (MIT)',
                             'Little Machine Data')
dest_folder = os.path.join(os.path.expanduser('~'),
                           'Dropbox (MIT)',
                           'littlemachine')

log = logbook.Logger('Program')


def file_transfer():
    init_logging()
    for filename, filepath in gen_files(origin_folder):
        match = re.search(r'^\w{3}\d{2}_\d{2}\.[arecdARECD]\d{2}$', filename)
        if match:
            origin_new_folder = create_folder(filename, origin=True)
            origin = os.path.join(origin_new_folder, filename)
            os.rename(filepath, origin)

            dest_new_folder = create_folder(filename, origin=False)
            dest = os.path.join(dest_new_folder, filename)
            if not os.path.isfile(dest):
                log.trace('File {} copied to {}'.format(origin, dest))
                copyfile(origin, dest)
    log.trace('Files transfer completed'.format(origin, dest))


def gen_files(folder):
    """
    List all files under the folder.
    """
    assert os.path.isdir(folder), 'Folder does not exist'
    log.trace('Folder located successfully: {}'.format(folder))
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, '*'):
            yield (filename, os.path.join(root, filename))


def create_folder(filename, origin=True):
    """
    Create src and dest folder if not exist, and return the folder path.
    """
    technique_dict = {'a': 'Auger', 'e': 'EELS', 'r': 'RGA', 'c': 'TDS',
                      'd': 'TDS'}
    month_dict = generate_month_dict()
    month_folder = month_dict[filename[:3].lower()] + '_' + filename[:3].lower()
    technique_folder = technique_dict[filename[9].lower()]
    if origin:
        folder = os.path.join(origin_folder,
                              technique_folder,
                              '20' + filename[6:8],
                              month_folder)
    else:
        folder = os.path.join(dest_folder,
                              '20' + filename[6:8],
                              month_folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)
        log.trace('Folder created: {}'.format(folder))
    return folder


def generate_month_dict():
    """
    Return a dictionary that maps month names to digital format.
    """
    keys = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    values = ['01', '02', '03', '04', '05', '06',
              '07', '08', '09', '10', '11', '12']
    return dict(zip(keys, values))


def init_logging(filename=None):
    """
    Initialize logging for this module.
    """
    level = logbook.TRACE
    if filename:
        logbook.TimedRotatingFileHandler(
            filename, level=level
        ).push_application()
    else:
        logbook.StreamHandler(sys.stdout, level=level).push_application()

    msg = 'Logging initialized, level: {}, mode: {}'.format(
        level,
        "stdout mod" if not filename else "file mode: " + filename
    )
    logger = logbook.Logger('Startup')
    logger.notice(msg)


if __name__ == '__main__':
    file_transfer()
