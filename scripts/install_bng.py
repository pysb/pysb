#!python

# Script to download and install BioNetGen
import tempfile
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve
import tarfile
import zipfile
import sys
import os
import shutil
import platform
import contextlib

bng_urls = {'Linux': "http://www.csb.pitt.edu/Faculty/Faeder"
            "/?smd_process_download=1&download_id=142",
            'Darwin': "http://www.csb.pitt.edu/Faculty/Faeder"
            "/?smd_process_download=1&download_id=148",
            'Windows': "http://www.csb.pitt.edu/Faculty/Faeder"
            "/?smd_process_download=1&download_id=151"}


def _print_download_progress(count, blocksize, totalsize):
    """ Print download progress to stdout """
    progress = count * blocksize * 100 / totalsize
    sys.stdout.write("\rDownloading BioNetGen, %3d%% complete..." % progress)
    sys.stdout.flush()


def _tar_extract_internal_dir(srctar, destdir, selector):
    """ Extracts files from a tar and moves them """
    if type(selector) is str:
        prefix = selector
        selector = lambda m: m.name.startswith(prefix)
    members = [m for m in srctar.getmembers() if selector(m)]
    for m in members:
        m.name = m.name[m.name]
    srctar.extractall(path=destdir, members=members)


def _get_default_bng_directory():
    """ Returns the BNG installation directory for this platform """
    return "c:/Program Files/BioNetGen" if platform.system() == "Windows" \
        else "/usr/local/share/BioNetGen"


@contextlib.contextmanager
def _make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def install_bng(target_directory=None, verbose=True):
    """ Downloads and installs BioNetGen to the specified directory,
    or platform default location if not specified """
    if target_directory is None:
        target_directory = _get_default_bng_directory()

    if os.path.exists(target_directory):
        raise Exception("BioNetGen directory %s already exists,"
                        "skipping installation" % target_directory)
    os.mkdir(target_directory)
    tf = None

    try:
        # Download BNG
        reporthook = _print_download_progress if verbose else None

        # Check binary availability for this platform
        this_os = platform.system()
        if this_os not in bng_urls.keys():
            raise Exception("Could not find a BioNetGen binary for your "
                            "operating system: %s. Please install it "
                            "manually." % this_os)

        tf, _ = urlretrieve(bng_urls[this_os], reporthook=reporthook)

        # Extract BNG
        if verbose:
            print("\nExtracting BioNetGen to %s..." % target_directory)
        with _make_temp_directory() as tempdir:
            if platform.system() == "Windows":
                with zipfile.ZipFile(tf, mode="r") as zf:
                    zf.extractall(path=tempdir)
            else:
                with tarfile.open(tf, mode="r:gz") as tar:
                    tar.extractall(path=tempdir)
            os.rmdir(target_directory)
            shutil.move(os.path.join(tempdir, os.listdir(tempdir)[0]),
                        target_directory)
    except:
        shutil.rmtree(target_directory, ignore_errors=True)
        raise
    finally:
        try:
            os.remove(tf)
        except:
            pass

    if verbose:
        print("BioNetGen installation complete.")


if __name__ == '__main__':
    install_bng(target_directory=(None if len(sys.argv) == 1 else sys.argv[1]))
