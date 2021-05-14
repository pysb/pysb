from pysb.importers.bngl import model_from_bngl
import pysb.pathfinder as pf
import subprocess
import os
import tempfile
import shutil
import re
from urllib.request import urlretrieve
from pysb.logging import get_logger, EXTENDED_DEBUG

BIOMODELS_REGEX = re.compile(r'(BIOMD|MODEL)[0-9]{10}')
BIOMODELS_URLS = {
    'ebi': 'http://www.ebi.ac.uk/biomodels-main/download?mid={}',
    'caltech': 'http://biomodels.caltech.edu/download?mid={}'
}


class SbmlTranslationError(Exception):
    pass


def sbml_translator(input_file,
                    output_file=None,
                    convention_file=None,
                    naming_conventions=None,
                    user_structures=None,
                    molecule_id=False,
                    atomize=False,
                    pathway_commons=False,
                    verbose=False):
    """
    Run the BioNetGen sbmlTranslator binary to convert SBML to BNGL

    This function runs the external program sbmlTranslator, included with
    BioNetGen, which converts SBML files to BioNetGen language (BNGL). If
    PySB was installed using "conda", you can install sbmlTranslator using
    "conda install -c alubbock atomizer". sbmlTranslator is bundled with
    BioNetGen if BNG is installed by manual download and unzip.

    Generally, PySB users don't need to run this function directly; an SBML
    model can be imported to PySB in a single step with
    :func:`model_from_sbml`. However, users may wish to note the parameters
    for this function, which alter the way the SBML file is processed. These
    parameters can be supplied as ``**kwargs`` to :func:`model_from_sbml`.

    For more detailed descriptions of the arguments, see the `sbmlTranslator
    documentation <http://bionetgen.org/index.php/SBML2BNGL>`_.

    Parameters
    ----------
    input_file : string
        SBML input filename
    output_file : string, optional
        BNGL output filename
    convention_file : string, optional
        Conventions filename
    naming_conventions : string, optional
        Naming conventions filename
    user_structures : string, optional
        User structures filename
    molecule_id : bool, optional
        Use SBML molecule IDs (True) or names (False).
        IDs are less descriptive but more BNGL friendly. Use only if the
        generated BNGL has syntactic errors
    atomize : bool, optional
        Atomize the model, i.e. attempt to infer molecular structure and
        build rules from the model (True) or just perform a flat import (False)
    pathway_commons : bool, optional
        Use pathway commons to infer molecule binding. This
        setting requires an internet connection and will query the pathway
        commons web service.
    verbose : bool or int, optional (default: False)
        Sets the verbosity level of the logger. See the logging levels and
        constants from Python's logging module for interpretation of integer
        values. False leaves the logging verbosity unchanged, True is equal
        to DEBUG.

    Returns
    -------
    string
        BNGL output filename
    """
    logger = get_logger(__name__, log_level=verbose)
    sbmltrans_bin = pf.get_path('atomizer')

    sbmltrans_args = [sbmltrans_bin, '-i', input_file]
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.bngl'
    sbmltrans_args.extend(['-o', output_file])

    if convention_file:
        sbmltrans_args.extend(['-c', convention_file])

    if naming_conventions:
        sbmltrans_args.extend(['-n', naming_conventions])

    if user_structures:
        sbmltrans_args.extend(['-u', user_structures])

    if molecule_id:
        sbmltrans_args.append('-id')

    if atomize:
        sbmltrans_args.append('-a')

    if pathway_commons:
        sbmltrans_args.append('-p')

    logger.debug("sbmlTranslator command: " + " ".join(sbmltrans_args))

    p = subprocess.Popen(sbmltrans_args,
                         cwd=os.getcwd(),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    if logger.getEffectiveLevel() <= EXTENDED_DEBUG:
        output = "\n".join([line for line in iter(p.stdout.readline, b'')])
        if output:
            logger.log(EXTENDED_DEBUG, "sbmlTranslator output:\n\n" + output)
    (p_out, p_err) = p.communicate()
    if p.returncode:
        raise SbmlTranslationError(p_out.decode('utf-8') + "\n" +
                                   p_err.decode('utf-8'))

    return output_file


def model_from_sbml(filename, force=False, cleanup=True, **kwargs):
    """
    Create a PySB Model object from an Systems Biology Markup Language (SBML)
    file, using BioNetGen's
    `sbmlTranslator <http://bionetgen.org/index.php/SBML2BNGL>`_,
    which can attempt to extrapolate higher-level (rule-based) structure
    from an SBML source file (argument atomize=True). The model is first
    converted into BioNetGen language by sbmlTranslator, then PySB's
    :class:`BnglBuilder` class converts the BioNetGen language model into a
    PySB Model.

    Notes
    -----

    Requires the sbmlTranslator program (also known at Atomizer). If
    PySB was installed using "conda", you can install sbmlTranslator using
    "conda install -c alubbock atomizer". It is bundled with BioNetGen if
    BNG is installed by manual download and unzip.

    Read the `sbmlTranslator documentation
    <http://bionetgen.org/index.php/SBML2BNGL>`_ for further information on
    sbmlTranslator's limitations.

    Parameters
    ----------
    filename :
        A Systems Biology Markup Language .sbml file
    force : bool, optional
        The default, False, will raise an Exception if there are any errors
        importing the model to PySB, e.g. due to unsupported features.
        Setting to True will attempt to ignore any import errors, which may
        lead to a model that only poorly represents the original. Use at own
        risk!
    cleanup : bool
        Delete temporary directory on completion if True. Set to False for
        debugging purposes.
    **kwargs: kwargs
        Keyword arguments to pass on to :func:`sbml_translator`
    """
    logger = get_logger(__name__, log_level=kwargs.get('verbose'))
    tmpdir = tempfile.mkdtemp()
    logger.debug("Performing SBML to BNGL translation in temporary "
                 "directory %s" % tmpdir)
    try:
        bngl_file = os.path.join(tmpdir, 'model.bngl')
        sbml_translator(filename, bngl_file, **kwargs)
        return model_from_bngl(bngl_file, force=force, cleanup=cleanup)
    finally:
        if cleanup:
            shutil.rmtree(tmpdir)


def model_from_biomodels(accession_no, force=False, cleanup=True,
                         mirror='ebi', **kwargs):
    """
    Create a PySB Model based on a BioModels SBML model

    Downloads file from BioModels (https://www.ebi.ac.uk/biomodels-main/)
    and runs it through :func:`model_from_sbml`. See that function for
    further details on additional arguments and implementation details.
    Utilizes BioNetGen's SBMLTranslator.

    Notes
    -----

    Requires the sbmlTranslator program (also known at Atomizer). If
    PySB was installed using "conda", you can install sbmlTranslator using
    "conda install -c alubbock atomizer". It is bundled with BioNetGen if
    BNG is installed by manual download and unzip.

    Read the `sbmlTranslator documentation
    <http://bionetgen.org/index.php/SBML2BNGL>`_ for further information on
    sbmlTranslator's limitations.

    Parameters
    ----------
    accession_no : str
        A BioModels accession number - the string 'BIOMD' followed by 10
        digits, e.g. 'BIOMD0000000001'. For brevity, just the last digits will
        be accepted as a string, e.g. '1' is equivalent the accession number
        in the previous sentence.
    force : bool, optional
        The default, False, will raise an Exception if there are any errors
        importing the model to PySB, e.g. due to unsupported features.
        Setting to True will attempt to ignore any import errors, which may
        lead to a model that only poorly represents the original. Use at own
        risk!
    cleanup : bool
        Delete temporary directory on completion if True. Set to False for
        debugging purposes.
    mirror : str
        Which BioModels mirror to use, either 'ebi' or 'caltech'
    **kwargs: kwargs
        Keyword arguments to pass on to :func:`sbml_translator`

    Examples
    --------

    >>> from pysb.importers.sbml import model_from_biomodels
    >>> model = model_from_biomodels('1')           #doctest: +SKIP
    >>> print(model)                                #doctest: +SKIP
    <Model 'pysb' (monomers: 12, rules: 17, parameters: 37, expressions: 0, ...
    """
    logger = get_logger(__name__, log_level=kwargs.get('verbose'))
    if not BIOMODELS_REGEX.match(accession_no):
        try:
            accession_no = 'BIOMD{:010d}'.format(int(accession_no))
        except ValueError:
            raise ValueError('accession_no must be an integer or a BioModels '
                             'accession number (BIOMDxxxxxxxxxx)')
    logger.info('Importing model {} to PySB'.format(accession_no))
    filename = _download_biomodels(accession_no, mirror=mirror)
    try:
        return model_from_sbml(filename, force=force, cleanup=cleanup,
                               **kwargs)
    finally:
        try:
            os.remove(filename)
        except OSError:
            pass


def _download_biomodels(accession_no, mirror):
    try:
        url_fmt = BIOMODELS_URLS[mirror]
    except KeyError:
        raise ValueError('Unknown Biomodels mirror: "{}". Choices are: {}'
                         .format(mirror, BIOMODELS_URLS.keys()))
    filename, _ = urlretrieve(url_fmt.format(accession_no))
    return filename
