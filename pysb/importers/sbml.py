from __future__ import print_function as _
from pysb.importers.bngl import model_from_bngl
from pysb.bng import _get_bng_path
import subprocess
import os
import tempfile
import shutil


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
    Runs the BioNetGen sbmlTranslator binary.

    For more descriptions of the arguments, see the `sbmlTranslator
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
    verbose : bool, optional
        Print the SBML conversion output to the console if True

    Returns
    -------
    string
        BNGL output filename
    """
    sbmltrans_bin = os.path.join(os.path.dirname(_get_bng_path()),
                                 'bin/sbmlTranslator')
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

    if verbose:
        print("sbmlTranslator command:")
        print(" ".join(sbmltrans_args))

    p = subprocess.Popen(sbmltrans_args,
                         cwd=os.getcwd(),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    if verbose:
        for line in iter(p.stdout.readline, b''):
            print(line, end="")
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

    Limitations
    -----------

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
        Keyword arguments to pass on to :func:`SbmlBuilder.sbml_translator`
    """
    tmpdir = tempfile.mkdtemp()
    verbose = kwargs.get('verbose', False)
    if verbose:
        print("Performing SBML to BNGL translation in temporary "
              "directory %s" % tmpdir)
    try:
        bngl_file = os.path.join(tmpdir, 'model.bngl')
        sbml_translator(filename, bngl_file, **kwargs)
        return model_from_bngl(bngl_file, force=force)
    finally:
        if cleanup:
            shutil.rmtree(tmpdir)
