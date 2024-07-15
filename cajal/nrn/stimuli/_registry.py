from cajal.common.logging import logger

from cajal.nrn.stimuli._stimuli import *


_STIM_NAMES = {
    "monophasic_pulse": MonophasicPulse,
    "sinusoid": Sinusoid,
    "sine_wave": SineWave,
    "cosine_wave": CosineWave,
    "arbitrary": Arbitrary,
    "step": Step,
    "sine_pulse": SinePulse,
    "cosine_pulse": CosinePulse,
    "sinusoid_pulse": SinusoidPulse,
}

_STIM_NAMES_BACKUP = _STIM_NAMES.copy()


def all_valid_stimulus_names():
    """Return the names of all stimuli registered to the API."""
    return _STIM_NAMES.keys()


def stimulus_dictionary():
    """Return dictionary relating names to stimulus classes."""
    return _STIM_NAMES.copy()


def check_valid_stimulus(stimulus_name):
    """Determine if a given stimulus modality is implemented and
    available for use.

    Parameters
    ----------
    stimulus_name : str
        String alias of stimulus.

    Returns
    -------
    str
        The stimulus alias name.

    Raises
    ------
    TypeError
        Exception if stimulus name is not recognised.
    """
    if stimulus_name not in _STIM_NAMES:
        raise TypeError(
            "{} is not a valid specification type. Choose from: {}".format(
                stimulus_name, _STIM_NAMES.keys()
            )
        )
    return stimulus_name


def get_stim_class(stimulus):
    """Convert a stimulus string alias to the stimulus class.

    Parameters
    ----------
    stimulus : str, type, Stimulus
        String alias of the stimulus (defined under the class
        attribute 'name'.)

    Returns
    -------
    Stimulus
        Stimulus class

    Raises
    ------
    ValueError
        Error if a class with the given alias is not implemented
        and registered with the stimuli interface.
    """
    if isinstance(stimulus, str):
        try:
            return _STIM_NAMES[stimulus]
        except KeyError:
            raise ValueError(
                "{} is not a valid stimulus type".format(stimulus)
            ) from None
    elif isinstance(stimulus, Stimulus):
        return stimulus.__class__
    elif issubclass(stimulus, Stimulus):
        return stimulus
    else:
        raise TypeError("The stimulus is not of valid type.")


def get_stim_params(stimulus_name):
    """Get all arguments required to instantiate a stimulus.

    Parameters
    ----------
    stimulus_name : str
        String alias of stimulus.

    Returns
    -------
    list
        List of parameter names.

    Raises
    ------
    ValueError
        Error if a class with the given alias is not implemented
        and registered with the stimuli interface.
    """
    if stimulus_name is None:
        return []
    _stim = get_stim_class(stimulus_name)
    return _stim.params()


# -- registration --


def register_stimulus(stimulus):
    """
    Register a custom stimulus with the API.
    """
    if stimulus.name is None:
        raise ValueError("The stimulus to be registered must be given a name.")
    if not issubclass(stimulus, Stimulus):
        raise TypeError(
            "All registered stimuli must subclass cajal.nrn.stimuli.Stimulus"
        )
    if stimulus.name in _STIM_NAMES:
        txt = (
            "A stimulus with the name {} is already registered.".format(stimulus.name)
            + " This registration overwrites the "
            + "previous registration."
        )
        logger.warning(txt)
    _STIM_NAMES.update({stimulus.name: stimulus})


def restore_stimulus_defaults():
    """Restore defaults within stimulus API."""
    global _STIM_NAMES  # pylint: disable=global-statement
    _STIM_NAMES = _STIM_NAMES_BACKUP.copy()
