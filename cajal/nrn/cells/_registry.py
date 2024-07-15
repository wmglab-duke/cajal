from cajal.common.logging import logger
from cajal.nrn.cells._cells import Cell
from cajal.nrn.cells._implementations import MRG, Sundt


_AXON_NAMES = {"myelinated": MRG, "mrg": MRG, "c": Sundt, "sundt": Sundt}

_AXON_NAMES_BACKUP = _AXON_NAMES.copy()


def all_valid_cell_names():
    """Return the names of all cells registered to the API."""
    return _AXON_NAMES.keys()


def cell_dictionary():
    """Return dictionary relating names to cell classes."""
    return _AXON_NAMES.copy()


def check_valid_cell(axon_name):
    """Determine if a given cell model is implemented and available
    for use.

    Parameters
    ----------
    axon_name : str
        String alias of axon.

    Returns
    -------
    str
        The axon alias name.

    Raises
    ------
    TypeError
        Exception if axon name is not recognised.
    """
    if axon_name not in _AXON_NAMES:
        raise TypeError(
            "{} is not a valid axon type. Choose from: {}".format(
                axon_name, _AXON_NAMES.keys()
            )
        )
    return axon_name


def get_axon_class(fiber_type):
    """Retrieve the cell class associated with the fibre_type alias.

    Parameters
    ----------
    fiber_type : str, type, Cell
        Alias

    Returns
    -------
    Cell
        Class with the name fibre_type

    Raises
    ------
    ValueError
        Error if the class does not exist.
    """
    if isinstance(fiber_type, str):
        try:
            _ax = _AXON_NAMES[fiber_type]
        except KeyError:
            raise ValueError(
                "{} is not a valid fiber type. Choose from {}".format(
                    fiber_type, set(_AXON_NAMES.keys())
                )
            ) from None
    elif isinstance(fiber_type, Cell):
        return fiber_type.__class__
    elif issubclass(fiber_type, Cell):
        return fiber_type
    else:
        raise TypeError("The fiber_type is not of valid type.")


def get_axon_params(fiber_type):
    """Get all arguments required to instantiate an Axon class.

    Parameters
    ----------
    fiber_type: str
        String alias of Axon.

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
    if fiber_type is None:
        return []
    return get_axon_class(fiber_type).params()


# -- registration --


def register_cell(cell):
    """Register your custom cell / axon class with the cells interface.

    Parameters
    ----------
    cell : type (Cell)
        Subclasses Cell

    Raises
    ------
    TypeError
        Error if class does not subclass Cell.
    """
    if cell.name is None:
        raise ValueError("The cell to be registered must be given a name.")
    if not issubclass(cell, Cell):
        raise TypeError("All registered cells must subclass cajal.nrn.cells.Cell")
    if cell.name in _AXON_NAMES:
        txt = (
            "An axon with the name {} is already registered.".format(cell.name)
            + " This registration overwrites the "
            + "previous registration."
        )
        logger.warning(txt)
    _AXON_NAMES.update({cell.name: cell})


def reset_counts():
    """Reset all cell counts to 0 (used for implicit numbering)."""
    for v in set(_AXON_NAMES.values()):
        v.n = 0


def restore_cells_defaults():
    """Restore defaults within cells API."""
    global _AXON_NAMES  # pylint: disable=global-statement
    _AXON_NAMES = _AXON_NAMES_BACKUP.copy()
