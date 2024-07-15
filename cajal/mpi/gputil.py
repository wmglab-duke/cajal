from distutils import spawn
import math
import os
import platform
import random
from subprocess import Popen, PIPE, SubprocessError

from mpi4py import MPI

comm = MPI.COMM_WORLD


class GPU:
    """Helper class to handle the attributes of each GPU. Attribute
    descriptions are copied from corresponding descriptions by nvidia-smi.

    Parameters
    ----------
    id : int
        Zero based index of the GPU. Can change at each boot.
    uuid : str
        This value is the globally unique immutable alphanumeric identifier of
        the GPU. It does not correspond to any physical label on the board.
        Does not change across reboots.
    load : float
        Relative GPU load. 0 to 1 (100%, full load). Percent of time over the
        past sample period during which one or more kernels was executing on
        the GPU. The sample period may be between 1 second and 1/6 second depending
        on the product.
    total_memory : float
        Total installed GPU memory.
    used_memory : float
        Total GPU memory allocated by active contexts.
    free_memory : float
        Total free GPU memory.
    driver : str
        The version of the installed NVIDIA display driver.
    name : str
        The official product name of the GPU.
    serial : str
        This number matches the serial number physically printed on each board.
        It is a globally unique immutable alphanumeric value.
    display_connected : str
        A flag that indicates whether a physical display (e.g. monitor) is
        currently connected to any of the GPU's connectors. "Enabled" indicates
        an attached display. "Disabled" indicates otherwise."
    display_active : str
        A flag that indicates whether a display is initialized on the GPU
        (e.g. memory is allocated on the device for display). Display can be
        active even when no monitor is physically attached. "Enabled" indicates
        an active display. "Disabled" indicates otherwise.
    """

    def __init__(
        self,
        ID,
        uuid,
        load,
        total_memory,
        used_memory,
        free_memory,
        driver,
        name,
        serial,
        display_connected,
        display_active,
        temperature,
    ):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.utilised_memory = float(used_memory) / float(total_memory)
        self.total_memory = total_memory
        self.used_memory = used_memory
        self.free_memory = free_memory
        self.driver = driver
        self.name = name
        self.serial = serial
        self.display_connected = display_connected == "Enabled"
        self.display_active = display_active == "Enabled"
        self.temperature = temperature

    def __eq__(self, other):
        if isinstance(other, GPU):
            return (other.uuid == self.uuid) and (other.serial == self.serial)
        return NotImplemented

    def __repr__(self):
        return (
            f"/gpu:{self.id} :: {self.name} {self.total_memory}MB :: "
            f"{(1 - self.utilised_memory)*100:.2f}% available"
        )


def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float("nan")
    return number


def gpus():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi could not be found from
        # the environment path, try to find it from system drive with default
        # installation path
        nvidia_smi = spawn.find_executable("nvidia-smi")
        if nvidia_smi is None:
            env = os.environ["systemdrive"]
            nvidia_smi = (
                f"{env}\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
            )
    else:
        nvidia_smi = "nvidia-smi"

    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,utilization.gpu,memory.total,"
                "memory.used,memory.free,driver_version,name,gpu_serial,"
                "display_active,display_mode,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=PIPE,
        )
        stdout, stderror = p.communicate()
    except (SubprocessError, FileNotFoundError):
        return []
    output = stdout.decode("UTF-8")

    # Split on line break
    lines = output.split(os.linesep)

    numDevices = len(lines) - 1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        vals = line.split(", ")
        GPUs.append(
            GPU(
                ID=int(vals[0]),
                uuid=vals[1],
                load=safeFloatCast(vals[2]) / 100,
                total_memory=safeFloatCast(vals[3]),
                used_memory=safeFloatCast(vals[4]),
                free_memory=safeFloatCast(vals[5]),
                driver=vals[6],
                name=vals[7],
                serial=vals[8],
                display_connected=vals[9],
                display_active=vals[10],
                temperature=safeFloatCast(vals[11]),
            )
        )
    return GPUs


def available(
    order="first",
    limit=1,
    max_load=0.5,
    max_memory=0.5,
    free_memory=0,
    include_nan=False,
    excludeID=None,
    excludeUUID=None,
):
    # order = first | last | random | load | memory
    #    first --> select the GPU with the lowest ID (DEFAULT)
    #    last --> select the GPU with the highest ID
    #    random --> select a random available GPU
    #    load --> select the GPU with the lowest load
    #    memory --> select the GPU with the most memory available
    # limit = 1 (DEFAULT), 2, ..., Inf
    #     Limit sets the upper limit for the number of GPUs to return.
    #     E.g. if limit = 2, but only one is available, only one is returned.

    # Get device IDs, load and memory usage
    GPUs = gpus()

    # Determine, which GPUs are available
    gpu_avail = availability(
        GPUs,
        max_load=max_load,
        max_memory=max_memory,
        free_memory=free_memory,
        include_nan=include_nan,
        excludeID=excludeID,
        excludeUUID=excludeUUID,
    )
    available_gpu_index = [
        idx for idx in range(len(gpu_avail)) if (gpu_avail[idx] == 1)
    ]
    # Discard unavailable GPUs
    GPUs = [GPUs[g] for g in available_gpu_index]

    # Sort available GPUs according to the order argument
    if order == "first":
        GPUs.sort(
            key=lambda x: float("inf") if math.isnan(x.id) else x.id, reverse=False
        )
    elif order == "last":
        GPUs.sort(
            key=lambda x: float("-inf") if math.isnan(x.id) else x.id, reverse=True
        )
    elif order == "random":
        random.shuffle(GPUs)
    elif order == "load":
        GPUs.sort(
            key=lambda x: float("inf") if math.isnan(x.load) else x.load, reverse=False
        )
    elif order == "memory":
        GPUs.sort(
            key=lambda x: float("inf")
            if math.isnan(x.utilised_memory)
            else x.utilised_memory,
            reverse=False,
        )

    # Extract the number of desired GPUs, but limited to the total number of available GPUs
    GPUs = GPUs[0 : min(limit, len(GPUs))]

    # Extract the device IDs from the GPUs and return them
    deviceIds = [gpu.id for gpu in GPUs]

    return deviceIds


def availability(
    GPUs,
    max_load=0.5,
    max_memory=0.5,
    free_memory=0,
    include_nan=False,
    excludeID=None,
    excludeUUID=None,
):
    excludeID = excludeID if excludeID is not None else []
    excludeUUID = excludeUUID if excludeUUID is not None else []
    # Determine, which GPUs are available
    avail = [
        1
        if (gpu.free_memory >= free_memory)
        and (gpu.load < max_load or (include_nan and math.isnan(gpu.load)))
        and (
            gpu.utilised_memory < max_memory
            or (include_nan and math.isnan(gpu.utilised_memory))
        )
        and ((gpu.id not in excludeID) and (gpu.uuid not in excludeUUID))
        else 0
        for gpu in GPUs
    ]
    return avail


def all_rank_ids():
    ids = [gpu.uuid for gpu in gpus()]
    ids = comm.allgather(ids)
    return ids


def all_rank_nodes():
    node = platform.node()
    node = comm.gather(node)
    return node
