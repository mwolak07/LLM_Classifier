import psutil


def get_ram_b() -> float:
    """
    Gets the amount of RAM on the system in B.

    Returns:
        The amount of RAM on this computer in B.
    """
    # Getting the memory in B.
    mem = psutil.virtual_memory()
    return mem.total


def get_ram_gb() -> float:
    """
    Gets the amount of RAM on the system in GB.

    Returns:
        The amount of RAM on this computer in GB.
    """
    # Converting bytes to GB.
    return get_ram_b() / (1024 ** 3)
