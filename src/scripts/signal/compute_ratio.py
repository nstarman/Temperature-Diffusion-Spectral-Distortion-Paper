"""Simulate the P2LS data on a grid of spll, sprp."""


from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################

with (paths.output / "signal.txt").open("r") as f:
    base, power_info = f.read().split("\\times")
    power = power_info.replace("^{", "e").replace("}", "").lstrip()
    signal = float(base) * float(power)

with (paths.output / "undamped_signal.txt").open("r") as f:
    base, power_info = f.read().split("\\times")
    power = power_info.replace("^{", "e").replace("}", "").lstrip()
    undamped_signal = float(base) * float(power)


# ##############################################################################

with (paths.output / "signal_ratio.txt").open("w") as f:
    f.write(f"{undamped_signal / signal:.0f}")
