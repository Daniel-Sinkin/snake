# type: ignore
def ndc_to_screenspace(x, y, width, height) -> tuple[int, int]:
    return int((x + 1.0) / 2.0 * width), int((-y + 1.0) / 2.0 * height)


def screenspace_to_ndc(x, y, width, height) -> tuple[float, float]:
    return 2.0 * x / width - 1.0, 1.0 - 2.0 * y / height
