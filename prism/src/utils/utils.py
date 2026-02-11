import torch


def cartesian_to_polar(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, eps: float = 1e-9):
    """
    Convert Cartesian (x, y, z) to cylindrical polar coordinates (radius, angle, z).

    Includes epsilon clamping to prevent NaN gradients when radius is near 0.

    Args:
        x, y, z: Input coordinate tensors of any shape.
        eps: Small value to ensure numerical stability for sqrt.

    Returns:
        r, theta, z
        :param eps:
        :param z:
        :param y:
        :param x:
    """
    # Clamp inside sqrt prevents NaN gradients at r=0
    r = torch.sqrt(x.pow(2) + y.pow(2) + eps)
    theta = torch.atan2(y, x)
    return r, theta, z


def compute_angle_difference(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    """
    Calculates the smallest signed angular difference (in radians) between two angles.

    This implementation is fully differentiable and stable across the -π/π boundary.
    It returns values in the range [-π, π].

    Math: atan2(sin(a-b), cos (a-b))
    """
    difference = angle_a - angle_b
    return torch.atan2(torch.sin(difference), torch.cos(difference))