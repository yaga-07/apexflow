import numpy as np


def naca_4digit(
    m: float,
    p: float,
    t: float,
    n_points: int = 200,
    closed_te: bool = True,
) -> np.ndarray:
    """
    Generate NACA 4-digit airfoil surface coordinates.

    Parameters
    ----------
    m : max camber as fraction (e.g. 0.02 for 2%)
    p : chordwise position of max camber as fraction (e.g. 0.4 for 40%)
    t : thickness ratio as fraction (e.g. 0.12 for 12%)
    n_points : total surface points; must be even
    closed_te : use modified trailing-edge coefficient (0.1036) to force yt(1) = 0

    Returns
    -------
    np.ndarray, shape (n_points, 2)
        Coordinates ordered counterclockwise from trailing edge.
        When closed_te=True, first and last points are both (1, 0).
    """
    if t <= 0:
        raise ValueError(f"Thickness t must be positive, got {t!r}")
    if n_points < 4 or n_points % 2 != 0:
        raise ValueError(f"n_points must be a positive even integer, got {n_points!r}")
    if m != 0 and p <= 0:
        raise ValueError(f"Camber location p must be > 0 when camber m != 0, got {p!r}")

    half = n_points // 2

    # Cosine-spaced parameter β from 0 → π gives denser sampling near LE and TE.
    # xc maps β to chord position x/c ∈ [0, 1]:  0 = LE,  1 = TE.
    beta = np.linspace(0.0, np.pi, half + 1)   # shape: (half+1,)
    xc   = 0.5 * (1.0 - np.cos(beta))           # shape: (half+1,)  LE→TE

    # yt: NACA half-thickness at each chord station (perpendicular offset from camber line).
    # a4=0.1036 closes the trailing edge exactly (yt(1)=0); 0.1015 leaves a tiny gap.
    a4 = 0.1036 if closed_te else 0.1015
    yt = 5.0 * t * (                             # shape: (half+1,)
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - a4    * xc**4
    )

    # yc: camber-line height above chord; dyc: its slope (used to rotate thickness vector).
    # For symmetric profiles (m=0) both are identically zero — avoids division by p²=0.
    if m == 0.0:
        yc  = np.zeros_like(xc)   # shape: (half+1,)
        dyc = np.zeros_like(xc)   # shape: (half+1,)
    else:
        fore = xc <= p             # shape: (half+1,)  boolean mask — fore vs aft of camber peak
        yc = np.where(
            fore,
            (m / p**2)        * (2.0 * p * xc - xc**2),                          # fore formula
            (m / (1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc**2),      # aft formula
        )                          # shape: (half+1,)
        dyc = np.where(
            fore,
            (2.0 * m / p**2)         * (p - xc),   # fore slope
            (2.0 * m / (1.0 - p)**2) * (p - xc),   # aft slope
        )                          # shape: (half+1,)

    # theta: local camber-line angle; rotates the thickness vector off the chord line.
    theta = np.arctan(dyc)         # shape: (half+1,)

    # (xu, yu): upper surface;  (xl, yl): lower surface — both in LE→TE order.
    xu = xc - yt * np.sin(theta)  # shape: (half+1,)
    yu = yc + yt * np.cos(theta)  # shape: (half+1,)
    xl = xc + yt * np.sin(theta)  # shape: (half+1,)
    yl = yc - yt * np.cos(theta)  # shape: (half+1,)

    # Assemble counterclockwise: TE → upper surface → (LE bridge) → lower surface → TE
    #
    #   xu[::-1][:-1]  reverses upper to TE→LE order, drops the LE point          → shape: (half, 2)
    #   xl[1:]         lower stays LE→TE order, drops the LE point                 → shape: (half, 2)
    #
    # Dropping the LE from both sides removes the duplicate and keeps total = n_points.
    # With closed_te=True: yt(1)=0 and yc(1)=0, so upper[0] = lower[-1] = (1, 0).
    upper = np.column_stack([xu[::-1][:-1], yu[::-1][:-1]])   # shape: (half, 2)
    lower = np.column_stack([xl[1:],        yl[1:]])           # shape: (half, 2)

    return np.vstack([upper, lower])   # shape: (n_points, 2)
