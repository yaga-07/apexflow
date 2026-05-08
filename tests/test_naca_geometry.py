import numpy as np
import pytest
from mcp_server.naca_geometry import naca_4digit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def signed_area(coords: np.ndarray) -> float:
    """Shoelace formula; positive → counterclockwise."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---------------------------------------------------------------------------
# Shape and basic properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_points", [40, 100, 200, 400])
def test_output_shape(n_points):
    coords = naca_4digit(0, 0.4, 0.12, n_points=n_points)
    assert coords.shape == (n_points, 2), f"Expected ({n_points}, 2), got {coords.shape}"


def test_counterclockwise_ordering():
    coords = naca_4digit(0.02, 0.4, 0.12, n_points=200)
    assert signed_area(coords) > 0, "Coordinates should be ordered counterclockwise"


# ---------------------------------------------------------------------------
# Trailing-edge closure
# ---------------------------------------------------------------------------

def test_trailing_edge_closed_when_requested():
    coords = naca_4digit(0.02, 0.4, 0.12, n_points=200, closed_te=True)
    gap = np.linalg.norm(coords[0] - coords[-1])
    assert gap < 1e-6, f"TE gap with closed_te=True should be < 1e-6, got {gap}"


def test_trailing_edge_open_without_closure():
    coords = naca_4digit(0.02, 0.4, 0.12, n_points=200, closed_te=False)
    gap = np.linalg.norm(coords[0] - coords[-1])
    # With open TE the two surfaces don't meet exactly at (1,0)
    assert gap > 1e-6, f"TE gap with closed_te=False should be > 1e-6, got {gap}"


# ---------------------------------------------------------------------------
# NACA 0012 — symmetric profile
# ---------------------------------------------------------------------------

class TestNACA0012:
    def setup_method(self):
        self.coords = naca_4digit(m=0, p=0.4, t=0.12, n_points=200, closed_te=True)
        self.half = 100

    def test_shape(self):
        assert self.coords.shape == (200, 2)

    def test_max_thickness_location(self):
        # For NACA 0012 (symmetric), upper surface is first half; max y ≈ 0.06 near x ≈ 0.3
        upper = self.coords[:self.half]
        idx_max = np.argmax(upper[:, 1])
        x_max = upper[idx_max, 0]
        y_max = upper[idx_max, 1]
        assert abs(x_max - 0.3) < 0.05, f"Max thickness x should be near 0.3, got {x_max:.3f}"
        assert abs(y_max - 0.06) < 0.005, f"Max half-thickness should be ≈0.06, got {y_max:.4f}"

    def test_mirror_symmetry(self):
        # For symmetric (m=0), upper and lower surfaces should be mirrors across y=0.
        # upper[k] corresponds to lower reversed at index k (same x, opposite y).
        upper = self.coords[:self.half]
        lower = self.coords[self.half:][::-1]
        np.testing.assert_allclose(upper[:, 0], lower[:, 0], atol=1e-12,
                                   err_msg="Upper/lower x coords should match")
        np.testing.assert_allclose(upper[:, 1], -lower[:, 1], atol=1e-12,
                                   err_msg="Upper/lower y coords should be negatives")

    def test_closed_loop(self):
        gap = np.linalg.norm(self.coords[0] - self.coords[-1])
        assert gap < 1e-6


# ---------------------------------------------------------------------------
# NACA 2412 — cambered profile
# ---------------------------------------------------------------------------

class TestNACA2412:
    def setup_method(self):
        # m=2%=0.02, p=40%=0.4, t=12%=0.12
        self.coords = naca_4digit(m=0.02, p=0.4, t=0.12, n_points=200, closed_te=True)
        self.half = 100

    def test_shape(self):
        assert self.coords.shape == (200, 2)

    def test_upper_surface_max_y(self):
        # Upper surface max y > 0.06 (camber lifts it above pure symmetric)
        upper = self.coords[:self.half]
        y_max = np.max(upper[:, 1])
        assert y_max > 0.06, f"NACA 2412 upper max y should exceed 0.06, got {y_max:.4f}"
        assert y_max < 0.12, f"NACA 2412 upper max y should be < 0.12, got {y_max:.4f}"

    def test_asymmetric_profile(self):
        # Cambered profile: |max upper y| != |min lower y|
        upper = self.coords[:self.half]
        lower = self.coords[self.half:]
        assert not np.isclose(np.max(upper[:, 1]), -np.min(lower[:, 1]), atol=1e-3), \
            "NACA 2412 should be asymmetric (not a mirror-image profile)"

    def test_closed_loop(self):
        gap = np.linalg.norm(self.coords[0] - self.coords[-1])
        assert gap < 1e-6

    def test_max_thickness_near_030_chord(self):
        # Half-thickness is still maximised near x=0.3c regardless of camber
        upper = self.coords[:self.half]
        idx = np.argmax(upper[:, 1])
        x_max = upper[idx, 0]
        # Allow wider tolerance since camber + thickness shift the actual peak slightly
        assert 0.2 < x_max < 0.5, f"Max upper y x-location should be in [0.2, 0.5], got {x_max:.3f}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_negative_thickness_raises():
    with pytest.raises(ValueError, match="[Tt]hickness"):
        naca_4digit(0, 0.4, -0.12)


def test_zero_thickness_raises():
    with pytest.raises(ValueError, match="[Tt]hickness"):
        naca_4digit(0, 0.4, 0.0)


def test_odd_n_points_raises():
    with pytest.raises(ValueError, match="even"):
        naca_4digit(0, 0.4, 0.12, n_points=201)


def test_nonzero_camber_zero_p_raises():
    with pytest.raises(ValueError):
        naca_4digit(m=0.02, p=0.0, t=0.12)
