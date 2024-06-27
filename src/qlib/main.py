"""Main module."""

from qlib.brownian import brownian_trajectories, plot_brownians


def main() -> None:
    """Test."""
    p = brownian_trajectories(1, size=100, n=10)
    plot_brownians(p)


if __name__ == "__main__":
    main()
