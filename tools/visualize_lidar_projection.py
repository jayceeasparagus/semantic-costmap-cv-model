#!/usr/bin/env python3
"""Compatibility command for the renamed semantic point-painting demo."""

from paint_semantic_points import main


if __name__ == "__main__":
    print("This command now uses independent calibration projection.")
    main()
