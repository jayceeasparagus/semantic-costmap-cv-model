# Repository guide

This file explains the pieces that may be unfamiliar and why they are—or are
not—present right now.

## `requirements.txt`

This is a plain list of third-party Python libraries. Running
`python -m pip install -r requirements.txt` installs that list into the active
virtual environment.

It does not contain project code and does not automatically run anything.

## Why `pyproject.toml` was removed

`pyproject.toml` is the modern configuration file for packaging a Python project
as an installable library. It can also configure tools such as pytest and code
formatters.

That is useful later, but the project does not yet have a stable reusable Python
package. Introducing packaging now made basic data work feel more complicated
than it needed to be. We can add it when multiple scripts genuinely need to
import shared modules.

TOML itself was not a downloaded dependency. It was only a text configuration
format.

## `configs/a2d2_class_list.json`

This JSON file is A2D2's color legend: it connects each RGB label color to a
source class name. It is dataset metadata, not model architecture.

Our reduced six-class mapping will eventually be written in straightforward
Python near the label-conversion code, where it is easy to read and debug.

## `notebooks/`

The first useful artifact will be a clean Colab notebook for dataset inspection,
training, validation, checkpoint saving, and visual predictions. A notebook is
appropriate here because Colab supplies the GPU and makes plots easy to inspect.

Reusable logic should move into normal Python files only after it is working and
used in more than one place.

## `data/` and `outputs/`

These folders are ignored by Git because datasets and model checkpoints are
large. Git stores source code and small configuration files; it should not store
gigabytes of downloaded or generated artifacts.

Ignoring a file does not delete it from your computer.

## Why there are no tests yet

Tests are valuable, but a test should protect behavior that actually exists.
The previous structure created several layers of test and package machinery
before the training pipeline was settled. We will add focused tests as soon as
we create reusable label conversion, projection, and costmap functions.

## Why ROS 2 and Docker are not folders yet

ROS 2 becomes relevant after the offline cost-grid code works. Docker becomes
relevant after dependencies and runnable commands are stable. Delaying their
folders does not remove them from the project; it keeps the current phase clear.

## Why PyTorch is not in `requirements.txt` yet

PyTorch packages depend on the CPU/CUDA runtime. Colab already supplies a
compatible build, while a local WSL installation may need a different command.
We will record exact versions after the clean training notebook succeeds.

## Git safety used during this cleanup

The first attempt remains on `archive/first-attempt`. The unfinished edits that
were in the working tree were saved in a Git stash named:

```text
unfinished label conversion before structure simplification
```

A stash is a recoverable snapshot, not a deletion. The previous committed files
also remain in the `main` branch history even though this cleanup removes them
from the current version.

## The rule going forward

Add a tool or structure when the project has a concrete need for it, and explain
that need before using it. The intended order is model, fusion, costmap, then ROS
2/Nav2, followed by tests, benchmarks, and Docker as each becomes relevant.
