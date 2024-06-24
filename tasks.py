# pylint: disable=missing-function-docstring

"""Project tasks that can be run via command line using inv command."""
import os

from invoke import Context, task  # type: ignore[attr-defined]

from constants import PROJECT_ROOT_PATH, MLFLOW_BACKEND_STORE_PATH, \
    TENSORBOARD_LOGS_PATH

SOURCE_PATH = os.path.join(PROJECT_ROOT_PATH, "world_and_self_model")
TESTS_PATH = os.path.join(PROJECT_ROOT_PATH, "tests")
UNIT_TESTS_PATH = os.path.join(TESTS_PATH, "unit")
COMPONENT_TESTS_PATH = os.path.join(TESTS_PATH, "component")


@task
def test(c: Context, run_mode: str) -> None:
    if run_mode == "coverage":
        c.run(
            f"poetry run pytest "
            f"-n auto "
            f"--randomly-seed=1234 "
            f"--cov-report term-missing:skip-covered "
            f"--cov={SOURCE_PATH} {UNIT_TESTS_PATH} {COMPONENT_TESTS_PATH}"
        )
    else:
        specific_tests_path = os.path.join(TESTS_PATH, run_mode)
        c.run(
            f"poetry run pytest -n auto --randomly-seed=1234 "
            f"{specific_tests_path}"
        )


@task
def mypy(c: Context) -> None:
    c.run(
        f"poetry run mypy --install-types --non-interactive {SOURCE_PATH} "
        f"{TESTS_PATH} constants.py tasks.py"
    )


@task
def black(c: Context, only_check: bool = False) -> None:
    if only_check:
        command = (
            f"poetry run black --check {SOURCE_PATH} {TESTS_PATH} "
            f"constants.py tasks.py"
        )
    else:
        command = (
            f"poetry run black {SOURCE_PATH} {TESTS_PATH} "
            f"constants.py tasks.py"
        )
    c.run(command)


@task
def isort(c: Context, only_check: bool = False) -> None:
    if only_check:
        command = (
            f"poetry run isort --check {SOURCE_PATH} {TESTS_PATH} "
            f"constants.py tasks.py"
        )
    else:
        command = (
            f"poetry run isort {SOURCE_PATH} {TESTS_PATH} "
            f"constants.py tasks.py"
        )
    c.run(command)


@task
def lint(c: Context) -> None:
    c.run(
        f"poetry run pylint --jobs=0 --recursive=y {SOURCE_PATH} "
        f"{TESTS_PATH} constants.py tasks.py"
    )


@task
def bandit(c: Context) -> None:
    c.run(f"poetry run bandit -c pyproject.toml -r {SOURCE_PATH}")


@task
def tensorboard(c: Context) -> None:
    c.run(f"poetry run tensorboard --logdir={TENSORBOARD_LOGS_PATH}")


@task
def mlflow_ui(c: Context) -> None:
    c.run(f"poetry run mlflow ui --backend-store-uri {MLFLOW_BACKEND_STORE_PATH}")
