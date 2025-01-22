"""
custom exeptions
"""

from pathlib import Path


class InvalidStepError(Exception):
    pass


class BadImageFolder(InvalidStepError):
    def __init__(self, image_folder: str):
        self.image_folder = image_folder
        super().__init__(f"missing image {image_folder}")


class BadInputImages(InvalidStepError):
    def __init__(self, input_paths: list[Path]):
        self.input_paths = input_paths
        super().__init__(
            f"{', '.join([str(p) for p in input_paths])} are at the wrong step"
        )


class SkippingStep(InvalidStepError):
    def __init__(self, current_step: int, image_step: int):
        self.current_step = current_step
        self.image_step = image_step
        super().__init__(
            f"You are trying to run a step with number {current_step}"
            f" on an image that is at {image_step}"
        )


class NoRawData(InvalidStepError):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        super().__init__(f"None of [{', '.join([str(p) for p in paths])}] can be read")


class ChannelValidationError(InvalidStepError):
    pass


class UninitializedDatabase(InvalidStepError):
    pass


class CannotFindTemplate(InvalidStepError):
    pass
