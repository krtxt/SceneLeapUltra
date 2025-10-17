from enum import Enum, auto


class HandModelType(Enum):
    LEAP = auto()

    def __str__(self) -> str:
        """Return enum member name as string."""
        return self.name 