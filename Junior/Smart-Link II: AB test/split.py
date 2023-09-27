import random
import string
from typing import Tuple, List
random.seed(42)

class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
            self,
            experiment_id: int,
            groups: Tuple[str] = ("A", "B"),
            group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights
        random.seed(experiment_id)

        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        self.salt = ''.join(random.choice(string.ascii_letters) for i in range(8))

        # Define the group weights if they are not provided equaly distributed
        # Check input group weights. They must be non-negative and sum to 1.
        if not self.group_weights:
            self.group_weights = [0.5, 0.5]

        assert sum(self.group_weights) == 1

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        # Assign the click to a group randomly based on the group weights
        # Return the group id and group name

        click_id = str(click_id) + self.salt
        click_id = hash(click_id)
        res = click_id % 100 / 100
        s = self.group_weights[0]
        for i in range(len(self.groups)):
            if res < s:
                group_id = i
                break
            else:
                s += self.group_weights[i+1]

        return group_id, self.groups[group_id]
