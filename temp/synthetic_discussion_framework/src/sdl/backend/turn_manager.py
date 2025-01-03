import abc
import itertools
import random
import warnings


class TurnManager(abc.ABC):
    """
    A class that handles which of a list of users gets to speak in the next dialogue turn.
    """

    def __init__(self, usernames: list[str], config: dict[str, float] = {}):
        """
        Construct a new TurnManager.

        :param usernames: a list of all usernames in the conversation
        :type usernames: list[str]
        :param config: a dictionary of other configurations, defaults to {}
        :type config: dict[str, float], optional
        """
        self.usernames = usernames
        self.config = config

    @abc.abstractmethod
    def next_turn_username(self) -> str:
        """
        Get the username of the next speaker.

        :raises NotImplemented: abstract method
        :return: the next speaker's username
        :rtype: str
        """
        raise NotImplemented


class RoundRobbin(TurnManager):
    """
    A simple turn manager which gives priority to the next user in the queue.
    """

    def __init__(self, usernames: list[str]):
        super().__init__(usernames)
        self.username_loop = itertools.cycle(self.usernames)
        self.curr_turn = 0

    def next_turn_username(self) -> str:
        return next(self.username_loop)


class RandomWeighted(TurnManager):
    """
    Enable a participant to reply with a set probability, else randomly select other participant.
    """

    DEFAULT_RESPOND_PROBABILITY = 0.5

    def __init__(self, usernames: list[str], config: dict[str, float] = {}):
        super().__init__(usernames, config)

        if config.get("respond_probability") is None:
            self.chance_to_respond = RandomWeighted.DEFAULT_RESPOND_PROBABILITY
        else:
            warnings.warn(
                "Warning: No respond_probability set in RandomWeighted TurnManager instance, "
                + f"defaulting to {RandomWeighted.DEFAULT_RESPOND_PROBABILITY}"
            )
            self.chance_to_respond = config["respond_probability"]
            assert 0 < self.chance_to_respond < 1

        self.second_to_last_speaker = None
        self.last_speaker = None

    def next_turn_username(self) -> str:
        # If first time asking for a speaker, return random speaker
        if self.second_to_last_speaker is None:
            next_speaker = self._select_other_random_speaker()
            self.last_speaker = next_speaker
            return next_speaker

        # Check if the last speaker will respond based on the weighted coin flip
        if self._weighted_coin_flip():
            next_speaker = self.last_speaker
        else:
            next_speaker = self._select_other_random_speaker()

        # Update the speaker history
        self.second_to_last_speaker = self.last_speaker
        self.last_speaker = next_speaker

        assert next_speaker is not None
        return next_speaker

    def _weighted_coin_flip(self) -> bool:
        return self.chance_to_respond > random.uniform(0, 1)

    def _select_other_random_speaker(self) -> str:
        other_usernames = [
            username for username in self.usernames if username != self.last_speaker
        ]
        return random.choice(other_usernames)


def turn_manager_factory(
    turn_manager_type: str, usernames: list[str], config: dict[str, float] = {}
) -> TurnManager:
    """
    A factory which returns a instansiated TurnManager of the type specified by a string.

    :param turn_manager_type: the string specifying the concrete TurnManager class.
    Can be of one of "round_robbin", "random_weighted"
    :type turn_manager_type: TurnManager
    :param usernames: a list of all usernames of each participant in the conversation
    :type usernames: list[str]
    :raises ValueError: if turn_manager_type does not match any classes
    :return: the instansiated TurnManager of the specified type
    :rtype: TurnManager
    """
    turn_manager_type = turn_manager_type.lower()
    if turn_manager_type == "round_robin":
        return RoundRobbin(usernames=usernames)
    elif turn_manager_type == "random_weighted":
        return RandomWeighted(usernames=usernames, config=config)
    else:
        raise ValueError(
            f"There is no turn manager option called {turn_manager_type}\n"
            + "Valid values: round_robin, random_weighted"
        )
