from typing import Protocol, Any

from gymnasium import Space


class Agent(Protocol):
    """Agent Protocol"""

    def act(self, obs: Any) -> Any:
        """Make agent to take action based on the observation.

        Args:
            obs: Agent's observation

        Returns:
            Agent's action
        """
        pass
