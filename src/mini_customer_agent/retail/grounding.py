"""Keep simulated customer identifiers grounded in their visible information.

This never reads the database, standard actions, or evaluation criteria.
"""
from copy import deepcopy
import re


VERSION = "visible-identifiers-v1"
IDENTIFIERS = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\bORD[-_]\w+\b|(?<!\w)#?\d{5,}(?!\w)", re.I)
GUIDANCE = """
Factual grounding rules (apply throughout this conversation):
- Use the exact identity and facts in the task instructions. Provide known email/name when asked.
- Never invent an email, postal code, order ID, item ID, payment ID, or tracking number.
- If a requested identifier is absent from your instructions and the conversation, say you do not know it and ask the agent to look it up. Do not pretend to look it up yourself.
- Do not adopt an incorrect identity proposed by the agent. Correct it using your known facts.
- You may choose preferences, confirm an offered action, and respond naturally as instructed.
"""


class SimulatorGroundingError(RuntimeError):
    pass


def identifiers(text):
    return {match.group().lstrip("#").lower() for match in IDENTIFIERS.finditer(text)}


def make_grounded_user(spec, scenario, llm_args):
    from tau2.user.user_simulator import UserSimulator
    from tau2.data_model.message import SystemMessage

    class GroundedUser(UserSimulator):
        def __init__(self):
            super().__init__(tools=None, instructions=scenario + GUIDANCE, llm=spec["model"], llm_args=llm_args)
            self.grounding_events = []

        def _generate_next_message(self, message, state):
            visible = scenario + "\n" + "\n".join(str(m.content or "") for m in state.messages)
            if message is not None:
                visible += "\n" + str(message.content or "")
            allowed = identifiers(visible)
            correction = ""
            for attempt in range(3):
                candidate_state = deepcopy(state)
                if correction:
                    candidate_state.system_messages.append(SystemMessage(role="system", content=correction))
                response = super()._generate_next_message(message, candidate_state)
                invented = sorted(identifiers(response.content or "") - allowed)
                if not invented:
                    # Parent appends the incoming message; commit that exactly once.
                    state.messages = candidate_state.messages
                    return response
                self.grounding_events.append({"attempt": attempt + 1, "invented": invented,
                                              "rejected_content": response.content})
                correction = ("Your previous candidate response was rejected because it invented these identifiers: "
                              + ", ".join(invented) + ". Generate a corrected response. Use known facts; "
                              "say you do not know missing identifiers. Do not repeat invented values.")
            raise SimulatorGroundingError("User simulator invented identifiers after three attempts")

    return GroundedUser()
