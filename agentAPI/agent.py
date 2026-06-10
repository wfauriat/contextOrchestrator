from typing import Callable

from .backend import (
    Backend, ToolCall, ChatResult, StopReason,
    Message, UserMessage, AssistantMessage, ToolResultMessage
)
from .tools import REGISTRY

def _approve_y_n(tool: ToolCall) -> bool:
    print(f"Model wants to run function: {tool.name}")
    print(f"with arguments: {tool.arguments}")
    approve = input("Do you approve command: y/n?\n").strip().lower()
    return approve == "y"


class Agent:
    def __init__(
        self,
        backend: Backend,
        approve: Callable[[ToolCall], bool] = _approve_y_n,
        max_rounds: int = 10):
        self.backend = backend
        self.max_rounds = max_rounds
        self.approve = approve


    def _execute(self, tc:ToolCall) -> str:
        if tc.name not in REGISTRY:
            return f"error: unknown tool {tc.name}"
        if not self.approve(tc):
            return "user declined to use this command"
        try:
            return REGISTRY[tc.name].func(**tc.arguments)
        except Exception as e:
            return f"error: {e}"
    
    def run(self, messages: list[Message]) -> ChatResult:
        result = self.backend.call_model(
                    messages=messages)
        rounds = 0
        while result.stop_reason == StopReason.TOOL and \
            rounds < self.max_rounds:
            messages.append(AssistantMessage(
                result.content, result.tool_calls
            ))
            for tc in result.tool_calls:
                messages.append(ToolResultMessage(tc,
                                    self._execute(tc)))
            result = self.backend.call_model(messages=messages)
            rounds += 1
        messages.append(AssistantMessage(result.content))
        return result
    
    def repl(self):
        messages: list[Message] = []
        while True:
            print("="*80)
            print("User:")
            print("="*80)
            try:
                prompt = input("> ")
                if prompt.strip().lower() == "quit" or prompt == "/quit":
                    print("\nbye")
                    break
                user_msg = UserMessage(content=prompt)
                working = [*messages, user_msg]
                reply = self.run(working)
                messages = working
                print("\n", "="*80)
                print("Assistant:")
                print("="*80)
                print(reply.content, "\n")
                print(f"tokens_in: {reply.tokens_in}, "
                    f"tokens_out: {reply.tokens_out}")
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                break