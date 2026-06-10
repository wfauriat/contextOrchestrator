import subprocess
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class BashResult:
    returncode: int
    stdout: str
    stderr: str
    timeout: bool = False 

    def _truncate_head_tail(self, s: str, n: int) -> str:
        if len(s) <= n:
            return s
        n2 = n // 2
        return s[:n2] + f"\n…[truncated {len(s) - n} chars]…\n" + s[-n2:]

    def render(self, max_length: int = 2000) -> str:
        output = [f"exit {self.returncode}"]
        if self.timeout:
            output.append("timed out, output may be incomplete")
        output.append("stdout:")
        output.append(
            self._truncate_head_tail(self.stdout, max_length)
              if self.stdout else "(no output)")
        if self.stderr:
            output.append("stderr:")
            output.append(
            self._truncate_head_tail(self.stderr, max_length))
        formated_output = "\n".join(output)
        return formated_output


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

_TOOLS = [
    Tool(name= "run_bash", 
         description= "Run subprocess inside python to execute bash (unix) commands. Results is readable form stdout and stderr.",
         parameters={"type" : "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]},
        func=lambda command: run_bash(command).render())
]

REGISTRY = {t.name: t for t in _TOOLS}

def _normalize(x: bytes | str | None) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return bytes(x).decode(errors="replace")

def run_bash(command: str) -> BashResult:
    try:
        output = subprocess.run(["bash", "-c", command], shell=False,
                    capture_output=True, check=False,
                    stdin=subprocess.DEVNULL,
                    text=True, timeout=10)
        return BashResult(output.returncode, output.stdout, output.stderr)
    except subprocess.TimeoutExpired as e:
        return BashResult(-9, _normalize(e.stdout),
                          _normalize(e.stderr), timeout=True)