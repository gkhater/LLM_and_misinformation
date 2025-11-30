"""Compatibility entrypoint.

Runs the Groq Llama-3 70B config by default via the new CLI stack.
"""
from src import cli


def main():
    cli.main()


if __name__ == "__main__":
    main()
