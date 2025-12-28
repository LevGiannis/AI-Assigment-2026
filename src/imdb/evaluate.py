# Wrapper CLI for classical IMDB evaluation (for test_cli_artifacts compatibility)
from src.imdb.eval_test import main as _main

def main():
    return _main()

if __name__ == "__main__":
    raise SystemExit(main())
