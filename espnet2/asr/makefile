EXEC := pyenv exec

.PHONY: format
format:
	$(EXEC) ruff format
	$(EXEC) ruff check --select I --fix
