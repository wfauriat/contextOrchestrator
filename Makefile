PYTHON = /opt/venvs/pyDS/bin/python

.PHONY: help agentAPI chat_ollama chat_anthropic chat_mistral test

help:
	@echo "====help===="
	@echo "make chat_ollama"
	@echo "make chat_anthropic"
	@echo "make chat_mistral"
	@echo "make test"
	@echo "make agentAPI"	
	@echo "============"

agentAPI:
	$(PYTHON) -m agentAPI

chat_ollama:
	$(PYTHON) -m agentAPI -b "ollama"

chat_anthropic:
	$(PYTHON) -m agentAPI -b "anthropic"

chat_mistral:
	$(PYTHON) -m agentAPI -b "mistral"

test:
	$(PYTHON) -m pytest -v