DEV = docker compose --compatibility -f docker-compose.yml -f docker-compose-dev.yml
PROD = docker compose --compatibility -f docker-compose.yml -f docker-compose-production.yml

server-compose-build-nocache:
	docker compose --compatibility build --no-cache

# Local: http://localhost:8001 (foreground / background)
server-compose-interactive:
	$(DEV) up --build

server-compose:
	$(DEV) up --build -d

# ocean-metabolome-portal.gnps2.org (needs the nginx-net network from nginx-proxy)
server-compose-production:
	$(PROD) up --build -d

server-compose-down:
	docker compose down

# Import new or changed datasets from datasets.yaml into the running container.
#   make import                     everything new or changed
#   make import ARGS="go_dom_eb"    one dataset
#   make import ARGS="--force"      re-import everything
#   make import ARGS="--list"       show what is imported
import:
	docker exec ocean-portal python import_datasets.py $(ARGS)

attach:
	docker exec -i -t ocean-portal /bin/bash

logs:
	docker logs -f ocean-portal

.PHONY: server-compose-build-nocache server-compose-interactive server-compose server-compose-production server-compose-down import attach logs
