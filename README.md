# Hexagon Mesh

A network of charging phones that runs AI jobs overnight. Each job is split into small tickets, every ticket runs on 2 phones, and matching results are accepted and credited.

## Layout

- coordinator/ : server (jobs, tickets, verification, credit ledger)
- tools/simulate.py : end-to-end test with fake phones, no models needed
- node/ : phone agent (next)
- bench/ : S25 benchmarks (next)

## Run in Termux

    pkg install python
    pip install -r requirements.txt
    python -m coordinator.app

In a second Termux session:

    python tools/simulate.py

Expected: job complete, 100 vectors, phone-a and phone-b earn credits, cheater earns 0.

## Settings (environment variables)

- HM_CUSTOMER_KEY, HM_NODE_KEY : API keys. Leave unset only for local testing.
- HM_DB : database path (default hexagon_mesh.db)
- HM_HOST / HM_PORT : default 127.0.0.1:8080
- HM_LEASE : seconds a phone has to finish a ticket (default 300)

## API

- POST /jobs : customer submits {job_type, model, input}
- GET /jobs/ID : progress, plus result when complete
- POST /nodes/checkin : phone registers {node_id, wallet, ram_gb, models}
- POST /work/next : phone asks for a ticket
- POST /work/submit : phone returns a result
- GET /nodes/ID/balance : credits earned

Job types: embeddings (input.texts) and text (input.prompts, temperature 0).
