# Federated learning across three laptops

This procedure targets Flower 1.39.0: Windows laptop A is the server,
Windows laptop B is client 0, and the Mac is client 1. Two clients train;
the server coordinates them. Keep each client's CSV on its own laptop.

## Environment

Use Python 3.11 or 3.12 in an activated virtual environment on every laptop.
From the repository root, install the training runtime:

```text
python -m pip install -r requirements-fl.txt "flwr==1.39.0"
```

On macOS, use `python3` if that is your environment's Python command.
The existing partition summary describes two shards. Do not regenerate them
unless deliberately replacing the experiment. Client 0 needs `client_0.csv`;
client 1 needs `client_1.csv`. This checkout currently has only `client_1.csv`.

## Start the server on Windows A

```text
flower-superlink --insecure --fleet-api-address 0.0.0.0:9092 --disable-runtime-dependency-installation
```

Run this in the environment where requirements-fl.txt was installed. The flag
keeps the server using that prepared environment. This is an unencrypted lab
setup; TLS, authentication, secure aggregation and DP remain separate work.
Allow the two client laptops to reach TCP port 9092 on the server's private LAN.

## Connect the two clients

Replace `SERVER_LAN_IP` and the example paths. Use absolute paths with forward
slashes on Windows. TOML string values inside node-config need quotes.

Windows B, PowerShell (one line):

```powershell
flower-supernode --insecure --superlink SERVER_LAN_IP:9092 --node-config "partition-id=0 num-partitions=2 data-path='C:/upi-data/client_0.csv'"
```

Mac, Terminal (one line):

```bash
flower-supernode --insecure --superlink SERVER_LAN_IP:9092 --node-config "partition-id=1 num-partitions=2 data-path='/Users/yourname/upi-data/client_1.csv'"
```

Wait for both nodes to connect. Duplicate partition IDs are rejected. A run
also fails if fewer than two clients are connected when it starts.

## Submit training and display server logs

On Windows A, run `flwr config list` to locate Flower's config.toml. Add:

```toml
[superlink.upi-lan]
address = "127.0.0.1:8000"
insecure = true
```

Create a run override TOML file on the server, for example `lan-run.toml`,
using the server's actual repository path:

```toml
num-trees = 3
min-clients = 2
forest-every-level = 1
eval-every-trees = 0
out-dir = 'C:/path/to/Test_Fl_Multi_Device'
```

From the repository root, submit the smoke run:

```text
flwr run fl upi-lan --stream --run-config lan-run.toml
```

`--stream` displays ServerApp output. The default `flwr run .` connection can
launch a local simulation instead of using your laptops. Use the named
`upi-lan` connection explicitly. The old `flwr server` and `flwr client`
commands are not valid in Flower 1.39.

After the smoke run succeeds, set num-trees to 60. To evaluate during training,
set eval-every-trees to 10 and server-test-path to the absolute path of the
held-out server_test.csv. Never use a client training shard as the test set.

## What the demo should show

- `[send]`: target hostname, depth and whether the completed forest was sent.
- `[receive]`: client partition, synchronized tree count and histogram bytes.
- `[aggregate]`: all contributing node IDs and the histogram sum operation.
- `[model]`: the new tree's node count and leaf values.

Clients compute gradient, Hessian and count histograms locally. The server sums
those statistics and constructs the global tree; it does not average neural
network weights. The last tree is saved on the server; deployment of the final
model to a POS inference service is a later milestone.

A restarted ClientApp recovers its score from the supplied forest. If a client
fails to contribute after three attempts, training aborts rather than using a
partial aggregate. Already accepted replies are not counted twice.

## Evidence and checks

After successful completion, inspect these files under out-dir:

- models/fl/global_final.json: completed global model.
- reports/fl/fl_rounds.json: participating hostnames, partition IDs and tree log.

For this setup, every completed tree should show the Windows B and Mac client
hostnames. Hostnames are client-reported evidence, not authenticated identities.
Save the streamed terminal output with the demo evidence. Histogram byte counts
exclude transport overhead and server-to-client traffic.

Run the protocol regression suite from the repository root:

```text
python -m unittest discover -s tests -v
```

These tests use real Flower message objects with an in-memory transport. They
cover restarts, resync, duplicate replies, node/run isolation and missing
participants. They do not prove network connectivity or execution on the Mac.

Command reference: [Flower 1.39 deployment guide](https://flower.ai/docs/framework/1.39/en/how-to-run-flower-with-deployment-engine.html).
