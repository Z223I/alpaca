# Port Forwarding on EXOS Router for Market Sentinel VM

This guide covers how to expose the Market Sentinel web UI (running in a KVM VM) to the public internet through an Extreme Networks ExtremeXOS (EXOS) router.

## Network Topology

```text
Internet
    |
    v
EXOS Router (WAN: 162.246.236.7, LAN: 192.168.1.1)
    |
    v
KVM Host   192.168.1.168  (Ubuntu, runs libvirt/KVM)
    |  (KVM NAT bridge virbr0: 192.168.122.1/24)
    v
KVM VM     192.168.122.33  (apache-web / apache-web-test)
```

Traffic must traverse **two hops**:

1. EXOS router → KVM host (`192.168.1.168`)
2. KVM host → VM (`192.168.122.33`) via iptables DNAT

---

## Port Reference

| Port | Protocol | Service | Exposure |
| ---- | -------- | ------- | -------- |
| 80 | TCP | Apache HTTP (web UI + CGI) | **External** — forward through both hops |
| 8766 | TCP | WebSocket browser proxy (Time & Sales live trades) | **External** — forward through both hops |
| 8765 | TCP | Trade stream backend (Alpaca SIP feed) | Internal only — VM-local, not forwarded |
| 5000 | TCP | Market Data API (Flask, called by CGI scripts) | Internal only — VM-local, not forwarded |
| 22 | TCP | SSH | Internal only — KVM host access only |

---

## Step 1 — EXOS Router Port Forwarding (Web UI)

> **Public URL:** `http://162.246.236.7:8080/`

### Open the Port Forwarding page

1. Open a browser and navigate to `http://192.168.1.1`.
2. Log in with your admin credentials.
3. Navigate to the **Port Forwarding** page (**Create New Association** form).

### Add rule — HTTP (WAN port 8080 → KVM host port 80)

1. Under **Local Port and IP**, select the **IP Address** radio button.
2. Enter IP: `192.168.1.168`
3. Set **Protocol** to `TCP`.
4. Set **Port Start** to `80` and **Port End** to `80`.
5. Under **Remote IP**, leave **All IP Addresses** selected.
6. Under **WAN Ports**, set **Port Start** to `8080` and **Port End** to `8080`.
7. Click **Save** (or **Apply**).

### Add rule — WebSocket proxy (WAN port 8766 → KVM host port 8766)

1. Click **Create New Association** to start a second rule.
2. Under **Local Port and IP**, select the **IP Address** radio button.
3. Enter IP: `192.168.1.168`
4. Set **Protocol** to `TCP`.
5. Set **Port Start** to `8766` and **Port End** to `8766`.
6. Under **Remote IP**, leave **All IP Addresses** selected.
7. Under **WAN Ports**, set **Port Start** to `8766` and **Port End** to `8766`.
8. Click **Save** (or **Apply**).

### Save configuration

1. Click **Save Configuration** to write the rules to flash so they survive a reboot.

---

## Step 2 — KVM Host iptables DNAT Rules

The KVM host receives traffic on ports 8080 and 8766 and must forward it to the VM at `192.168.122.33`.

### Add DNAT rules

```bash
# Forward external port 8080 → VM port 80 (Apache)
sudo iptables -t nat -A PREROUTING -i <LAN-IFACE> -p tcp --dport 8080 \
  -j DNAT --to-destination 192.168.122.33:80

# Forward port 8766 → VM port 8766 (WebSocket proxy)
sudo iptables -t nat -A PREROUTING -i <LAN-IFACE> -p tcp --dport 8766 \
  -j DNAT --to-destination 192.168.122.33:8766

# Allow forwarded traffic through
sudo iptables -A FORWARD -p tcp -d 192.168.122.33 --dport 80 -j ACCEPT
sudo iptables -A FORWARD -p tcp -d 192.168.122.33 --dport 8766 -j ACCEPT
```

Replace `<LAN-IFACE>` with the KVM host's LAN interface name (e.g., `eno1`, `eth0`).
Find it with:

```bash
ip route get 192.168.1.1 | awk '{print $5; exit}'
```

### Enable IP forwarding (if not already on)

```bash
sudo sysctl -w net.ipv4.ip_forward=1
```

Make it persistent:

```bash
echo "net.ipv4.ip_forward = 1" | sudo tee /etc/sysctl.d/99-ip-forward.conf
```

### Persist iptables rules across reboots

```bash
sudo apt install iptables-persistent
sudo netfilter-persistent save
```

---

## Step 3 — Verify VM UFW Allows the Ports

SSH into the VM and confirm UFW is open on ports 80 and 8766:

```bash
ssh ubuntu@192.168.122.33
sudo ufw status
```

Expected output includes:

```text
Apache Full               ALLOW IN    Anywhere
8766/tcp                  ALLOW IN    Anywhere   # browser WebSocket proxy
```

---

## Step 4 — Test End-to-End

From an external machine (outside your LAN):

```bash
# HTTP — web UI should return 200
curl -v http://162.246.236.7:8080/

# WebSocket proxy — should accept the upgrade
curl -v --no-buffer \
  -H "Upgrade: websocket" \
  -H "Connection: Upgrade" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  http://162.246.236.7:8766/
```

The browser UI at `http://162.246.236.7:8080/` should load the Market Sentinel dashboard with:

- Candlestick chart loading data from the Flask CGI endpoint (port 5000, internal)
- Time & Sales panel receiving live trades via the WebSocket proxy (port 8766, forwarded)

---

## Troubleshooting

| Symptom | Check |
| ------- | ----- |
| HTTP times out at router | EXOS NAT rule missing; verify in the router UI under NAT → Port Forwarding |
| HTTP reaches host but not VM | iptables DNAT not added, or `ip_forward` disabled on host |
| WebSocket connects but no trades | `browser_proxy.service` or `market_sentinel_trade_stream.service` not running in VM; check `systemctl status browser_proxy` |
| "connection limit exceeded" in trade stream logs | Another process is holding the Alpaca SIP WebSocket connection; find with `ss -tp \| grep 8765` and kill the duplicate |
| CGI returns 500 | `market-data-api.service` not running; check `systemctl status market-data-api` |

### Useful commands on the VM

```bash
# Check all four services
systemctl status market-data-api market_sentinel_trade_stream browser_proxy momentum_alerts

# Watch trade stream logs in real time
journalctl -fu market_sentinel_trade_stream

# Watch browser proxy logs
journalctl -fu browser_proxy
```
