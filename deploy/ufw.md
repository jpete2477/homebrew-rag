# Firewall: LAN only

The API binds `0.0.0.0` so other machines on the LAN can reach it. What keeps
that from meaning "the internet" is the firewall rule plus not forwarding the
port on the router.

```bash
sudo apt install -y ufw

# Replace 10.10.0.0/24 with your own subnet.
sudo ufw allow from 10.10.0.0/24 to any port 8000 proto tcp   # API
sudo ufw allow from 10.10.0.0/24 to any port 6333 proto tcp   # Qdrant (only if
                                                              # you inspect it remotely)
sudo ufw enable
sudo ufw status verbose
```

Two things this rule does *not* do:

- It does not authenticate anyone. Every machine on the subnet can call `/query`
  until you set `RAG_API_KEY`.
- It does not encrypt anything. Fine on a trusted LAN; if this ever needs to be
  reachable from outside — port forwarding, Tailscale, a public domain — put
  Caddy or Nginx in front with TLS and never expose uvicorn directly.
