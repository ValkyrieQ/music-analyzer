# Extra trusted CA certificates

Drop `*.crt` files here (PEM format) to have them installed into the image's system trust
store at build time. Leave the directory empty on a normal network — you need nothing here.

## When you need this

If your network runs TLS inspection (Zscaler, Netskope, Blue Coat, a corporate MITM proxy),
the container cannot verify upstream certificates. The symptom is Demucs' first-run weight
download failing and the analysis silently degrading to the HPSS fallback:

```
WARNING app.analysis.separate: falling back to HPSS separation:
  demucs failed: URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>
```

## Exporting your proxy's root CA

macOS:

```bash
security find-certificate -a -c "Zscaler" -p /Library/Keychains/System.keychain \
  > backend/certs/zscaler-root.crt
```

Substitute your proxy vendor's name. To find it, look at who issued the chain your
container actually sees:

```bash
docker compose exec api sh -c \
  'echo | openssl s_client -connect dl.fbaipublicfiles.com:443 \
     -servername dl.fbaipublicfiles.com -showcerts 2>/dev/null | grep "^ *i:"'
```

Linux: the CA is usually already in `/usr/local/share/ca-certificates/` or
`/etc/pki/ca-trust/source/anchors/` — copy the `.crt` from there.

Rebuild after adding a certificate: `docker compose build api`.

## Note

`zscaler-root.crt` in this directory is a public root CA certificate, not a secret — it
contains only a public key. It is specific to this deployment's network, though, so remove
it if you deploy elsewhere.
