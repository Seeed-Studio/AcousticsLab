# Install

One package installs everything: the `acousticslabd` daemon, the web console (`acousticslab-webd`), and the `acousticslab` management CLI, wired up as systemd services.

## Requirements

|          |                                                                                  |
| -------- | -------------------------------------------------------------------------------- |
| OS       | systemd Linux with glibc 2.39+ - Debian 13+, Ubuntu 24.04+, Fedora 40+, RHEL 10+ |
| Arch     | `arm64` (aarch64) or `amd64` (x86_64)                                            |
| Audio    | any ALSA capture device - optional, a 1 kHz test tone stands in without one      |
| NPU      | optional; Rockchip NPU is used when present, else CPU                            |

## Download

Get the package for your architecture from the releases page along with the matching `SHA256SUMS-<arch>`, and verify it:

```bash
sha256sum --ignore-missing -c SHA256SUMS-arm64
```

## Install

```bash
# Debian / Ubuntu
sudo apt install ./acousticslab_<version>_arm64.deb

# Fedora / RHEL
sudo dnf install ./acousticslab-<version>-1.aarch64.rpm
```

This creates the `acousticslab` system user, then enables and starts both services. Confirm:

```bash
sudo acousticslab status
```

Every subsystem should report `ok`. The daemon's socket is group-restricted, so `status` needs root - or add yourself once and skip the `sudo`:

```bash
sudo usermod -aG acousticslab "$USER"   # log out and back in to take effect
```

## Open the console

The console listens on **`127.0.0.1:8080` only**: the control plane has no authentication, so it is not published to the network by default. On the device itself, browse to <http://127.0.0.1:8080>.

From another machine, forward it over SSH:

```bash
ssh -L 8080:127.0.0.1:8080 user@device   # then open http://127.0.0.1:8080
```

## Choose a microphone

Out of the box the daemon opens the ALSA `default` device, falling back to a test tone if that fails. To be explicit:

```bash
acousticslab mic list                        # what the system sees
sudo acousticslab mic use hw:1,0 --restart   # one named device
sudo acousticslab mic use all --restart      # every detected device, auto-failover
```

The daemon records from one microphone at a time; `all` makes it open whichever is live and fail over between them, rather than mixing them. It refuses if the console has the microphone pinned to a specific device, since a pinned mic never fails over.

## NPU acceleration (optional)

The CPU backbone ships with the package. On a Rockchip NPU device, add the matching RKNN backbone and the daemon prefers it automatically, falling back to CPU whenever it is missing or unsupported:

```bash
sudo acousticslab backbone fetch <url> --sha256 <hex>
sudo systemctl restart acousticslabd
```

## Managing the services

```bash
systemctl status acousticslabd acousticslab-webd
journalctl -u acousticslabd -f                   # daemon logs
sudo systemctl disable --now acousticslab-webd   # headless: API only, no console
```

The daemon serves its HTTP/WebSocket/SSE API only on the Unix socket `/run/acousticslab/api.sock` (readable by the `acousticslab` group, never a TCP port); `acousticslab-webd` is what bridges it to a browser.

## Uninstall

```bash
sudo apt purge acousticslab     # also removes /var/lib/acousticslab + the user
sudo dnf remove acousticslab    # keeps /var/lib/acousticslab + the user
```
