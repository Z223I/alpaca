# Apache2 VM Setup for public_html

## Overview

This guide covers creating a VM to serve `public_html` files via Apache2, hardening it against attacks, and setting up a recovery strategy using VM snapshots.

**Q: Can you create a VM with everything necessary?**
Yes. A VM with Apache2, your public_html files, and a basic web stack is straightforward to set up.

**Q: Can you make it resistant to hacking?**
Yes, significantly. No system is 100% unhackable, but hardening reduces the attack surface dramatically.

**Q: Can you just restart the VM if hacked?**
Yes — this is a key advantage of VMs. With a clean snapshot taken before any hack, you can revert in minutes. This is called the **immutable/disposable VM** pattern.

---

## Hypervisor: KVM (Recommended)

KVM (Kernel-based Virtual Machine) is built into the Linux kernel and is the best performing option for Ubuntu hosts.

### Install KVM

```bash
sudo apt install qemu-kvm libvirt-daemon-system virt-manager
```

### Can multiple KVM VMs run simultaneously?

Yes. The only limits are practical — RAM, CPU cores, and disk space. A minimal Ubuntu Server VM for Apache2 needs only **512MB–1GB RAM** and **10–20GB disk**, so you can run many VMs on modest hardware.

**Example** — a host with 32GB RAM could run:

- 1 VM with 16GB RAM
- 4 VMs with 4GB RAM each
- 8 VMs with 2GB RAM each (fine for lightweight web servers)

### Managing multiple VMs

```bash
# List all VMs
virsh list --all

# Start a VM
virsh start vm-name

# Snapshot before changes
virsh snapshot-create-as vm-name snapshot-name

# Revert to snapshot
virsh snapshot-revert vm-name snapshot-name
```

`virt-manager` provides a GUI dashboard to manage all VMs from one place.

### Multi-VM use cases

- **Separate VM per website** — isolates each site so a hack on one doesn't affect others
- **Staging + Production** — identical VMs, one for testing changes before pushing live
- **Snapshot-based recovery** — keep a golden master VM, clone it for each deployment

Running one VM per public-facing service is a security best practice.

### Other hypervisor options

| Hypervisor | Type | Best for |
| --- | --- | --- |
| KVM | Type 1 (bare metal) | Servers, best performance |
| VirtualBox | Type 2 (hosted) | Desktop, easy GUI |
| VMware Workstation | Type 2 (hosted) | Commercial, feature-rich |
| GNOME Boxes | Type 2 (hosted) | Beginners, simple UI |
| LXD/LXC | Containers | Lightweight Linux isolation |

---

## To-Do List

### VM Setup

- [ ] Choose hypervisor or cloud provider (KVM recommended for Ubuntu servers)
- [ ] Install Ubuntu Server LTS (minimal install)
- [ ] Install Apache2: `sudo apt install apache2`
- [ ] Configure `public_html` directory or virtual host in `/etc/apache2/sites-available/`

- [ ] Copy `public_html` into the VM over SSH

  ```bash
  # Use rsync to sync changes incrementally (faster on repeat runs)
  rsync -avz --delete /path/to/public_html/ user@vm-ip:/var/www/html/
  ```

  Fix ownership after copying so Apache can serve the files:

  ```bash
  ssh user@vm-ip "sudo chown -R root:www-data /var/www/html && sudo chmod -R 750 /var/www/html"
  ```

- [ ] Test pages load correctly

### Security Hardening

- [ ] **Firewall**: Enable `ufw`, allow only ports 80, 443, and SSH (22 or custom port)

  ```bash
  sudo ufw allow OpenSSH
  sudo ufw allow 'Apache Full'
  sudo ufw enable
  ```

- [ ] **SSH hardening**: Disable root login, disable password auth, use key-only auth

  ```text
  # /etc/ssh/sshd_config
  PermitRootLogin no
  PasswordAuthentication no
  ```

- [ ] **Fail2ban**: Install to block brute-force attempts

  ```bash
  sudo apt install fail2ban
  ```

- [ ] **Disable directory listing**: Set `Options -Indexes` in Apache config

- [ ] **Hide Apache version**: Set in `/etc/apache2/conf-available/security.conf`

  ```text
  ServerTokens Prod
  ServerSignature Off
  ```

- [ ] **TLS/HTTPS**: Install Certbot + Let's Encrypt for free SSL

  ```bash
  sudo apt install certbot python3-certbot-apache
  sudo certbot --apache
  ```

- [ ] **Keep OS/packages updated**: Enable automatic security patches

  ```bash
  sudo apt install unattended-upgrades
  sudo dpkg-reconfigure unattended-upgrades
  ```

- [ ] **ModSecurity**: Install Apache WAF (Web Application Firewall) module

  ```bash
  sudo apt install libapache2-mod-security2
  sudo a2enmod security2
  ```

- [ ] **Limit file permissions**: Apache user (`www-data`) should have read-only access to web files

  ```bash
  sudo chown -R root:www-data /var/www/html
  sudo chmod -R 750 /var/www/html
  ```

- [ ] **Disable unused Apache modules**: Remove what you don't need

  ```bash
  sudo a2dismod autoindex status
  ```

- [ ] **Regular backups**: Snapshot the VM before deploying any changes

### Recovery Plan

- [ ] **Take a clean snapshot** after initial setup and hardening (before going live)
- [ ] **Document the restore procedure**: Know how to revert to snapshot in your hypervisor/cloud console
- [ ] **Separate data from config**: Store content/data outside the VM if possible (e.g., NFS, S3) so revert doesn't lose content
- [ ] **Set up monitoring/alerting**: Use `Logwatch`, `Tripwire`, or cloud monitoring to detect compromise early

  ```bash
  sudo apt install logwatch
  ```

- [ ] **Automate re-provisioning**: A shell script or Ansible playbook to rebuild the VM from scratch if needed

### Optional (Advanced)

- [ ] Use a **read-only filesystem** for web files (mount as read-only)
- [ ] Deploy behind a **CDN/reverse proxy** (Cloudflare) to hide the origin IP
- [ ] Enable **GeoIP blocking** if traffic should only come from specific countries
- [ ] Set up **intrusion detection** (OSSEC or Wazuh)

---

## Recovery Strategy

**Recommended approach**: Take a VM snapshot after full hardening is complete. If compromised:

1. Revert to the clean snapshot in your hypervisor/cloud console
2. Investigate logs from the compromised instance before going live again
3. Patch the vulnerability that was exploited
4. Take a new clean snapshot

For production environments, consider **infrastructure-as-code** (Terraform + Ansible) so the VM is fully rebuilable from scratch in minutes with no manual steps.
