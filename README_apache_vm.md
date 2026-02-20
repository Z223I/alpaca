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

- [ ] Install KVM and cloud-image tools on the host

  ```bash
  sudo apt install qemu-kvm libvirt-daemon-system virtinst cloud-image-utils
  sudo systemctl enable --now libvirtd
  sudo usermod -aG libvirt,kvm $USER
  ```

- [ ] Generate an SSH key pair for VM access

  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "kvm-vm-access"
  ```

- [ ] Download Ubuntu 24.04 LTS cloud image (no interactive installer needed)

  ```bash
  sudo wget -O /var/lib/libvirt/images/ubuntu-24.04-cloud.img \
    https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img
  ```

- [ ] Create cloud-init files to automate first boot (user, SSH key, packages)

  ```bash
  PUB_KEY=$(cat ~/.ssh/id_ed25519.pub)

  cat > /tmp/user-data <<EOF
  #cloud-config
  hostname: apache-web
  users:
    - name: ubuntu
      groups: sudo
      shell: /bin/bash
      sudo: ALL=(ALL) NOPASSWD:ALL
      ssh_authorized_keys:
        - ${PUB_KEY}
  packages:
    - apache2
    - ufw
    - fail2ban
  package_update: true
  runcmd:
    - systemctl enable apache2
    - ufw allow OpenSSH
    - ufw allow 'Apache Full'
    - ufw --force enable
  EOF

  cat > /tmp/meta-data <<EOF
  instance-id: apache-web-01
  local-hostname: apache-web
  EOF

  cloud-localds /tmp/apache-web-seed.iso /tmp/user-data /tmp/meta-data
  sudo cp /tmp/apache-web-seed.iso /var/lib/libvirt/images/apache-web-seed.iso
  ```

- [ ] Create the VM disk and boot the VM

  ```bash
  sudo cp /var/lib/libvirt/images/ubuntu-24.04-cloud.img /var/lib/libvirt/images/apache-web.qcow2
  sudo qemu-img resize /var/lib/libvirt/images/apache-web.qcow2 20G

  sudo virt-install \
    --name apache-web \
    --memory 1024 \
    --vcpus 2 \
    --disk /var/lib/libvirt/images/apache-web.qcow2,format=qcow2 \
    --disk /var/lib/libvirt/images/apache-web-seed.iso,device=cdrom \
    --os-variant ubuntu24.04 \
    --network network=default \
    --graphics none \
    --noautoconsole \
    --import \
    --boot hd
  ```

- [ ] Wait for VM IP and SSH to become available

  ```bash
  # Get VM IP (may take 30-60s for DHCP)
  sudo virsh domifaddr apache-web

  # Wait for SSH
  ssh -i ~/.ssh/id_ed25519 ubuntu@<vm-ip> 'echo ready'

  # Wait for cloud-init to finish installing packages
  ssh -i ~/.ssh/id_ed25519 ubuntu@<vm-ip> 'sudo cloud-init status --wait'
  ```

- [ ] Copy web files and application directories into the VM

  ```bash
  VM=ubuntu@<vm-ip>
  SSH_OPTS="-i ~/.ssh/id_ed25519"

  # Copy public_html to web root
  rsync -avz --delete -e "ssh $SSH_OPTS" ./public_html/ $VM:/tmp/public_html_staging/
  ssh $SSH_OPTS $VM "sudo cp -r /tmp/public_html_staging/. /var/www/html/"

  # Copy application directories and .env credentials
  scp $SSH_OPTS -r ./cgi-bin ./analysis ./atoms ./code ./molecules ./services ./python-holidays ./historical_data ./.env $VM:~/
  ```

  Fix ownership so Apache can serve the files:

  ```bash
  ssh $SSH_OPTS $VM "sudo chown -R root:www-data /var/www/html && sudo chmod -R 750 /var/www/html"
  ```

- [ ] Install Miniconda and create the `alpaca` conda environment

  ```bash
  ssh $SSH_OPTS $VM "
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh &&
    bash /tmp/miniconda.sh -b -p ~/miniconda3 &&
    ~/miniconda3/bin/conda init bash &&
    ~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &&
    ~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r &&
    ~/miniconda3/bin/conda create -n alpaca python=3.12 -y
  "
  ```

- [ ] Install Python packages into the `alpaca` conda environment

  Use `--ignore-installed` to ensure packages go into the conda env's own
  site-packages, not `~/.local` (which `www-data` cannot read):

  ```bash
  ssh $SSH_OPTS $VM "
    ~/miniconda3/envs/alpaca/bin/pip install --ignore-installed \
      websockets alpaca-py alpaca-trade-api python-dotenv \
      flask flask-cors pandas pytz tzdata holidays
  "
  ```

- [ ] Install the `python-holidays` submodule (replaces pip version to match host)

  ```bash
  ssh $SSH_OPTS $VM "~/miniconda3/envs/alpaca/bin/pip install ~/python-holidays/"
  ```

- [ ] Create the logs directory required by `momentum_alerts`

  ```bash
  ssh $SSH_OPTS $VM "mkdir -p ~/logs"
  ```

- [ ] Install services to run on VM startup

  Fix hardcoded host paths, user, and conda references in service files:

  ```bash
  ssh $SSH_OPTS $VM "
    # Fix repo path and user
    sed -i 's|/home/wilsonb/dl/github.com/Z223I/alpaca|/home/ubuntu|g; s|User=wilsonb|User=ubuntu|g' ~/services/*.service

    # Replace miniconda python path with VM conda env python
    sed -i 's|/home/wilsonb/miniconda3/envs/alpaca/bin/python|/home/ubuntu/miniconda3/envs/alpaca/bin/python|g' ~/services/*.service

    # Bypass conda activation lines in the market-data-api shell script
    sed -i 's|source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"|# conda not available via systemd|' ~/services/market_data_api_service.sh
    sed -i 's|conda activate alpaca|# conda not available via systemd|' ~/services/market_data_api_service.sh
  "
  ```

  Add `User=ubuntu` and `PYTHONPATH` to `momentum_alerts` service (not present by default):

  ```bash
  ssh $SSH_OPTS $VM "
    sudo sed -i '/^WorkingDirectory=/a User=ubuntu' /etc/systemd/system/momentum_alerts.service || true
  "
  ```

  Copy service files into systemd and enable them:

  ```bash
  ssh $SSH_OPTS $VM "
    sudo cp ~/services/market-data-api.service \
             ~/services/market_sentinel_trade_stream.service \
             ~/services/momentum_alerts.service \
             /etc/systemd/system/ &&
    sudo systemctl daemon-reload &&
    sudo systemctl enable market-data-api.service market_sentinel_trade_stream.service momentum_alerts.service &&
    sudo systemctl start  market-data-api.service market_sentinel_trade_stream.service momentum_alerts.service
  "
  ```

  Verify all services are running:

  ```bash
  ssh $SSH_OPTS $VM "sudo systemctl status market-data-api.service market_sentinel_trade_stream.service momentum_alerts.service --no-pager"
  ```

- [ ] Configure Apache CGI for `cgi-bin/` scripts

  The default Apache ScriptAlias points `/cgi-bin/` to `/usr/lib/cgi-bin/`.
  Override it to use the project's `cgi-bin/`, enable the CGI module, allow
  `www-data` to traverse the home directory, and fix shebangs:

  ```bash
  ssh $SSH_OPTS $VM "
    # Enable CGI module
    sudo a2enmod cgid

    # Allow www-data to traverse /home/ubuntu (execute only, no read)
    chmod o+x /home/ubuntu

    # Fix shebangs in all cgi-bin scripts to use VM conda python
    find ~/cgi-bin -name '*.py' -exec chmod +x {} \;
    find ~/cgi-bin -name '*.py' -exec sed -i \
      's|#!/home/wilsonb/miniconda3/envs/alpaca/bin/python.*|#!/home/ubuntu/miniconda3/envs/alpaca/bin/python|' {} \;
  "
  ```

  Override the Apache virtual host to redirect `/cgi-bin/` to the project directory:

  ```bash
  ssh $SSH_OPTS $VM "sudo tee /etc/apache2/sites-available/000-default.conf > /dev/null <<'EOF'
  <VirtualHost *:80>
      ServerAdmin webmaster@localhost
      DocumentRoot /var/www/html

      ScriptAlias /cgi-bin/ /home/ubuntu/cgi-bin/
      <Directory /home/ubuntu/cgi-bin>
          Options +ExecCGI
          AddHandler cgi-script .py
          AllowOverride None
          Require all granted
      </Directory>

      <Directory /var/www/html>
          Options +FollowSymLinks
          AllowOverride None
          Require all granted
      </Directory>

      ErrorLog \\\${APACHE_LOG_DIR}/error.log
      CustomLog \\\${APACHE_LOG_DIR}/access.log combined
  </VirtualHost>
  EOF
  sudo systemctl restart apache2"
  ```

  Verify the scanner API responds with JSON:

  ```bash
  curl -s http://<vm-ip>/cgi-bin/api/scanner_api.py | python3 -m json.tool | head -6
  ```

- [ ] Sync `historical_data` to the VM

  Alert JSON files are written continuously by `momentum_alerts` on the host.
  The initial `scp` only copies what exists at that moment — rsync afterwards
  to get current alerts:

  ```bash
  rsync -avz -e "ssh $SSH_OPTS" \
    ./historical_data/ \
    $VM:~/historical_data/
  ```

  Fix ownership if rsync reports permission errors (files written by the VM's
  `momentum_alerts` service will be owned by `ubuntu` already; host files may not be):

  ```bash
  ssh $SSH_OPTS $VM "sudo chown -R ubuntu:ubuntu ~/historical_data/"
  ```

  Re-run this rsync any time you want to push today's alerts to the VM.

- [ ] Test pages load correctly

  ```bash
  for page in "" index.html scanner.html test_live.html test_ws.html ws_test.html; do
    echo "$(curl -s -o /dev/null -w '%{http_code}' http://<vm-ip>/${page})  /${page}"
  done
  ```

- [ ] Take a clean baseline snapshot

  ```bash
  sudo virsh snapshot-create-as apache-web clean-baseline
  ```

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
