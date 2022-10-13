This file is used to describe how to run Snort++ in IPS mode.

```
export PATH=$PATH:/usr/local/snort/bin
iptables -I FORWARD -j NFQUEUE --queue-num=0
cd Snort3/snort3-3.1.31.0
stdbuf -oL snort -c /usr/local/snort/etc/snort/snort.lua -Q > /mnt/hgfs/share-folders/snort-log/console.log
```