This file is used to describe how to run Suricata in IPS mode.

```
export SC_LOG_OP_FILTER=DPITESTING
export SC_LOG_LEVEL=Debug
iptables -I FORWARD -j NFQUEUE
cd suricata-6.0.3-state-instrumentation
stdbuf -oL ./src/suricata -c ./suricata-ips.yaml -q 0 > /mnt/hgfs/share-folders/suricata-log/console.log
```