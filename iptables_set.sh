iptables -A INPUT -p tcp --sport 80 -j NFQUEUE --queue-num 1
iptables -A OUTPUT -p tcp --dport 80 -j NFQUEUE --queue-num 2
