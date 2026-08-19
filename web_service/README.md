# Remote dashboard

```bash
BWA_REMOTE_ROOT=/workspace/bwa-baselines python3 web_service/server.py
```

默认监听 `http://127.0.0.1:8088`。通过 `ssh -p 10328 -L 8080:127.0.0.1:8088 root@69.176.92.104` 后访问本机 `http://localhost:8080`。
