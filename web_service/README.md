# Dashboard compatibility entrypoint

`python3 before-we-act/web_service/server.py` 仍可使用，但现在会启动工作区根目录下的
`web_service/server.py`，从而避免维护两套不同且可能过时的监测页面。

完整说明见 [`../../web_service/README.md`](../../web_service/README.md)。
