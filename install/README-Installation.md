# Install scripts

The supported install instructions are in the repository [README](../README.md#installation).

| Platform | Command |
|----------|---------|
| macOS / Linux | `./install/install.sh` or the `curl` one-liner in the README |
| Windows | `install\install.bat` after cloning, or the PowerShell steps in the README |

Both scripts create a virtualenv under `MGRCLI_HOME` (default `~/.mgrcli`), compile `mgrcli_rust`, and check `mgrcli --version`. They do not need administrator rights.
