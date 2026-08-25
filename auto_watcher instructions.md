# Auto Watcher Instructions

This project uses a Python script (`auto_watcher.py`) to automatically keep `.ipynb` notebooks and `.py` script files perfectly synchronized.

The script uses a library called `watchdog` to monitor your project directory for file changes. Whenever you save a notebook or a Python script, the watcher detects it and instantly runs the `jupytext --sync` command in the background to update its paired file.

The script itself is cross-platform and identical on macOS and Windows. Only the way you activate the Conda environment differs.

> First time on this machine? Create the Conda environment first — see the Installation section of `README.md`.

## How to Start the Watcher

Whenever you start a new coding session, follow these steps to get the auto-sync running in the background.

### 1. Open a new terminal

In VS Code, go to **Terminal → New Terminal** (shortcut `` Ctrl + ` ``). Use a dedicated terminal, because the watcher occupies it for the whole session.

### 2. Activate the Conda environment

This is **crucial**: it ensures the terminal can find both the `watchdog` package and the `jupytext` command.

**macOS (zsh):**

```bash
conda activate neuralprophet_env
```

**Windows (PowerShell):**

```powershell
conda activate neuralprophet_env
```

*(You should see `(neuralprophet_env)` appear on the left side of your terminal prompt once successful.)*

If `conda activate` is not recognised, Conda has not been initialised for that shell yet. Run `conda init zsh` on macOS or `conda init powershell` on Windows, then close and reopen the terminal. This is a one-time step per machine.

### 3. Run the watcher script

Once the environment is active, start the watcher from the repository root:

```bash
python auto_watcher.py
```

You should see a message saying `Watching for file saves recursively... (Press Ctrl+C to stop)`.

### You're all set!

You can now leave this terminal running in the background while you work. Every time you hit "Save" on a `.ipynb` or `.py` file, you'll see a quick message pop up in this terminal confirming that it has synced the other file.

**To stop the watcher:** Click on the terminal where it's running and press `Ctrl+C`, or simply click the trash can icon to close the terminal pane.

## Notes

- **VS Code shortcut:** instead of typing the command, use **Terminal → Run Task… → Start Jupytext watcher**. The task runs `auto_watcher.py` with the interpreter selected for the workspace, so it behaves the same on macOS and Windows.
- **Google Drive:** the repository lives inside a synced Drive folder. File-system events are delivered normally there on macOS, so the watcher works without modification.
- **`jupytext` not found:** the most common failure. It means the terminal is not inside the `neuralprophet_env` environment. Activate it and restart the watcher.
- **Manual sync:** if you prefer not to run the watcher, sync a single file by hand with `jupytext --sync <file>`.
