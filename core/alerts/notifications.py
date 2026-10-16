
import subprocess


def show_notification(decision, transcript):
    if decision == "IGNORE":
        return

    title = (
        "Vaani - Important Speech"
        if decision == "FULL_ALERT"
        else "Vaani - Possible Attention Needed"
    )

    # Escape text for PowerShell single-quoted strings.
    title = title.replace("'", "''")
    message = transcript[:180].replace("'", "''")

    script = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$notification = New-Object System.Windows.Forms.NotifyIcon
$notification.Icon = [System.Drawing.SystemIcons]::Information
$notification.Visible = $true
$notification.BalloonTipTitle = '{title}'
$notification.BalloonTipText = '{message}'
$notification.BalloonTipIcon = 'Info'
$notification.ShowBalloonTip(5000)

Start-Sleep -Seconds 6
$notification.Dispose()
"""

    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            script,
        ],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
