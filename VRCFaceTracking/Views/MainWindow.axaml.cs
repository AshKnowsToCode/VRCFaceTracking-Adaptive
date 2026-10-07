using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using VRCFaceTracking.Core.Adaptive.Recording;

namespace VRCFaceTracking.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        // Adaptive-calibration fork: a distinct title, so this build can't be mistaken for the Steam install.
        Title = Core.Utils.DataFolderName;

        try
        {
            var iconPath = Path.Combine(AppContext.BaseDirectory, "Assets", "WindowIcon.ico");
            if (File.Exists(iconPath))
            {
                using var stream = File.OpenRead(iconPath);
                Icon = new WindowIcon(stream);
            }
        }
        catch { }

        // Adaptive-calibration fork: F8 drops a session-recorder marker while this window has focus.
        AddHandler(KeyDownEvent, (_, e) =>
        {
            if (e.Key == Key.F8)
            {
                SessionRecorder.Current?.AddMarker();
            }
        }, RoutingStrategies.Tunnel);
    }
}
