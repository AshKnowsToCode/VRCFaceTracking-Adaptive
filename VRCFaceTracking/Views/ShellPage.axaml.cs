using Avalonia.Controls;
using FluentAvalonia.UI.Controls;
using VRCFaceTracking.Contracts;

namespace VRCFaceTracking.Views;

public partial class ShellPage : UserControl
{
    private readonly MainPage _mainPage = new();
    private readonly OutputPage _outputPage = new();
    private readonly ModuleRegistryPage _moduleRegistryPage = new();
    private readonly MutatorPage _mutatorPage = new();
    private readonly SettingsPage _settingsPage = new();
    private readonly GuidedCalibrationPage _guidedPage = new(); // adaptive-calibration fork

    private Control? _currentPage;

    public ShellPage()
    {
        InitializeComponent();

        PageContainer.Children.Add(_mainPage);
        PageContainer.Children.Add(_outputPage);
        PageContainer.Children.Add(_moduleRegistryPage);
        PageContainer.Children.Add(_mutatorPage);
        PageContainer.Children.Add(_settingsPage);
        PageContainer.Children.Add(_guidedPage);

        foreach (var child in PageContainer.Children)
            child.IsVisible = false;

        // Select the first nav item and show the main page
        NavView.SelectedItem = NavView.MenuItems[0];
        ShowPage(_mainPage);

        // Adaptive-calibration fork, testing aid: VRCFT_START_PAGE=Guided opens the guided calibration page first.
        if (Environment.GetEnvironmentVariable("VRCFT_START_PAGE") == "Guided")
        {
            NavView.SelectedItem = NavView.MenuItems.OfType<NavigationViewItem>().First(i => i.Tag as string == "Guided");
            ShowPage(_guidedPage);
        }
    }

    private void ShowPage(Control page)
    {
        if (_currentPage != null)
            _currentPage.IsVisible = false;

        page.IsVisible = true;
        _currentPage = page;

        // Notify module registry when navigated to so it loads data
        if (page is INotifyNavigated notifyNavigated)
            notifyNavigated.OnNavigatedTo();
    }

    private void OnNavigationSelectionChanged(object? sender, NavigationViewSelectionChangedEventArgs e)
    {
        if (e.IsSettingsSelected)
        {
            ShowPage(_settingsPage);
            return;
        }

        if (e.SelectedItem is NavigationViewItem { Tag: string tag })
        {
            ShowPage(tag switch
            {
                "Main" => (Control)_mainPage,
                "Output" => _outputPage,
                "ModuleRegistry" => _moduleRegistryPage,
                "Mutator" => _mutatorPage,
                "Guided" => _guidedPage,
                _ => _mainPage
            });
        }
    }
}