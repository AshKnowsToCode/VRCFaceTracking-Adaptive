using System.Collections.ObjectModel;
using System.Reflection;
using CommunityToolkit.Mvvm.ComponentModel;
using Microsoft.Extensions.Logging;
using VRCFaceTracking.Core.Contracts.Services;
using VRCFaceTracking.Core.Params.Data.Mutation;

namespace VRCFaceTracking.Core.Params.Data;

/// <summary>
/// Container of all functions and structures retaining to mutating the incoming Expression Data to be usable for output parameters.
/// </summary>
public partial class UnifiedTrackingMutator : ObservableObject
{
    [ObservableProperty]
    [property: SavedSetting("MutatorEnabled", true)]
    private bool _enabled;

    private readonly ILogger<UnifiedTrackingMutator> _logger;
    private readonly ILocalSettingsService _localSettingsService;
    private readonly object _mutationsLock = new();
    private UnifiedTrackingData _inputBuffer;
    public ObservableCollection<TrackingMutation> _mutations = new();

    // Adaptive-calibration fork: discovery index of each loaded mutation, the last tie-breaker of the run order.
    private readonly Dictionary<TrackingMutation, int> _discoveryIndex = new();

    /// <summary>
    /// Adaptive-calibration fork: module identity for mutations (e.g. per-module calibration priors). Set by
    /// MainStandalone before <see cref="Load"/>; not constructor-injected because
    /// ILibManager -> SendCoordinator -> ParameterSenderService -> this mutator would be a DI cycle.
    /// </summary>
    public ILibManager? LibManager { get; set; }

    public UnifiedTrackingMutator(ILogger<UnifiedTrackingMutator> logger, ILocalSettingsService localSettingsService)
    {
        UnifiedTracking.Mutator = this;
        _logger = logger;
        _localSettingsService = localSettingsService;
            
        Enabled = false;
        _inputBuffer = new UnifiedTrackingData();
    }

    /// <summary>
    /// Takes in the latest base expression Weight data from modules and mutates into the Weight data for output parameters.
    /// </summary>
    /// <returns> Mutated Expression Data. </returns>
    public UnifiedTrackingData MutateData(UnifiedTrackingData input)
    {
        if (!Enabled)
        {
            return input;
        }

        _inputBuffer.CopyPropertiesOf(input);

        lock (_mutationsLock)
        {
            foreach (var mutator in _mutations.Where(m => m.IsActive))
            {
                mutator.MutateData(ref _inputBuffer);
            }

            // Adaptive-calibration fork: second tap on the final output (the session recorder's "out" block).
            foreach (var mutation in _mutations)
            {
                if (mutation.IsActive && mutation is IMutationOutputObserver observer)
                {
                    observer.OnMutationsApplied(_inputBuffer);
                }
            }
        }

        return _inputBuffer;
    }

    public async Task Save()
    {
        _logger.LogDebug("Saving mutation configuration...");
        await _localSettingsService.Save(this);

        // This is cringe but since the collection can change mid-save, we'll just make a quick copy of em
        List<TrackingMutation> mutationsToSave;
        lock (_mutationsLock)
        {
            mutationsToSave = _mutations.ToList();
        }

        foreach (var mutation in mutationsToSave)
        {
            _logger.LogInformation($"Saving {mutation.Name} data.");
            await _localSettingsService.SaveSettingAsync(mutation.Name, mutation, true);
        }

        _logger.LogDebug("Mutation data saved.");
    }

    public void Initialize()
    {
        // Try to load config and propogate data into Unified if they exist.
        _logger.LogDebug("Initializing mutations...");
        lock (_mutationsLock)
        {
            foreach (var mutation in _mutations)
            {
                _logger.LogInformation($"Initializing {mutation.Name}");
                mutation.Initialize(UnifiedTracking.Data);
            }
        }

        _logger.LogDebug("Mutations initialized successfully.");
    }
    
    // Adaptive-calibration fork: settings reads finish in any order, so each mutation is initialised and inserted in
    // run order as it arrives instead of appended (Step order was not guaranteed and a single Initialize() pass at
    // the end of Load could run before the list was complete).
    private async Task CreateMutation(TrackingMutation mutation, int discoveryIndex)
    {
        _logger.LogInformation($"Loading {mutation.Name}");
        try
        {
            Type mutationType = mutation.GetType();

            MethodInfo readSettingMethod = typeof(ILocalSettingsService)
                .GetMethod("ReadSettingAsync")
                .MakeGenericMethod(mutationType);

            var task = (Task)readSettingMethod.Invoke(_localSettingsService, new object[] { mutation.Name, mutation, true });
            await task.ConfigureAwait(false);

            PropertyInfo resultProperty = task.GetType().GetProperty("Result");
            var typedMutation = resultProperty.GetValue(task);

            mutation = (TrackingMutation)typedMutation ?? mutation;
        }
        catch (Exception ex)
        {
            _logger.LogError($"Creating new mutation data. {ex.Message}");
        }

        mutation.Logger = _logger;
        mutation.LocalSettingsService =  _localSettingsService;
        // Adaptive-calibration fork: building a mutation's settings UI walks all of its fields recursively and could
        // throw out of this async chain, which took the whole app down at startup (UPSTREAM_BUGS.md #5).
        try
        {
            mutation.CreateProperties();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Failed to build the settings of {mutation.Name}");
            mutation.Components = new ObservableCollection<IMutationComponent>();
        }

        try
        {
            _logger.LogInformation($"Initializing {mutation.Name}");
            mutation.Initialize(UnifiedTracking.Data);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Failed to initialize {mutation.Name}");
        }

        lock (_mutationsLock) InsertInRunOrder(mutation, discoveryIndex);
    }

    // Adaptive-calibration fork: run order is (Step, Order, discovery index). Caller holds _mutationsLock.
    private void InsertInRunOrder(TrackingMutation mutation, int discoveryIndex)
    {
        _discoveryIndex[mutation] = discoveryIndex;
        var key = RunOrderKey(mutation);
        var index = _mutations.Count;
        while (index > 0 && RunOrderKey(_mutations[index - 1]).CompareTo(key) > 0)
        {
            index--;
        }
        _mutations.Insert(index, mutation);
    }

    private (MutationPriority, int, int) RunOrderKey(TrackingMutation mutation) =>
        (mutation.Step, mutation.Order, _discoveryIndex.GetValueOrDefault(mutation));

    public async void Load()
    {
        // Try to load config and propogate data into Unified if they exist.
        _logger.LogDebug("Loading mutation data...");
        var mutations = TrackingMutation.GetImplementingMutations(true);
        await _localSettingsService.Load(this);

        // Adaptive-calibration fork: CreateMutation initialises each mutation itself (see above).
        await Task.WhenAll(mutations.Select(CreateMutation));

        lock (_mutationsLock)
        {
            _logger.LogInformation("Mutation run order: " + string.Join(" > ", _mutations.Select(m => m.Name)));
        }
        _logger.LogDebug("Mutation data loaded.");
    }
}
