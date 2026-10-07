using System.Collections.ObjectModel;
using System.Collections.Specialized;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using DynamicData;
using Microsoft.Extensions.Logging;
using VRCFaceTracking.Core.Params.Data;
using VRCFaceTracking.Core.Params.Data.Mutation;

namespace VRCFaceTracking.ViewModels;

public class MutatorViewModel : ObservableRecipient
{
    public ObservableCollection<TrackingMutation> Mutations { get; } = new();

    public MutatorViewModel(UnifiedTrackingMutator trackingMutator)
    {
        Mutations.AddRange(trackingMutator._mutations);

        trackingMutator._mutations.CollectionChanged += OnSourceCollectionChanged;
    }

    private void OnSourceCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        Dispatcher.UIThread.Post(() =>
        {
            // Adaptive-calibration fork: mirror the source position so the page lists mutations in run order.
            if (e.NewItems != null)
            {
                var index = e.NewStartingIndex;
                foreach (TrackingMutation m in e.NewItems)
                {
                    if (index >= 0 && index <= Mutations.Count)
                        Mutations.Insert(index++, m);
                    else
                        Mutations.Add(m);
                }
            }

            if (e.OldItems != null)
                foreach (TrackingMutation m in e.OldItems)
                    Mutations.Remove(m);
        });
    }
}
