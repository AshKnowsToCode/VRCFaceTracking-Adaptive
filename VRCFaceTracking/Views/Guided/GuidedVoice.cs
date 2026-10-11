using System.Collections.Concurrent;
using VRCFaceTracking.Core.Adaptive.Guided;

namespace VRCFaceTracking.Views.Guided;

/// <summary>
/// Speaks the guided calibration's prompts with the Windows voice (SAPI, the same voice as the pose-session audio
/// guide) and plays its sounds: a two-note chime when a pose is done and a short tick, rising in pitch, at each quarter
/// of the ring, so it can be followed in the headset without looking at the window. Windows only; elsewhere it is
/// silent. Everything runs on one background thread (SAPI wants the thread that created it).
/// </summary>
public sealed class GuidedVoice : IDisposable
{
    private readonly BlockingCollection<Action> _work = new();
    private readonly Thread? _thread;
    private dynamic? _voice;

    public GuidedVoice()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        _thread = new Thread(Run) { IsBackground = true, Name = "Guided calibration voice" };
        _thread.SetApartmentState(ApartmentState.STA);
        _thread.Start();
    }

    public bool Enabled { get; set; } = true;

    public void Play(GuidedCue cue)
    {
        if (_thread == null || !Enabled)
        {
            return;
        }

        switch (cue.Kind)
        {
            case GuidedCueKind.Speak when cue.Text != null:
                var text = cue.Text;
                _work.Add(() => Speak(text));
                break;
            case GuidedCueKind.Chime:
                _work.Add(() =>
                {
                    Beep(880, 90);
                    Beep(1320, 140);
                });
                break;
            case GuidedCueKind.Tick:
                var pitch = (int)(500 + 500 * Math.Clamp(cue.Value, 0, 1));
                _work.Add(() => Beep(pitch, 60));
                break;
        }
    }

    private void Run()
    {
        try
        {
            var type = Type.GetTypeFromProgID("SAPI.SpVoice");
            _voice = type != null ? Activator.CreateInstance(type) : null;
        }
        catch (Exception)
        {
            _voice = null; // no voice: the page still shows everything
        }

        foreach (var job in _work.GetConsumingEnumerable())
        {
            try
            {
                job();
            }
            catch (Exception)
            {
                // a sound that fails never stops the calibration
            }
        }
    }

    // async and replacing whatever is still being said, so prompts never queue up behind each other
    private void Speak(string text) => _voice?.Speak(text, 1 | 2);

    private static void Beep(int frequency, int ms)
    {
        if (OperatingSystem.IsWindows())
        {
            Console.Beep(frequency, ms);
        }
    }

    public void Dispose() => _work.CompleteAdding();
}
