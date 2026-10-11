# Upstream bugs to report

Found while building the adaptive-calibration fork. Target: `benaclejames/VRCFaceTracking` issues. Line numbers refer
to upstream `master` at 6432e6a (2026-09-11). None reported yet.

| # | Bug | Severity | In this fork |
|---|-----|----------|--------------|
| 1 | Mutation run order not guaranteed; `Initialize()` races the load | Latent race | Fixed |
| 2 | Data Filter sliders are never saved | Annoyance, every restart | Fixed |
| 3 | Parameter Adjustment "Reset Values" does nothing | Annoyance | Not fixed |
| 4 | Data Filter and Calibration assume 10 Hz updates (a 10 ms / 10 Hz mix-up) | Wrong slider units; feel depends on frame rate | Not fixed |
| 5 | Building a mutation's settings UI walks every field recursively; one throw kills the app at startup | Crash at startup (latent for the built-in mutations) | Guarded |
| 6 | Sandbox process found relative to the working directory | Crash at startup when launched from another folder | Fixed |

## 1. Mutation run order not guaranteed; Initialize() races the load

- **Where:** `VRCFaceTracking.Core/Params/Data/UnifiedTrackingMutator.cs:96-147`.
- **What:** `Load()` calls `CreateMutation()` (`async void`) for each mutation without awaiting it. Each one awaits its
  settings read, which runs on `Task.Run` (`Helpers/Json.cs:8`), so they finish in any order. `_mutations.Add`
  (lines 119, 127) appends in that finish order, which loses the `Step` order that `GetImplementingMutations` sorted.
  `Load()` then calls `Initialize()` (line 147) at once. That only initialises mutations already in the list.
- **Consequence:**
  - Mutations can run in the wrong order, e.g. the Data Filter before the Unified Correctors.
  - A mutation that arrives late never gets `Initialize()`. For Calibration that leaves `calData` null. While
    Calibration is on, `MutateData` then throws every frame, and `SendCoordinator` catches it as "OSC emit failed",
    so no OSC output is sent at all.
  - Not seen failing: the Phase 0 log showed all four mutations initialised, in order. It's a timing race that
    usually wins.
- **Smaller issues in the same code:**
  - If a saved setting deserialises to `null`, line 114 stores `null` and the catch block then throws inside an
    `async void`.
  - The Mutations page appends new items and ignores `NewStartingIndex`
    (`VRCFaceTracking/ViewModels/MutatorViewModel.cs:29`), so its list order is arrival order.
- **Fork fix:** `CreateMutation` returns a `Task`, initialises each mutation as it arrives and inserts it by
  (`Step`, new virtual `TrackingMutation.Order`, discovery index). `Load()` awaits them all and logs
  "Mutation run order: ...". The view model now inserts at `NewStartingIndex`. For upstream, awaiting the loads one
  by one would be the minimal fix.

## 2. Data Filter sliders are never saved

- **Where:** `VRCFaceTracking.Core/Params/Data/Mutation/Filter.cs:15-19`.
- **What:** `minCutoff`, `beta` and `dCutoff` are `public static` fields. Json.NET skips static fields unless they
  are marked `[JsonProperty]`, so the saved setting is only `{"Name","Step","IsSaved","IsActive"}` (checked in the
  fork's `LocalSettings.json`).
- **Consequence:** slider changes work until VRCFT closes, then come back as 1.0 / 0.5 / 0.1 on the next start. Each
  slider move does trigger a save, but the save doesn't include the values. The on/off switch is saved fine.
- **Fork fix:** `[JsonProperty]` on the three fields. Tested: values round-trip, the sliders show the loaded values,
  and settings saved by the old build still load (with the defaults). Upstream might prefer making them instance
  fields instead.
- **Same pattern:** Calibration's `points`, `sDelta` and `cDelta` are static too, but only appear in DEBUG builds.

## 3. Parameter Adjustment "Reset Values" does nothing

- **Where:** `VRCFaceTracking.Core/Params/Data/Mutation/ParameterAdjustment.cs:12-19`.
- **What:** the button's method is a loop with an empty body.
- **Consequence:** the button has no effect, so every range has to be dragged back by hand.
- **Fix note:** reset each range to (0, 1), or (-1, 1) for head, and refresh the page's components. The components
  are built from the field values once, so changing the fields alone won't move the sliders.

## 4. Data Filter and Calibration assume 10 Hz updates

- **Where:**
  - `Filter.cs:20` (`hz = 10`), used for the derivative (line 35) and the smoothing factor.
  - `Calibration.cs:165` (`dT: 100f/1000f`, hard-coded although the parameter exists).
- **What:** both mutations compute as if frames were 0.1 s apart. They never were:
  - When the filter was added (980dcd6, 2024-08-20) and still in Steam 5.4.5 (7add148), the update loop was
    `Task.Delay(10)` with a 1 ms timer resolution (`ParameterSenderService.cs`). That is a frame about every 10-11 ms,
    roughly 90 per second (from the code, not measured).
  - 6.0 (7d6fe87) runs once per module frame instead. Its commit message calls the old 10 ms loop a "10hz update
    loop", the same mix-up as the filter's constant.
  - So this is not a 6.0 regression. It looks like an unintended units mix-up (nothing suggests the weaker smoothing
    was a design choice), but the default slider values were presumably tuned by feel against it, so today's feel is
    the established, tested behaviour.
- **Consequence (calculated, not measured):**
  - A 1 Hz Minimum Cutoff behaves like about 7 Hz at 90 frames/s (5.6 Hz at 72), and the speed term that Beta scales
    is 7-9x too small. The sliders' numbers don't mean what their units say.
  - How much the filter smooths depends on the frame rate. In 6.0 that is the module's rate, so the same settings feel
    different per module or headset, and differ from 5.4.5 if the module sends fewer or more than ~90 frames/s.
  - Calibration's "has it moved enough to sample" threshold (0.015 per frame) isn't tied to time either.
- **To do before reporting:** measure the real Virtual Desktop frame rate (`python tools/replay/vftrec.py info` on
  a recording prints it).
- **Fixing it is not a drop-in change:**
  - **Data Filter:** using the real time between frames is correct, but with today's default sliders it makes
    smoothing much heavier. Simulated at 72 Hz: a 4 Hz speech-rhythm movement keeps 84% of its swing today and 26%
    after the fix, and a sudden jaw open takes 208 ms instead of 42 ms to reach 90%. Ship the fix with new defaults
    that match today's feel, and expect anyone who tuned the sliders to retune.
  - **Calibration:** plugging the real frame time into `dT` makes it worse. The per-frame sampling threshold drops
    from 0.015 to about 0.002, so the 64-sample window fills in about a second of movement and the maxima freeze from
    that. It needs a time-based threshold instead (e.g. compare against the value 100 ms ago).

## 5. Building a mutation's settings UI walks every field recursively; one throw kills the app at startup

- **Where:** `VRCFaceTracking.Core/Params/Data/Mutation/UI/MutationFactory.cs:67-129` (`ProcessFields`), called from
  `TrackingMutation.CreateProperties()` in `UnifiedTrackingMutator.CreateMutation` (line 126, outside its try blocks).
- **What:** `ProcessFields` takes every field of the mutation (`Public | NonPublic | Instance | Static`) and recurses
  into any non-null object a field holds (line 118-124), then into that object's fields and statics, and so on. It
  only needs the `[MutationProperty]` fields and `[MutationButton]` methods.
- **Consequence:**
  - A mutation that holds an ordinary framework object in any field when it loads (found with a static
    `System.Text.Json.JsonSerializerOptions`) sends it deep into framework internals until something throws
    (`NullReferenceException` at line 123 here).
  - The throw leaves `CreateMutation` and the `async void Load()`, so the whole app exits at startup with an
    unhandled exception and no log line saying which mutation.
  - The built-in mutations survive because their fields happen to be simple. Any new or third-party mutation is one
    field away from it; the walk also does needless work on large object graphs.
- **Fork:** `CreateMutation` now catches the throw, logs which mutation failed and gives it an empty settings list
  (the app keeps running). The walk itself is unchanged; the Adaptive Calibration mutation keeps its runtime objects
  out of fields at load (comment in `AdaptiveCalibration.cs`). For upstream: only recurse into fields that carry
  `[MutationProperty]`, or only into types from the mutation's own assembly, and catch per mutation.

## 6. Sandbox process found relative to the working directory

- **Where:** `VRCFaceTracking.Core/Library/UnifiedLibManager.cs:40`.
- **What:** `Path.GetFullPath("VRCFaceTracking.ModuleProcess.exe")` resolves against the current working directory,
  not the app's folder.
- **Consequence:** starting `VRCFaceTracking.exe` from any other folder (a script, a terminal, a shortcut whose
  "Start in" is elsewhere) throws `FileNotFoundException: Failed to find sandbox process` while the main page is
  being created, and the app exits. Seen on 2026-10-11 when launching the fork's build from a terminal.
- **Fork fix:** `Path.Combine(AppContext.BaseDirectory, ...)`.

## Also see

DESIGN.md section 1 covers the stock calibrator's design problems (peaks only rise, noise amplified on rarely used
expressions, the learning counter isn't saved so every start re-opens a 64-sample window). Those are behaviour
issues, not bugs in the sense above, but belong in the same conversation with upstream.
