' Speaks each line of a UTF-16 list file ("output.wav<TAB>text") into a WAV file with the Windows SAPI voice.
' Used by poses.py to build the pose-session audio guide. Runs under Windows Script Host, so it does not depend on
' PowerShell (which can hang at startup on this PC).
'   cscript //nologo tts_sapi.vbs list.txt ["Microsoft Zira Desktop"]
Option Explicit
Dim args, listPath, voiceName, fso, ts, v, tok, line, p, outFile, txt, st
Set args = WScript.Arguments
listPath = args(0)
voiceName = ""
If args.Count > 1 Then voiceName = args(1)

Set fso = CreateObject("Scripting.FileSystemObject")
Set ts = fso.OpenTextFile(listPath, 1, False, -1)
Set v = CreateObject("SAPI.SpVoice")
If voiceName <> "" Then
    For Each tok In v.GetVoices()
        If InStr(tok.GetDescription(), voiceName) > 0 Then Set v.Voice = tok
    Next
End If

Do Until ts.AtEndOfStream
    line = ts.ReadLine
    p = InStr(line, vbTab)
    If p > 0 Then
        outFile = Left(line, p - 1)
        txt = Mid(line, p + 1)
        Set st = CreateObject("SAPI.SpFileStream")
        st.Format.Type = 22   ' SAFT22kHz16BitMono
        st.Open outFile, 3, False
        Set v.AudioOutputStream = st
        v.Speak txt
        st.Close
    End If
Loop
ts.Close
