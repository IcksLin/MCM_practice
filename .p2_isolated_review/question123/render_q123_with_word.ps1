$ErrorActionPreference = 'Stop'
$inputPath = (Resolve-Path -LiteralPath 'C题\doc\C题问题1-3统合报告_v1.docx').Path
$outputDirectory = Join-Path (Get-Location) 'C题\doc\_qa_q123_render'
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$outputPdf = Join-Path $outputDirectory 'C题问题1-3统合报告_v1.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($inputPath, $false, $true)
    $document.ExportAsFixedFormat($outputPdf, 17)
}
finally {
    if ($null -ne $document) { $document.Close($false) }
    if ($null -ne $word) { try { $word.Quit() } catch { } }
    if ($null -ne $document) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null }
    if ($null -ne $word) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
Write-Output $outputPdf
