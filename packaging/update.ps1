# QUANT LAB updater (run through update.bat next to QuantLab.exe).
#
# 1. Asks GitHub for the latest release (the build made after a merge to main).
# 2. Stops if this folder already has that build, or if QuantLab.exe from this folder is running.
# 3. Downloads QuantLab-windows.zip, checks its SHA-256 against the release, unpacks it.
# 4. Replaces only the program files: the _internal folder, QuantLab.exe, build-info.txt and
#    the updater itself. Anything else in this folder and the work folder stay as they are.
#    The previous _internal folder is kept in %TEMP% until the next update.
param(
    [string]$ZipPath = "",      # CI: install from a local zip instead of downloading
    [switch]$NoPause,           # CI: do not wait for Enter at the end
    [switch]$NoLaunch           # CI: do not start the app afterwards
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the progress bar makes downloads very slow in Windows PowerShell
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Repo = "a01030930368-pixel/quant-LAB"
$App = $PSScriptRoot
$Work = Join-Path $env:TEMP "quantlab-update"
$swapping = $false

function Say($text) { Write-Host $text }
function Finish($code) {
    if (-not $NoPause) { Read-Host "`nEnter 키를 누르면 창이 닫힙니다" | Out-Null }
    exit $code
}

try {
    if (-not (Test-Path (Join-Path $App "QuantLab.exe"))) {
        throw "이 파일은 QuantLab.exe가 있는 폴더에서 실행해야 합니다: $App"
    }
    $infoFile = Join-Path $App "build-info.txt"
    $current = if (Test-Path $infoFile) { (Get-Content $infoFile -Encoding UTF8 | Select-Object -First 1).Trim() } else { "(알 수 없음)" }
    Say "QUANT LAB 업데이트"
    Say "지금 버전: $current"

    $running = Get-Process -Name QuantLab -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($App, [StringComparison]::OrdinalIgnoreCase) }
    if ($running) {
        Say "`nQUANT LAB이 실행 중입니다. 앱을 닫은 뒤 update.bat을 다시 실행하세요."
        Finish 2
    }

    if (Test-Path $Work) { Remove-Item $Work -Recurse -Force }
    New-Item -ItemType Directory -Path $Work | Out-Null
    $zip = Join-Path $Work "QuantLab-windows.zip"

    if ($ZipPath) {
        Copy-Item $ZipPath $zip
        $latest = "(로컬 파일)"
    } else {
        $rel = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ "User-Agent" = "QuantLab-updater" }
        $latest = $rel.tag_name
        Say "최신 버전: $latest"
        if ($latest -eq $current) {
            Say "`n이미 최신 버전입니다."
            Finish 0
        }
        $asset = $rel.assets | Where-Object { $_.name -eq "QuantLab-windows.zip" } | Select-Object -First 1
        if (-not $asset) { throw "최신 릴리스에 QuantLab-windows.zip이 없습니다." }
        Say "`n내려받는 중… ($([math]::Round($asset.size / 1MB)) MB)"
        Invoke-WebRequest -UseBasicParsing -Uri $asset.browser_download_url -OutFile $zip
        if ($asset.digest -and $asset.digest.StartsWith("sha256:")) {
            $want = $asset.digest.Substring(7).ToLower()
            $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
            if ($want -ne $got) { throw "내려받은 파일의 SHA-256이 릴리스와 다릅니다. 다시 시도하세요." }
            Say "파일 검증 완료 (SHA-256)"
        }
    }

    Say "압축 푸는 중…"
    $unz = Join-Path $Work "new"
    Expand-Archive -Path $zip -DestinationPath $unz -Force
    $src = Join-Path $unz "QuantLab"
    if (-not (Test-Path (Join-Path $src "QuantLab.exe"))) { throw "압축 파일 안에 QuantLab\QuantLab.exe가 없습니다." }

    Say "프로그램 파일 바꾸는 중…"
    $old = Join-Path $env:TEMP "quantlab-previous"
    if (Test-Path $old) { Remove-Item $old -Recurse -Force }
    New-Item -ItemType Directory -Path $old | Out-Null
    $swapping = $true
    if (Test-Path (Join-Path $App "_internal")) { Move-Item (Join-Path $App "_internal") (Join-Path $old "_internal") }
    Copy-Item (Join-Path $src "_internal") (Join-Path $App "_internal") -Recurse
    foreach ($f in Get-ChildItem $src -File) {
        if ($f.Name -eq "update.bat") {
            # cmd.exe is still reading update.bat; replace it only when its content changed
            $dst = Join-Path $App "update.bat"
            if ((Test-Path $dst) -and ((Get-FileHash $dst).Hash -eq (Get-FileHash $f.FullName).Hash)) { continue }
        }
        Copy-Item $f.FullName (Join-Path $App $f.Name) -Force
    }
    $swapping = $false
    Remove-Item $Work -Recurse -Force

    $now = (Get-Content $infoFile -Encoding UTF8 | Select-Object -First 1).Trim()
    Say "`n업데이트 완료: $now"
    Say "작업 폴더의 자료와 기록은 그대로입니다."
    if (-not $NoLaunch) { Start-Process (Join-Path $App "QuantLab.exe") }
    Finish 0
} catch {
    Say "`n업데이트 실패: $($_.Exception.Message)"
    $prev = Join-Path $env:TEMP "quantlab-previous\_internal"
    if ($swapping -and (Test-Path $prev)) {
        if (Test-Path (Join-Path $App "_internal")) { Remove-Item (Join-Path $App "_internal") -Recurse -Force }
        Move-Item (Join-Path $env:TEMP "quantlab-previous\_internal") (Join-Path $App "_internal")
        Say "이전 버전으로 되돌렸습니다."
    }
    Finish 1
}
