param([switch]$ValidateOnly, [switch]$SelfTest)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$XamlPath = Join-Path $PSScriptRoot 'RIDFusion.xaml'

function Resolve-Backend {
    $packaged = Join-Path $ProjectRoot 'backend\RIDFusionBackend.exe'
    if (Test-Path -LiteralPath $packaged) {
        return @{ Executable = $packaged; Prefix = '' }
    }
    $candidates = @(
        (Join-Path $ProjectRoot '.venv\Scripts\python.exe'),
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return @{ Executable = $candidate; Prefix = '-m rid_fusion.desktop_api ' }
        }
    }
    foreach ($name in @('python.exe', 'py.exe')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return @{ Executable = $cmd.Source; Prefix = '-m rid_fusion.desktop_api ' } }
    }
    throw '未找到可用的Python运行环境。请安装Python 3.9+并安装numpy。'
}

function Quote-Arg([string]$Value) { return ([char]34) + $Value.Replace([char]34, ' ') + ([char]34) }

$Backend = Resolve-Backend
[xml]$xaml = Get-Content -LiteralPath $XamlPath -Raw -Encoding UTF8
$reader = New-Object System.Xml.XmlNodeReader $xaml
$Window = [System.Windows.Markup.XamlReader]::Load($reader)

$names = @(
    'TopTitle','ActivityText','NavOverview','NavFusion','NavScenarios','NavImport','NavAnalysis','NavEvidence','NavAdapt','NavTests','NavSettings',
    'PageOverview','PageFusion','PageScenarios','PageImport','PageAnalysis','PageEvidence','PageAdapt','PageTests','PageSettings',
    'InputDrone','InputLat','InputLon','InputAlt','InputDuration','InputSeed','InputProtocols',
    'InputDt','InputSpeed','InputHeading','InputWind','InputRain','InputVisibility',
    'RunFusion','ClearFusion','MetricObs','MetricGroups','MetricStd','FusionDetail','PlotCanvas',
    'PresetSelect','RunPreset','MultiCount','MultiSpacing','MultiAltStep','MultiHeadingStep','RunMulti','ScenarioSummary','ScenarioHelp','MultiCanvas',
    'LocationQuery','SearchLocation','LocationResults','ApplyLocation','UseCurrentLocation','OpenLocationSettings','FetchWeather','LocationStatus','WeatherStatus',
    'ImportPath','BrowseImport','RunImport','ImportSummary','AnomalySummary','ReplaySlider','ReplayInfo','ImportCanvas','OpenSchemaGuide',
    'RunComparison','CompareBest','CompareAverage','CompareCov','ReportFormat','ExportReport','ReportStatus',
    'AdaptSeed','RunAdapt','AdaptBefore','AdaptAfter','AdaptGain','AdaptInfo',
    'RunTests','TestStatus','TestOutput'
)
$ui = @{}
foreach ($name in $names) {
    $ui[$name] = $Window.FindName($name)
    if ($null -eq $ui[$name]) { throw ('界面控件未找到：' + $name) }
}
if ($ValidateOnly) { Write-Output ('XAML_OK controls=' + $names.Count); exit 0 }

$pages = @{
    '总览'='PageOverview'; '融合实验'='PageFusion'; '场景与多目标'='PageScenarios';
    '数据导入与回放'='PageImport'; '对比与报告'='PageAnalysis'; '证据与协议'='PageEvidence';
    '少样本适配'='PageAdapt'; '验证中心'='PageTests'; '界面设置'='PageSettings'
}
$navs = @{
    '总览'='NavOverview'; '融合实验'='NavFusion'; '场景与多目标'='NavScenarios';
    '数据导入与回放'='NavImport'; '对比与报告'='NavAnalysis'; '证据与协议'='NavEvidence';
    '少样本适配'='NavAdapt'; '验证中心'='NavTests'; '界面设置'='NavSettings'
}
function Show-Page([string]$Title) {
    foreach ($entry in $pages.GetEnumerator()) { $ui[$entry.Value].Visibility = if ($entry.Key -eq $Title) {'Visible'} else {'Collapsed'} }
    foreach ($entry in $navs.GetEnumerator()) {
        $ui[$entry.Value].Background = if ($entry.Key -eq $Title) {'#ECECF1'} else {'Transparent'}
        $ui[$entry.Value].Foreground = if ($entry.Key -eq $Title) {'#202123'} else {'#6B7280'}
    }
    $ui.TopTitle.Text = $Title
}
foreach ($title in $navs.Keys) { $localTitle=$title; $ui[$navs[$title]].Add_Click({ Show-Page $localTitle }.GetNewClosure()) }

function Invoke-Api([string]$Arguments) {
    $ui.ActivityText.Text = '正在处理…'
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Backend.Executable
    $psi.Arguments = $Backend.Prefix + $Arguments
    $psi.WorkingDirectory = $ProjectRoot
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $psi.StandardOutputEncoding = [Text.Encoding]::UTF8
    $process = [Diagnostics.Process]::Start($psi)
    $stdout = $process.StandardOutput.ReadToEnd(); $stderr = $process.StandardError.ReadToEnd(); $process.WaitForExit()
    $ui.ActivityText.Text = '就绪'
    if ($process.ExitCode -ne 0) { throw ($stderr + "`n" + $stdout) }
    return $stdout | ConvertFrom-Json
}

function Test-FusionInputs {
    $controls=@($ui.InputDrone,$ui.InputLat,$ui.InputLon,$ui.InputAlt,$ui.InputDuration,$ui.InputSeed,$ui.InputDt,$ui.InputSpeed,$ui.InputHeading,$ui.InputWind,$ui.InputRain,$ui.InputVisibility)
    foreach($control in $controls){$control.BorderBrush='#D9D9E3'}
    $errors=New-Object System.Collections.Generic.List[string]
    if([string]::IsNullOrWhiteSpace($ui.InputDrone.Text) -or $ui.InputDrone.Text.Trim().Length -gt 64){
        $errors.Add('目标ID必须为1～64个字符，且不能留空。');$ui.InputDrone.BorderBrush='#EF6A6A'
    }
    $specs=@(
        @($ui.InputLat,'纬度',-90.0,90.0),@($ui.InputLon,'经度',-180.0,180.0),
        @($ui.InputAlt,'高度',-500.0,10000.0),@($ui.InputDuration,'时长',1.0,3600.0),
        @($ui.InputDt,'采样间隔',0.1,60.0),@($ui.InputSpeed,'速度',0.0,150.0),
        @($ui.InputHeading,'航向角',0.0,360.0),@($ui.InputWind,'风速',0.0,100.0),
        @($ui.InputRain,'降水',0.0,500.0),@($ui.InputVisibility,'能见度',1.0,100000.0)
    )
    foreach($spec in $specs){
        $value=0.0
        $ok=[double]::TryParse($spec[0].Text,[Globalization.NumberStyles]::Float,[Globalization.CultureInfo]::InvariantCulture,[ref]$value)
        if(-not $ok -or [double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt $spec[2] -or $value -gt $spec[3]){
            $errors.Add(($spec[1]+'必须是'+$spec[2]+'～'+$spec[3]+'之间的数字。'));$spec[0].BorderBrush='#EF6A6A'
        }
    }
    $seed=[uint64]0
    if(-not [uint64]::TryParse($ui.InputSeed.Text,[ref]$seed) -or $seed -gt 4294967295){
        $errors.Add('实验编号必须是0～4,294,967,295之间的整数。');$ui.InputSeed.BorderBrush='#EF6A6A'
    }
    if($errors.Count -gt 0){[System.Windows.MessageBox]::Show(($errors -join "`n"),'请检查输入','OK','Warning')|Out-Null;return $false}
    return $true
}

function Get-CommonArguments {
    $selected=$ui.InputProtocols.SelectedItem; $protocol=$selected.Tag
    return '--drone '+(Quote-Arg $ui.InputDrone.Text)+' --lat '+$ui.InputLat.Text+' --lon '+$ui.InputLon.Text+
        ' --alt '+$ui.InputAlt.Text+' --duration '+$ui.InputDuration.Text+' --seed '+$ui.InputSeed.Text+
        ' --protocols '+$protocol+' --dt '+$ui.InputDt.Text+' --speed '+$ui.InputSpeed.Text+
        ' --heading '+$ui.InputHeading.Text+' --wind '+$ui.InputWind.Text+' --precipitation '+$ui.InputRain.Text+
        ' --visibility '+$ui.InputVisibility.Text
}

function Set-ComboTag($Combo, [string]$Tag) {
    foreach($item in $Combo.Items){if($item -is [System.Windows.Controls.ComboBoxItem] -and [string]$item.Tag -eq $Tag){$Combo.SelectedItem=$item;return}}
    throw ('下拉选项不存在：'+$Tag)
}

function Set-ScenarioParameters($Config) {
    $ui.InputLat.Text=[string]$Config.lat;$ui.InputLon.Text=[string]$Config.lon;$ui.InputAlt.Text=[string]$Config.alt
    $ui.InputDuration.Text=[string]$Config.duration;$ui.InputSeed.Text=[string]$Config.seed;$ui.InputDt.Text=[string]$Config.dt
    $ui.InputSpeed.Text=[string]$Config.speed;$ui.InputHeading.Text=[string]$Config.heading;$ui.InputWind.Text=[string]$Config.wind
    $ui.InputRain.Text=[string]$Config.rain;$ui.InputVisibility.Text=[string]$Config.visibility
    $ui.MultiCount.Text=[string]$Config.count;$ui.MultiSpacing.Text=[string]$Config.spacing
    $ui.MultiAltStep.Text=[string]$Config.altStep;$ui.MultiHeadingStep.Text=[string]$Config.headingStep
    Set-ComboTag $ui.InputProtocols $Config.protocols
}

function Invoke-MultiExperiment([string]$Label='自定义多目标') {
    if(-not (Test-FusionInputs)){return}
    $args='multi '+(Get-CommonArguments)+' --count '+$ui.MultiCount.Text+' --spacing-m '+$ui.MultiSpacing.Text+' --altitude-step-m '+$ui.MultiAltStep.Text+' --heading-step-deg '+$ui.MultiHeadingStep.Text
    $d=(Invoke-Api $args).data
    $ui.ScenarioSummary.Text='场景：'+$Label+' | 目标：'+$d.stats.target_count+' | 观测：'+$d.stats.received_observations+' | 关联组：'+$d.stats.associated_groups+' | 融合状态：'+$d.stats.fused_states+' | 异常：'+@($d.anomalies).Count
    if(@($d.anomalies).Count -gt 0){$ui.ScenarioHelp.Text='已检测异常：'+((@($d.anomalies)|Select-Object -First 5|ForEach-Object{$_.message}) -join '；')}else{$ui.ScenarioHelp.Text='未检测到已定义规则覆盖的异常。不同目标按目标ID分别关联，图中颜色区分目标。'}
    Draw-Plot $d $ui.MultiCanvas
}

$script:LocationResultsData=@()
$script:SelectedLocation=$null
function Set-LocalLocation($Location, [string]$Source) {
    $script:SelectedLocation=$Location
    $ui.InputLat.Text=([string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.######}',[double]$Location.latitude))
    $ui.InputLon.Text=([string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.######}',[double]$Location.longitude))
    if($null -ne $Location.elevation -and [double]$Location.elevation -ge -500 -and [double]$Location.elevation -le 9880){$ui.InputAlt.Text=([string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.#}',([double]$Location.elevation+120)))}
    $place=@($Location.name,$Location.admin1,$Location.country)|Where-Object{-not [string]::IsNullOrWhiteSpace([string]$_)}
    $elevationText=if($null -ne $Location.elevation){' | 地面海拔约'+([math]::Round([double]$Location.elevation,1))+'m'}else{''}
    $ui.LocationStatus.Text='已应用：'+($place -join '，')+' | '+$ui.InputLat.Text+', '+$ui.InputLon.Text+$elevationText+' | 来源：'+$Source
    $ui.ScenarioSummary.Text='当地场景中心已更新。可以获取天气后运行自定义多目标，或切换到融合实验检查参数。'
}

function Await-WinRT($AsyncOperation, [Type]$ResultType) {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $method=([System.WindowsRuntimeSystemExtensions].GetMethods()|Where-Object{$_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1}|Select-Object -First 1)
    if($null -eq $method){throw '当前Windows环境无法转换定位异步任务'}
    $task=$method.MakeGenericMethod($ResultType).Invoke($null,@($AsyncOperation));$task.Wait();return $task.Result
}

function Draw-Plot($Data, $TargetCanvas=$null) {
    $canvas=if($null -ne $TargetCanvas){$TargetCanvas}else{$ui.PlotCanvas}; $canvas.Children.Clear()
    $w=[Math]::Max($canvas.ActualWidth,500); $h=[Math]::Max($canvas.ActualHeight,260)
    $all=@($Data.trajectory); if ($all.Count -lt 2) { return }
    $isImported=([string]$Data.plot_kind -eq 'imported')
    $minLon=($all|Measure-Object lon -Minimum).Minimum; $maxLon=($all|Measure-Object lon -Maximum).Maximum
    $minLat=($all|Measure-Object lat -Minimum).Minimum; $maxLat=($all|Measure-Object lat -Maximum).Maximum
    $legendWidth=215;$plotSpan=[Math]::Max($w-$legendWidth-75,120)
    function PointXY($lon,$lat) {
        $x=35+(($lon-$minLon)/[Math]::Max($maxLon-$minLon,1e-9))*$plotSpan
        $y=$h-30-(($lat-$minLat)/[Math]::Max($maxLat-$minLat,1e-9))*($h-55)
        return [System.Windows.Point]::new($x,$y)
    }
    $colors=@('#0F9D7A','#2563EB','#EA580C','#9333EA','#DC2626','#0891B2','#65A30D','#DB2777','#4F46E5','#CA8A04','#0D9488','#7C3AED','#E11D48','#0284C7','#16A34A','#C2410C','#6D28D9','#BE123C','#0369A1','#3F6212')
    $truthGroups=@($all | Group-Object { if($_.drone_id){$_.drone_id}elseif($_.track_key){$_.track_key}else{'target'} })
    $colorIndex=0;$legendEntries=@()
    foreach($group in $truthGroups){
        $color=$colors[$colorIndex % $colors.Count]; $colorIndex++
        $line=New-Object System.Windows.Shapes.Polyline; $line.Stroke='#7A7A7A'; $line.StrokeThickness=1.5; $line.StrokeDashArray='4,3'
        foreach($p in $group.Group){[void]$line.Points.Add((PointXY $p.lon $p.lat))}; [void]$canvas.Children.Add($line)
        $track=$group.Name
        $legendEntries+=[PSCustomObject]@{track=$track;color=$color}
        $start=$group.Group[0];$startPoint=PointXY $start.lon $start.lat
        $trackLabel=New-Object System.Windows.Controls.TextBlock;$trackLabel.Text=$track;$trackLabel.Foreground=$color;$trackLabel.Background='#EFFFFFFF';$trackLabel.FontSize=10;$trackLabel.FontWeight='SemiBold';$trackLabel.Padding='3,1'
        [System.Windows.Controls.Canvas]::SetLeft($trackLabel,$startPoint.X+5);[System.Windows.Controls.Canvas]::SetTop($trackLabel,$startPoint.Y-18);[System.Windows.Controls.Panel]::SetZIndex($trackLabel,20);[void]$canvas.Children.Add($trackLabel)
        foreach($s in @($Data.states | Where-Object { (-not $_.track_key) -or $_.track_key -eq $track })){
            if($null -eq $s.lat -or $null -eq $s.lon){continue}; $pt=PointXY $s.lon $s.lat
            $dot=New-Object System.Windows.Shapes.Ellipse; $dot.Width=7;$dot.Height=7;$dot.Fill=$color
            [System.Windows.Controls.Canvas]::SetLeft($dot,$pt.X-3.5);[System.Windows.Controls.Canvas]::SetTop($dot,$pt.Y-3.5);[void]$canvas.Children.Add($dot)
        }
    }
    $legendPanel=New-Object System.Windows.Controls.StackPanel
    $legendTitle=New-Object System.Windows.Controls.TextBlock;$legendTitle.Text='图例';$legendTitle.FontSize=14;$legendTitle.FontWeight='SemiBold';$legendTitle.Foreground='#202123';$legendTitle.Margin='0,0,0,6';[void]$legendPanel.Children.Add($legendTitle)
    $truthRow=New-Object System.Windows.Controls.TextBlock;$truthRow.Text=if($isImported){'┄┄  灰色虚线：导入状态连线（无参考真值）'}else{'┄┄  灰色虚线：模拟参考轨迹'};$truthRow.Foreground='#5F6368';$truthRow.FontSize=11;$truthRow.TextWrapping='Wrap';$truthRow.Margin='0,0,0,5';[void]$legendPanel.Children.Add($truthRow)
    foreach($entry in $legendEntries){
        $row=New-Object System.Windows.Controls.TextBlock;$row.FontSize=11;$row.TextWrapping='Wrap';$row.Margin='0,3,0,0'
        $marker=New-Object System.Windows.Documents.Run;$marker.Text='●  ';$marker.Foreground=$entry.color
        $label=New-Object System.Windows.Documents.Run;$label.Text=([string]$entry.track+'：融合位置点');$label.Foreground='#202123'
        [void]$row.Inlines.Add($marker);[void]$row.Inlines.Add($label);[void]$legendPanel.Children.Add($row)
    }
    $scroll=New-Object System.Windows.Controls.ScrollViewer;$scroll.VerticalScrollBarVisibility='Auto';$scroll.HorizontalScrollBarVisibility='Disabled';$scroll.Content=$legendPanel
    $legend=New-Object System.Windows.Controls.Border;$legend.Width=$legendWidth;$legend.Height=[Math]::Max(80,[Math]::Min($h-20,62+$legendEntries.Count*24));$legend.Background='#F9FAFB';$legend.BorderBrush='#D1D5DB';$legend.BorderThickness=1;$legend.CornerRadius=8;$legend.Padding='12,10';$legend.Child=$scroll
    [System.Windows.Controls.Canvas]::SetLeft($legend,$w-$legendWidth-8);[System.Windows.Controls.Canvas]::SetTop($legend,10);[System.Windows.Controls.Panel]::SetZIndex($legend,100);[void]$canvas.Children.Add($legend)
}

$ui.RunFusion.Add_Click({
    try {
        if(-not (Test-FusionInputs)){return}
        $args='fusion '+(Get-CommonArguments)
        $response=Invoke-Api $args; $d=$response.data
        $ui.MetricObs.Text=[string]$d.stats.received_observations; $ui.MetricGroups.Text=[string]$d.stats.associated_groups
        $ui.MetricStd.Text=if($null -ne $d.average_horizontal_std_m){'{0:N2} m' -f $d.average_horizontal_std_m}else{'N/A'}
        $weights=@(); foreach($p in $d.latest_protocol_weights.PSObject.Properties){$weights+=($p.Name+': '+('{0:P0}' -f $p.Value))}
        $ui.FusionDetail.Text='最新协议贡献：'+($weights -join ' · ')+'  |  证据观测：'+$d.latest_evidence_count+'条'
        Draw-Plot $d
    } catch { [System.Windows.MessageBox]::Show($_.Exception.Message,'运行失败','OK','Error') | Out-Null }
})
$ui.ClearFusion.Add_Click({ $ui.MetricObs.Text='—';$ui.MetricGroups.Text='—';$ui.MetricStd.Text='—';$ui.FusionDetail.Text='运行后显示协议贡献和证据状态';$ui.PlotCanvas.Children.Clear() })
$ui.RunPreset.Add_Click({
    try{
        $key=$ui.PresetSelect.SelectedItem.Tag
        if($key -in @('chengdu_basic','shenzhen_fast','crossing_targets','single_source','poor_visibility')){
            $d=(Invoke-Api ('presets --key '+$key)).data
            $ui.ScenarioSummary.Text='模板：'+$d.scenario.name+' | 目标：'+$d.stats.target_count+' | 观测：'+$d.stats.received_observations+' | 融合状态：'+$d.stats.fused_states
            $ui.ScenarioHelp.Text='协议组合：'+$d.scenario.protocols+'；内置模板使用固定参数和随机种子，因此可以复现实验。'
            Draw-Plot $d $ui.MultiCanvas
        }else{
            $configs=@{
                urban_dense=@{lat='39.9042';lon='116.4074';alt='120';duration='60';seed='101';dt='1';speed='12';heading='45';wind='3';rain='0';visibility='8000';protocols='all';count='8';spacing='50';altStep='3';headingStep='45'}
                suburban_open=@{lat='30.6500';lon='104.0000';alt='100';duration='90';seed='202';dt='1';speed='10';heading='90';wind='2';rain='0';visibility='20000';protocols='wifi_ble_nr';count='3';spacing='200';altStep='5';headingStep='0'}
                mountain_fog=@{lat='29.5630';lon='106.5516';alt='620';duration='60';seed='303';dt='1';speed='7';heading='135';wind='6';rain='2';visibility='500';protocols='all';count='4';spacing='100';altStep='12';headingStep='35'}
                coastal_wind=@{lat='36.0671';lon='120.3826';alt='100';duration='60';seed='404';dt='1';speed='14';heading='30';wind='18';rain='0';visibility='12000';protocols='all';count='5';spacing='120';altStep='4';headingStep='55'}
                heavy_rain=@{lat='23.1291';lon='113.2644';alt='120';duration='45';seed='505';dt='1';speed='8';heading='210';wind='9';rain='50';visibility='1200';protocols='wifi_ble_nr';count='3';spacing='80';altStep='5';headingStep='40'}
                single_source_stress=@{lat='30.5728';lon='104.0668';alt='120';duration='60';seed='606';dt='1';speed='9';heading='0';wind='0';rain='0';visibility='10000';protocols='wifi';count='10';spacing='60';altStep='2';headingStep='36'}
                close_crossing=@{lat='31.2304';lon='121.4737';alt='110';duration='45';seed='707';dt='1';speed='11';heading='0';wind='4';rain='0';visibility='9000';protocols='all';count='6';spacing='20';altStep='2';headingStep='60'}
            }
            if(-not $configs.ContainsKey([string]$key)){throw '未知扩展模板'}
            Set-ScenarioParameters $configs[[string]$key]
            Invoke-MultiExperiment ([string]$ui.PresetSelect.SelectedItem.Content)
        }
    }catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'模板运行失败','OK','Error')|Out-Null}
})
$ui.RunMulti.Add_Click({
    try{Invoke-MultiExperiment '自定义多目标'}catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'多目标运行失败','OK','Error')|Out-Null}
})

$ui.SearchLocation.Add_Click({
    try{
        $query=$ui.LocationQuery.Text.Trim();if($query.Length -lt 2){throw '请输入至少2个字符的城市或地区名称。'}
        $ui.ActivityText.Text='正在搜索地点…';[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
        $uri='https://geocoding-api.open-meteo.com/v1/search?name='+[Uri]::EscapeDataString($query)+'&count=10&language=zh&format=json'
        $response=Invoke-RestMethod -Uri $uri -Headers @{'User-Agent'='RID-Fusion-Studio/0.4.0'} -TimeoutSec 15
        $ui.LocationResults.Items.Clear();$script:LocationResultsData=@($response.results)
        for($index=0;$index -lt $script:LocationResultsData.Count;$index++){
            $place=$script:LocationResultsData[$index];$parts=@($place.name,$place.admin1,$place.country)|Where-Object{-not [string]::IsNullOrWhiteSpace([string]$_)}
            $item=New-Object System.Windows.Controls.ComboBoxItem;$item.Content=($parts -join '，')+'  ('+$place.latitude+', '+$place.longitude+')';$item.Tag=$index;$item.Foreground='#202123';[void]$ui.LocationResults.Items.Add($item)
        }
        if($ui.LocationResults.Items.Count -gt 0){$ui.LocationResults.SelectedIndex=0;$ui.LocationStatus.Text='找到'+$ui.LocationResults.Items.Count+'个结果，请选择后点击“应用所选地点”。'}else{$ui.LocationStatus.Text='没有找到匹配地点，请尝试城市全名或拼音。'}
    }catch{$ui.LocationStatus.Text='地点搜索失败：'+$_.Exception.Message}finally{$ui.ActivityText.Text='就绪'}
})
$ui.ApplyLocation.Add_Click({
    try{if($null -eq $ui.LocationResults.SelectedItem){throw '请先搜索并选择地点。'};$index=[int]$ui.LocationResults.SelectedItem.Tag;Set-LocalLocation $script:LocationResultsData[$index] 'Open-Meteo / GeoNames'}catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'无法应用地点','OK','Warning')|Out-Null}
})
$ui.UseCurrentLocation.Add_Click({
    try{
        $answer=[System.Windows.MessageBox]::Show('将调用Windows定位服务。系统可能请求位置权限；坐标只用于当前实验，点击“获取天气”前不会发送给第三方服务。是否继续？','使用当前位置','YesNo','Question');if($answer -ne 'Yes'){return}
        $ui.ActivityText.Text='正在请求Windows定位…'
        [Windows.Devices.Geolocation.Geolocator,Windows.Devices.Geolocation,ContentType=WindowsRuntime]|Out-Null
        $access=Await-WinRT ([Windows.Devices.Geolocation.Geolocator]::RequestAccessAsync()) ([Windows.Devices.Geolocation.GeolocationAccessStatus])
        if([string]$access -ne 'Allowed'){throw 'Windows未允许位置访问，请打开定位设置后重试。'}
        $locator=New-Object Windows.Devices.Geolocation.Geolocator;$locator.DesiredAccuracyInMeters=100
        $position=Await-WinRT ($locator.GetGeopositionAsync()) ([Windows.Devices.Geolocation.Geoposition])
        $basic=$position.Coordinate.Point.Position;$accuracy=$position.Coordinate.Accuracy
        $location=[PSCustomObject]@{name='Windows当前位置';admin1='';country='';latitude=$basic.Latitude;longitude=$basic.Longitude;elevation=$basic.Altitude}
        Set-LocalLocation $location ('Windows定位，精度约'+[math]::Round([double]$accuracy,0)+'m')
    }catch{$ui.LocationStatus.Text='当前位置获取失败：'+$_.Exception.Message}finally{$ui.ActivityText.Text='就绪'}
})
$ui.OpenLocationSettings.Add_Click({try{Start-Process 'ms-settings:privacy-location'}catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'无法打开定位设置','OK','Error')|Out-Null}})
$ui.FetchWeather.Add_Click({
    try{
        $lat=0.0;$lon=0.0
        if(-not [double]::TryParse($ui.InputLat.Text,[Globalization.NumberStyles]::Float,[Globalization.CultureInfo]::InvariantCulture,[ref]$lat) -or -not [double]::TryParse($ui.InputLon.Text,[Globalization.NumberStyles]::Float,[Globalization.CultureInfo]::InvariantCulture,[ref]$lon)){throw '当前经纬度无效，请先搜索地点、获取当前位置或手动填写。'}
        $answer=[System.Windows.MessageBox]::Show('将把当前经纬度发送给Open-Meteo以获取天气，并自动填写风速、降水和能见度。是否继续？','获取当地天气','YesNo','Question');if($answer -ne 'Yes'){return}
        $ui.ActivityText.Text='正在获取当地天气…';[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
        $latText=[string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.######}',$lat);$lonText=[string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.######}',$lon)
        $uri='https://api.open-meteo.com/v1/forecast?latitude='+$latText+'&longitude='+$lonText+'&current=temperature_2m,precipitation,wind_speed_10m,visibility,weather_code&wind_speed_unit=ms&timezone=auto'
        $weather=Invoke-RestMethod -Uri $uri -Headers @{'User-Agent'='RID-Fusion-Studio/0.4.0'} -TimeoutSec 15;$current=$weather.current
        $ui.InputWind.Text=[string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.##}',[double]$current.wind_speed_10m)
        $ui.InputRain.Text=[string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0.##}',[double]$current.precipitation)
        $ui.InputVisibility.Text=[string]::Format([Globalization.CultureInfo]::InvariantCulture,'{0:0}',[double]$current.visibility)
        $ui.WeatherStatus.Text='已回填：温度'+$current.temperature_2m+'°C，风速'+$ui.InputWind.Text+'m/s，降水'+$ui.InputRain.Text+'mm/h，能见度'+$ui.InputVisibility.Text+'m | '+$current.time+' | Open-Meteo'
    }catch{$ui.WeatherStatus.Text='天气获取失败：'+$_.Exception.Message}finally{$ui.ActivityText.Text='就绪'}
})

$script:ReplayFrames=@()
$ui.BrowseImport.Add_Click({
    $dialog=New-Object Microsoft.Win32.OpenFileDialog
    $dialog.Filter='RID观测文件 (*.csv;*.json;*.jsonl)|*.csv;*.json;*.jsonl|所有文件 (*.*)|*.*'
    if($dialog.ShowDialog()){$ui.ImportPath.Text=$dialog.FileName}
})
$ui.RunImport.Add_Click({
    try{
        if([string]::IsNullOrWhiteSpace($ui.ImportPath.Text)){throw '请先选择CSV、JSON或JSONL观测文件。'}
        $d=(Invoke-Api ('import --path '+(Quote-Arg $ui.ImportPath.Text)+' --bucket 1')).data
        $imp=$d.import; $ui.ImportSummary.Text='读取行：'+$imp.total_rows+' | 有效观测：'+$imp.accepted_rows+' | 无效行：'+$imp.rejected_rows+' | 融合状态：'+$d.state_count+' | 回放帧：'+$d.replay.frame_count
        $ui.AnomalySummary.Text=if(@($d.anomalies).Count -gt 0){'异常：'+((@($d.anomalies)|Select-Object -First 5|ForEach-Object{$_.message}) -join '；')}else{'未检测到已定义规则覆盖的异常。'}
        $script:ReplayFrames=@($d.frames); $ui.ReplaySlider.Maximum=[Math]::Max(0,$script:ReplayFrames.Count-1);$ui.ReplaySlider.Value=0
        if($script:ReplayFrames.Count -gt 0){$f=$script:ReplayFrames[0];$ui.ReplayInfo.Text='第1/'+$script:ReplayFrames.Count+'帧 | 时间：'+$f.timestamp+' | 观测：'+$f.observation_count+' | 目标：'+($f.targets -join ', ')+' | 协议：'+($f.protocols -join ', ')}
        $plot=[PSCustomObject]@{trajectory=@($d.states);states=@($d.states);plot_kind='imported'}; Draw-Plot $plot $ui.ImportCanvas
    }catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'导入失败','OK','Error')|Out-Null}
})
$ui.ReplaySlider.Add_ValueChanged({
    if($script:ReplayFrames.Count -eq 0){return};$index=[int][Math]::Round($ui.ReplaySlider.Value);$f=$script:ReplayFrames[$index]
    $ui.ReplayInfo.Text='第'+($index+1)+'/'+$script:ReplayFrames.Count+'帧 | 时间：'+$f.timestamp+' | 观测：'+$f.observation_count+' | 目标：'+($f.targets -join ', ')+' | 协议：'+($f.protocols -join ', ')
})
$ui.OpenSchemaGuide.Add_Click({
    try{$guide=Join-Path $ProjectRoot 'sample_data\数据格式说明.txt';if(-not (Test-Path -LiteralPath $guide)){throw '数据格式说明文件不存在'};Start-Process notepad.exe -ArgumentList (Quote-Arg $guide)}catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'无法打开说明','OK','Error')|Out-Null}
})

$ui.RunComparison.Add_Click({
    try{
        if(-not (Test-FusionInputs)){return};$d=(Invoke-Api ('compare '+(Get-CommonArguments))).data.comparison
        $ui.CompareBest.Text='{0:N2} m' -f $d.best_single_source.horizontal_rmse_m
        $ui.CompareAverage.Text='{0:N2} m' -f $d.simple_average.horizontal_rmse_m
        $ui.CompareCov.Text='{0:N2} m' -f $d.covariance_weighted.horizontal_rmse_m
    }catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'对比失败','OK','Error')|Out-Null}
})
$ui.ExportReport.Add_Click({
    try{
        if(-not (Test-FusionInputs)){return};$format=$ui.ReportFormat.SelectedItem.Tag
        $dialog=New-Object Microsoft.Win32.SaveFileDialog
        if($format -eq 'md'){$dialog.Filter='Markdown报告 (*.md)|*.md';$dialog.DefaultExt='.md'}elseif($format -eq 'json'){$dialog.Filter='JSON证据包 (*.json)|*.json';$dialog.DefaultExt='.json'}else{$dialog.Filter='CSV状态表 (*.csv)|*.csv';$dialog.DefaultExt='.csv'}
        $dialog.FileName='RID_Fusion_Report'
        if(-not $dialog.ShowDialog()){return}
        $d=(Invoke-Api ('export '+(Get-CommonArguments)+' --format '+$format+' --output '+(Quote-Arg $dialog.FileName))).data
        $ui.ReportStatus.Text='已导出：'+$d.path
    }catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'导出失败','OK','Error')|Out-Null}
})
$ui.RunAdapt.Add_Click({
    try{$r=(Invoke-Api ('adaptation --seed '+$ui.AdaptSeed.Text)).data;$ui.AdaptBefore.Text='{0:N4}' -f $r.unadapted_query_loss;$ui.AdaptAfter.Text='{0:N4}' -f $r.adapted_loss;$ui.AdaptGain.Text='{0:N2}×' -f $r.improvement_factor;$ui.AdaptInfo.Text='评价目标：'+$r.evaluation_target+'；适配样本：'+$r.n_adaptation_shots+'。结果仅代表当前合成实验。'}catch{[System.Windows.MessageBox]::Show($_.Exception.Message,'评估失败','OK','Error')|Out-Null}
})
$ui.RunTests.Add_Click({
    try{$ui.TestStatus.Text='正在验证…';$r=(Invoke-Api 'selftest').data;$ui.TestOutput.Text=$r.output;$ui.TestStatus.Text=if($r.ok){'全部通过 · '+$r.tests+'项'}else{'存在失败'}}catch{$ui.TestStatus.Text='运行失败';$ui.TestOutput.Text=$_.Exception.Message}
})

if($SelfTest){
    $testData=(Invoke-Api 'fusion --drone UI-SELFTEST --lat 30.5728 --lon 104.0668 --alt 120 --duration 2 --seed 123 --protocols wifi_ble_nr').data
    Draw-Plot $testData
    if($ui.PlotCanvas.Children.Count -lt 2){throw '轨迹绘制自检未生成图形元素'}
    $presetCount=@($ui.PresetSelect.Items|Where-Object{$_ -is [System.Windows.Controls.ComboBoxItem]}).Count
    if($presetCount -lt 12){throw '第一阶段场景模板数量不足'}
    $multiTest=(Invoke-Api 'multi --drone STAGE1 --lat 30.5728 --lon 104.0668 --alt 120 --duration 2 --seed 321 --protocols all --count 3 --spacing-m 50 --altitude-step-m 3 --heading-step-deg 60').data
    if($multiTest.stats.target_count -ne 3){throw '第一阶段多目标模板接口自检失败'}
    Draw-Plot $multiTest $ui.MultiCanvas
    $trackLabels=@($ui.MultiCanvas.Children|Where-Object{$_ -is [System.Windows.Controls.TextBlock] -and $_.Text -like 'STAGE1-*'})
    $legends=@($ui.MultiCanvas.Children|Where-Object{$_ -is [System.Windows.Controls.Border]})
    if($trackLabels.Count -ne 3 -or $legends.Count -lt 1){throw '多目标图例或轨迹ID标注自检失败'}
    $importPlot=[PSCustomObject]@{trajectory=@($testData.states);states=@($testData.states);plot_kind='imported'};Draw-Plot $importPlot $ui.ImportCanvas
    $importLegend=@($ui.ImportCanvas.Children|Where-Object{$_ -is [System.Windows.Controls.Border]})|Select-Object -First 1
    $importLegendTexts=@($importLegend.Child.Content.Children|Where-Object{$_ -is [System.Windows.Controls.TextBlock]}|ForEach-Object{$_.Text})
    if(-not ($importLegendTexts -join ' ').Contains('导入状态连线')){throw '导入轨迹图例语义自检失败'}
    [Windows.Devices.Geolocation.Geolocator,Windows.Devices.Geolocation,ContentType=WindowsRuntime]|Out-Null
    Write-Output ('WPF_DRAW_OK elements='+$ui.PlotCanvas.Children.Count)
    Write-Output ('WPF_STAGE1_OK presets='+$presetCount+' targets='+$multiTest.stats.target_count+' legend='+$trackLabels.Count+' importedLegend=ok geo=available')
    exit 0
}
Show-Page '总览'
[void]$Window.ShowDialog()
