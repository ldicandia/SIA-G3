use std::{
    collections::VecDeque,
    fs::File,
    io::{BufWriter, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, UdpSocket},
    path::Path,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::training::DigitEpochMetrics;

const QUEUE_CAPACITY: usize = 256;
const MAX_DASHBOARD_EVENTS: usize = 50_000;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LiveMetric {
    pub session: String,
    pub exercise: String,
    pub candidate: String,
    pub run: String,
    pub epoch: usize,
    pub train_loss: f64,
    pub evaluation_loss: Option<f64>,
    pub train_accuracy: f64,
    pub validation_accuracy: Option<f64>,
    pub timestamp_ms: u64,
}

enum PublisherCommand {
    Metric(LiveMetric),
    Finish,
}

pub struct LiveMetricsPublisher {
    exercise: String,
    session: String,
    sender: Option<SyncSender<PublisherCommand>>,
    worker: Option<JoinHandle<()>>,
    dropped: Arc<AtomicUsize>,
}

#[derive(Debug, Error)]
pub enum LiveError {
    #[error("could not initialize live metrics: {0}")]
    Io(#[from] std::io::Error),
}

impl LiveMetricsPublisher {
    pub fn disabled(exercise: &str) -> Self {
        Self {
            exercise: exercise.to_owned(),
            session: String::new(),
            sender: None,
            worker: None,
            dropped: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn new(exercise: &str, output: &Path, target: SocketAddr) -> Result<Self, LiveError> {
        let file = File::create(output.join("live_metrics.csv"))?;
        let socket = UdpSocket::bind("127.0.0.1:0")?;
        socket.set_nonblocking(true)?;
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let dropped = Arc::new(AtomicUsize::new(0));
        let session = format!("{}-{}", std::process::id(), unix_time_ms());
        let worker = thread::spawn(move || {
            let mut writer = csv::Writer::from_writer(BufWriter::new(file));
            while let Ok(command) = receiver.recv() {
                match command {
                    PublisherCommand::Metric(metric) => {
                        let _ = writer.serialize(&metric);
                        let _ = writer.flush();
                        if let Ok(payload) = serde_json::to_vec(&metric) {
                            let _ = socket.send_to(&payload, target);
                        }
                    }
                    PublisherCommand::Finish => break,
                }
            }
            let _ = writer.flush();
        });
        Ok(Self {
            exercise: exercise.to_owned(),
            session,
            sender: Some(sender),
            worker: Some(worker),
            dropped,
        })
    }

    pub fn publish(&self, candidate: &str, run: &str, metrics: &DigitEpochMetrics) {
        let Some(sender) = &self.sender else {
            return;
        };
        let metric = LiveMetric {
            session: self.session.clone(),
            exercise: self.exercise.clone(),
            candidate: candidate.to_owned(),
            run: run.to_owned(),
            epoch: metrics.epoch,
            train_loss: metrics.train_loss,
            evaluation_loss: metrics.validation_loss,
            train_accuracy: metrics.train_accuracy,
            validation_accuracy: metrics.validation_accuracy,
            timestamp_ms: unix_time_ms(),
        };
        if let Err(error) = sender.try_send(PublisherCommand::Metric(metric)) {
            if matches!(error, TrySendError::Full(_)) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn dropped_events(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }
}

pub fn spawn_monitor_process(
    udp_address: SocketAddr,
    http_address: SocketAddr,
) -> Result<u32, LiveError> {
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("monitor")
        .arg("--udp-address")
        .arg(udp_address.to_string())
        .arg("--http-address")
        .arg(http_address.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let child = command.spawn()?;
    Ok(child.id())
}

impl Drop for LiveMetricsPublisher {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(PublisherCommand::Finish);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn run_monitor(udp_address: SocketAddr, http_address: SocketAddr) -> Result<(), LiveError> {
    let socket = UdpSocket::bind(udp_address)?;
    socket.set_read_timeout(Some(Duration::from_millis(500)))?;
    let events = Arc::new(Mutex::new(VecDeque::<LiveMetric>::new()));
    let udp_events = Arc::clone(&events);
    thread::spawn(move || receive_metrics(socket, udp_events));

    let listener = TcpListener::bind(http_address)?;
    eprintln!("live dashboard: http://{http_address}");
    eprintln!("listening for training metrics on udp://{udp_address}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let request_events = Arc::clone(&events);
                thread::spawn(move || {
                    let _ = handle_http(stream, &request_events);
                });
            }
            Err(error) => eprintln!("dashboard connection error: {error}"),
        }
    }
    Ok(())
}

fn receive_metrics(socket: UdpSocket, events: Arc<Mutex<VecDeque<LiveMetric>>>) {
    let mut buffer = [0_u8; 65_507];
    loop {
        match socket.recv_from(&mut buffer) {
            Ok((size, _)) => {
                if let Ok(metric) = serde_json::from_slice::<LiveMetric>(&buffer[..size]) {
                    if let Ok(mut stored) = events.lock() {
                        if stored.len() == MAX_DASHBOARD_EVENTS {
                            stored.pop_front();
                        }
                        stored.push_back(metric);
                    }
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(error) => {
                eprintln!("dashboard UDP error: {error}");
                thread::sleep(Duration::from_millis(250));
            }
        }
    }
}

fn handle_http(
    mut stream: TcpStream,
    events: &Arc<Mutex<VecDeque<LiveMetric>>>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut request = [0_u8; 4096];
    let size = stream.read(&mut request)?;
    let first_line = String::from_utf8_lossy(&request[..size])
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    let path = first_line.split_whitespace().nth(1).unwrap_or("/");
    match path {
        "/" | "/index.html" => write_response(&mut stream, "text/html; charset=utf-8", DASHBOARD),
        "/metrics" => {
            let body = events
                .lock()
                .ok()
                .and_then(|events| serde_json::to_string(&*events).ok())
                .unwrap_or_else(|| "[]".to_owned());
            write_response(&mut stream, "application/json", &body)
        }
        "/health" => write_response(&mut stream, "text/plain; charset=utf-8", "ok"),
        _ => write_status(&mut stream, "404 Not Found", "not found"),
    }
}

fn write_response(stream: &mut TcpStream, content_type: &str, body: &str) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())
}

fn write_status(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

const DASHBOARD: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TP3 Training Monitor</title>
<style>
:root{color-scheme:dark;--bg:#0b0f12;--surface:#11171b;--line:#263139;--text:#f1f5f7;--muted:#84929b;--accent:#42e8c6;--secondary:#ff9d66;--danger:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;min-height:100vh}
header{display:flex;align-items:flex-end;justify-content:space-between;padding:30px 38px 22px;border-bottom:1px solid var(--line)}
h1{font-size:clamp(24px,3vw,42px);letter-spacing:-.04em;margin:0;font-weight:650}.eyebrow{color:var(--accent);font:600 11px/1.4 ui-monospace,SFMono-Regular,monospace;letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px}
.connection{display:flex;align-items:center;gap:9px;color:var(--muted);font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:var(--danger);transition:background .2s,box-shadow .2s}.dot.online{background:var(--accent);box-shadow:0 0 18px #42e8c688}
main{padding:24px 38px 40px}.toolbar{display:grid;grid-template-columns:minmax(200px,1.4fr) repeat(5,minmax(100px,1fr));gap:16px;align-items:end;padding-bottom:24px;border-bottom:1px solid var(--line)}
label,.metric-label{display:block;color:var(--muted);font-size:11px;letter-spacing:.09em;text-transform:uppercase;margin-bottom:7px}.metric-value{font:560 20px/1.1 ui-monospace,SFMono-Regular,monospace}.latest-run{font-size:14px;line-height:1.35;overflow-wrap:anywhere}
.series-legend{display:flex;flex-wrap:wrap;gap:10px 18px;padding:17px 0 0;color:var(--muted);font-size:12px}.series-item{display:inline-flex;align-items:center;gap:7px}.series-dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto}
.chart-section{padding:28px 0 18px;border-bottom:1px solid var(--line)}.chart-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px}.chart-head h2{font-size:16px;margin:0;font-weight:560}.legend{display:flex;gap:18px;color:var(--muted);font-size:12px}.swatch{display:inline-block;width:18px;margin-right:7px;vertical-align:middle;border-top:2px solid var(--muted)}.validation{border-top-style:dashed}
canvas{width:100%;height:280px;display:block}.empty{position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);font-size:14px;pointer-events:none}.canvas-wrap{position:relative}
@media(max-width:1050px){.toolbar{grid-template-columns:repeat(5,1fr)}.latest{grid-column:1/-1}}@media(max-width:850px){header{padding:24px 20px 18px}main{padding:20px}.toolbar{grid-template-columns:1fr 1fr}.latest{grid-column:1/-1}.metric-value{font-size:17px}canvas{height:230px}}
</style>
</head>
<body>
<header><div><div class="eyebrow">TP3 / live telemetry</div><h1>Training monitor</h1></div><div class="connection"><span id="dot" class="dot"></span><span id="status">Waiting for metrics</span></div></header>
<main>
  <section class="toolbar">
    <div class="latest"><span class="metric-label">Latest candidate / run</span><span id="latestRun" class="latest-run">Waiting for metrics</span></div>
    <div><span class="metric-label">Epoch</span><span id="epoch" class="metric-value">—</span></div>
    <div><span class="metric-label">Train loss</span><span id="trainLoss" class="metric-value">—</span></div>
    <div><span class="metric-label">Evaluation loss</span><span id="evaluationLoss" class="metric-value">—</span></div>
    <div><span class="metric-label">Train accuracy</span><span id="trainAccuracy" class="metric-value">—</span></div>
    <div><span class="metric-label">Validation accuracy</span><span id="validationAccuracy" class="metric-value">—</span></div>
  </section>
  <section id="seriesLegend" class="series-legend"></section>
  <section class="chart-section"><div class="chart-head"><h2>Cross-entropy loss</h2><div class="legend"><span><i class="swatch"></i>train</span><span><i class="swatch validation"></i>evaluation</span></div></div><div class="canvas-wrap"><canvas id="loss"></canvas><div id="lossEmpty" class="empty">Start a training run to stream epochs.</div></div></section>
  <section class="chart-section"><div class="chart-head"><h2>Accuracy</h2><div class="legend"><span><i class="swatch"></i>train</span><span><i class="swatch validation"></i>validation</span></div></div><div class="canvas-wrap"><canvas id="accuracy"></canvas><div id="accuracyEmpty" class="empty">Accuracy appears with the first epoch.</div></div></section>
</main>
<script>
const colors={grid:'#263139',text:'#84929b',series:['#42e8c6','#ff9d66','#7aa2ff','#e879f9','#f5d76e','#7ee787','#ff7b86','#a78bfa','#67d4ff','#f0a6ca']};let all=[],lastCount=0;
const key=m=>`${m.session}|${m.exercise}|${m.candidate}|${m.run}`;const label=m=>`${m.exercise} · ${m.candidate} · ${m.run}`;
function number(v,d=5){return v==null?'—':Number(v).toFixed(d)}
function grouped(){const groups=new Map();for(const m of all){const k=key(m);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(m)}for(const rows of groups.values())rows.sort((a,b)=>a.epoch-b.epoch);return groups}
function renderLegend(groups){seriesLegend.innerHTML='';let i=0;for(const rows of groups.values()){const item=document.createElement('span');item.className='series-item';const dot=document.createElement('i');dot.className='series-dot';dot.style.background=colors.series[i++%colors.series.length];item.append(dot,document.createTextNode(label(rows[0])));seriesLegend.appendChild(item)}}
function setup(canvas){const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w*ratio||canvas.height!==h*ratio){canvas.width=w*ratio;canvas.height=h*ratio}const c=canvas.getContext('2d');c.setTransform(ratio,0,0,ratio,0,0);return{c,w,h}}
function chart(canvas,groups,a,b,fixed){const {c,w,h}=setup(canvas),p={l:52,r:18,t:12,b:30},rows=[...groups.values()].flat();c.clearRect(0,0,w,h);if(!rows.length)return;const vals=rows.flatMap(x=>[x[a],x[b]]).filter(x=>x!=null&&Number.isFinite(x));let min=0,max=fixed?1:Math.max(...vals)*1.08;if(max<=min)max=1;const maxEpoch=Math.max(...rows.map(r=>r.epoch),1),x=epoch=>p.l+(epoch-1)*(w-p.l-p.r)/Math.max(1,maxEpoch-1),y=v=>p.t+(max-v)*(h-p.t-p.b)/(max-min);c.strokeStyle=colors.grid;c.fillStyle=colors.text;c.font='11px ui-monospace';c.lineWidth=1;c.setLineDash([]);for(let i=0;i<=4;i++){const yy=p.t+i*(h-p.t-p.b)/4;c.beginPath();c.moveTo(p.l,yy);c.lineTo(w-p.r,yy);c.stroke();const value=max-i*(max-min)/4;c.fillText(value.toFixed(fixed?2:3),4,yy+4)}let groupIndex=0;for(const seriesRows of groups.values()){const color=colors.series[groupIndex++%colors.series.length];for(const [field,dash] of [[a,[]],[b,[6,5]]]){c.strokeStyle=color;c.lineWidth=2;c.setLineDash(dash);c.beginPath();let started=false,last=null;for(const row of seriesRows){if(row[field]==null)continue;const xx=x(row.epoch),yy=y(row[field]);started?c.lineTo(xx,yy):c.moveTo(xx,yy);started=true;last=[xx,yy]}c.stroke();if(last){c.fillStyle=color;c.beginPath();c.arc(last[0],last[1],2.5,0,Math.PI*2);c.fill()}}}c.setLineDash([]);c.fillStyle=colors.text;c.fillText('1',p.l,h-8);c.textAlign='right';c.fillText(String(maxEpoch),w-p.r,h-8);c.textAlign='left'}
function render(){const groups=grouped(),latest=all.at(-1);for(const id of ['lossEmpty','accuracyEmpty'])document.querySelector('#'+id).style.display=all.length?'none':'grid';renderLegend(groups);if(latest){latestRun.textContent=label(latest);epoch.textContent=latest.epoch;trainLoss.textContent=number(latest.train_loss);evaluationLoss.textContent=number(latest.evaluation_loss);trainAccuracy.textContent=number(latest.train_accuracy,4);validationAccuracy.textContent=number(latest.validation_accuracy,4)}requestAnimationFrame(()=>{chart(document.querySelector('#loss'),groups,'train_loss','evaluation_loss',false);chart(document.querySelector('#accuracy'),groups,'train_accuracy','validation_accuracy',true)})}
async function refresh(){const statusText=document.querySelector('#status');try{const response=await fetch('/metrics',{cache:'no-store'});all=await response.json();render();dot.classList.add('online');statusText.textContent=all.length===lastCount?'Connected · no new epoch':`Live · ${all.length} epochs received`;lastCount=all.length}catch(_){dot.classList.remove('online');statusText.textContent='Monitor disconnected'}}
setInterval(refresh,500);addEventListener('resize',render);refresh();
</script>
</body>
</html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_metric_json_round_trip_preserves_optional_evaluation_values() {
        let metric = LiveMetric {
            session: "s".into(),
            exercise: "exercise2".into(),
            candidate: "candidate".into(),
            run: "search".into(),
            epoch: 3,
            train_loss: 0.2,
            evaluation_loss: Some(0.3),
            train_accuracy: 0.9,
            validation_accuracy: Some(0.8),
            timestamp_ms: 1,
        };
        let decoded: LiveMetric =
            serde_json::from_slice(&serde_json::to_vec(&metric).unwrap()).unwrap();
        assert_eq!(decoded.epoch, 3);
        assert_eq!(decoded.evaluation_loss, Some(0.3));
    }

    #[test]
    fn disabled_publisher_drops_metrics_without_queueing_work() {
        let publisher = LiveMetricsPublisher::disabled("exercise2");
        let metrics = DigitEpochMetrics {
            epoch: 1,
            train_loss: 0.4,
            validation_loss: Some(0.5),
            train_accuracy: 0.8,
            validation_accuracy: Some(0.7),
        };

        publisher.publish("candidate", "search", &metrics);

        assert_eq!(publisher.dropped_events(), 0);
        assert!(publisher.sender.is_none());
        assert!(publisher.worker.is_none());
    }
}
