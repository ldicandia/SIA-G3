use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, UdpSocket},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use super::{LiveError, LiveMetric};

const MAX_DASHBOARD_EVENTS: usize = 50_000;

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

const DASHBOARD: &str = include_str!("dashboard.html");
