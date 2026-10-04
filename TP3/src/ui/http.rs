//! Minimal HTTP/1.1 plumbing for the local UI: bounded request reader, response
//! builder and the guards that keep other web pages away from the server.
//! Standard library only; every connection serves exactly one request.

use std::io::{self, Read};

use serde::Serialize;

pub const MAX_HEAD_BYTES: usize = 16 * 1024;
pub const MAX_BODY_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: Vec<(String, String)>,
    /// Header names are stored lowercased.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn query_param(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

#[derive(Debug)]
pub enum HttpError {
    Malformed,
    HeadTooLarge,
    BodyTooLarge,
    Unsupported,
    /// The socket failed or timed out; the connection is simply closed.
    Io,
}

impl HttpError {
    pub fn status(&self) -> u16 {
        match self {
            Self::Malformed | Self::Io => 400,
            Self::HeadTooLarge => 431,
            Self::BodyTooLarge => 413,
            Self::Unsupported => 501,
        }
    }

    pub fn message(&self) -> &'static str {
        match self {
            Self::Malformed => "malformed request",
            Self::HeadTooLarge => "request head too large",
            Self::BodyTooLarge => "request body too large",
            Self::Unsupported => "unsupported request feature",
            Self::Io => "could not read the request",
        }
    }
}

impl From<io::Error> for HttpError {
    fn from(_: io::Error) -> Self {
        Self::Io
    }
}

fn find_head_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

pub fn read_request(reader: &mut impl Read) -> Result<Request, HttpError> {
    let mut buffer: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        if let Some(position) = find_head_end(&buffer) {
            break position;
        }
        if buffer.len() > MAX_HEAD_BYTES {
            return Err(HttpError::HeadTooLarge);
        }
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            return Err(HttpError::Malformed);
        }
        buffer.extend_from_slice(&chunk[..read]);
    };
    if head_end > MAX_HEAD_BYTES {
        return Err(HttpError::HeadTooLarge);
    }
    let head = std::str::from_utf8(&buffer[..head_end]).map_err(|_| HttpError::Malformed)?;
    let mut lines = head.split("\r\n");
    let request_line = lines.next().ok_or(HttpError::Malformed)?;
    let mut parts = request_line.split(' ');
    let (method, target, version) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(method), Some(target), Some(version), None) => (method, target, version),
        _ => return Err(HttpError::Malformed),
    };
    if method.is_empty()
        || !method.bytes().all(|byte| byte.is_ascii_uppercase())
        || !target.starts_with('/')
        || !version.starts_with("HTTP/1.")
    {
        return Err(HttpError::Malformed);
    }
    let mut headers = Vec::new();
    for line in lines {
        let (name, value) = line.split_once(':').ok_or(HttpError::Malformed)?;
        if name.is_empty() || name.bytes().any(|byte| byte.is_ascii_whitespace()) {
            return Err(HttpError::Malformed);
        }
        headers.push((name.to_ascii_lowercase(), value.trim().to_owned()));
    }
    if headers.iter().any(|(name, _)| name == "transfer-encoding") {
        return Err(HttpError::Unsupported);
    }
    let (path, query) = split_target(target);
    let mut request = Request {
        method: method.to_owned(),
        path,
        query,
        headers,
        body: Vec::new(),
    };
    if request.method == "POST" {
        let declared = request
            .header("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or(HttpError::Malformed)?;
        if declared > MAX_BODY_BYTES {
            return Err(HttpError::BodyTooLarge);
        }
        let mut body = buffer[head_end + 4..].to_vec();
        body.truncate(declared);
        while body.len() < declared {
            let want = (declared - body.len()).min(chunk.len());
            let read = reader.read(&mut chunk[..want])?;
            if read == 0 {
                return Err(HttpError::Malformed);
            }
            body.extend_from_slice(&chunk[..read]);
        }
        request.body = body;
    }
    Ok(request)
}

/// Splits `/path?a=b&c=d`. No percent-decoding: routes only accept ids made of
/// ASCII letters, digits and '-', so there is nothing to decode.
pub fn split_target(target: &str) -> (String, Vec<(String, String)>) {
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path, query),
        None => (target, ""),
    };
    let pairs = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((key, value)) => (key.to_owned(), value.to_owned()),
            None => (pair.to_owned(), String::new()),
        })
        .collect();
    (path.to_owned(), pairs)
}

#[derive(Clone, Debug)]
pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
    pub extra_headers: Vec<(String, String)>,
}

impl Response {
    pub fn json(status: u16, value: &impl Serialize) -> Self {
        match serde_json::to_vec(value) {
            Ok(body) => Self {
                status,
                content_type: "application/json".into(),
                body,
                extra_headers: Vec::new(),
            },
            Err(_) => Self::error(500, "could not serialize the response"),
        }
    }

    pub fn text(status: u16, content_type: &str, body: String) -> Self {
        Self {
            status,
            content_type: content_type.to_owned(),
            body: body.into_bytes(),
            extra_headers: Vec::new(),
        }
    }

    pub fn error(status: u16, message: &str) -> Self {
        let body = serde_json::json!({ "error": message }).to_string();
        Self::text(status, "application/json", body)
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.extra_headers.push((name.to_owned(), value.to_owned()));
        self
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n",
            self.status,
            reason_phrase(self.status),
            self.content_type,
            self.body.len()
        );
        for (name, value) in &self.extra_headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");
        let mut bytes = head.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "Unknown",
    }
}

pub fn allowed_hosts(port: u16) -> Vec<String> {
    vec![
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        format!("[::1]:{port}"),
    ]
}

/// Defence against other web pages reaching the server through the browser
/// (CSRF and DNS rebinding): allowed Host, allowed Origin when present, and a
/// JSON content type on POST (which forces a CORS preflight the server never
/// answers; OPTIONS is a 405).
pub fn check_guards(request: &Request, port: u16) -> Result<(), Response> {
    let hosts = allowed_hosts(port);
    match request.header("host") {
        Some(host) if hosts.iter().any(|allowed| allowed == host) => {}
        _ => return Err(Response::error(403, "forbidden host")),
    }
    if let Some(origin) = request.header("origin") {
        if !hosts
            .iter()
            .any(|allowed| origin == format!("http://{allowed}"))
        {
            return Err(Response::error(403, "forbidden origin"));
        }
    }
    if request.method == "POST" {
        let json = request
            .header("content-type")
            .is_some_and(|value| value.to_ascii_lowercase().starts_with("application/json"));
        if !json {
            return Err(Response::error(
                415,
                "content type must be application/json",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OneByteReader<'a>(&'a [u8]);

    impl Read for OneByteReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.0[0];
            self.0 = &self.0[1..];
            Ok(1)
        }
    }

    struct PanicAfter<'a> {
        data: &'a [u8],
    }

    impl Read for PanicAfter<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.data.is_empty() {
                panic!("the reader must not be polled for the body");
            }
            let n = self.data.len().min(buf.len());
            buf[..n].copy_from_slice(&self.data[..n]);
            self.data = &self.data[n..];
            Ok(n)
        }
    }

    fn parse(raw: &str) -> Result<Request, HttpError> {
        read_request(&mut raw.as_bytes())
    }

    fn request(method: &str, headers: &[(&str, &str)]) -> Request {
        Request {
            method: method.into(),
            path: "/".into(),
            query: Vec::new(),
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), (*v).to_owned()))
                .collect(),
            body: Vec::new(),
        }
    }

    #[test]
    fn get_with_query_string_parses() {
        let request =
            parse("GET /api/runs/best/config.toml?group=g1-2&exercise=exercise2 HTTP/1.1\r\nHost: localhost:1\r\n\r\n")
                .unwrap();
        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/api/runs/best/config.toml");
        assert_eq!(request.query_param("group"), Some("g1-2"));
        assert_eq!(request.query_param("exercise"), Some("exercise2"));
        assert_eq!(request.query_param("missing"), None);
        assert!(request.body.is_empty());
    }

    #[test]
    fn post_with_body_parses() {
        let request =
            parse("POST /api/runs HTTP/1.1\r\nHost: x\r\nContent-Length: 7\r\n\r\n{\"a\":1}")
                .unwrap();
        assert_eq!(request.body, b"{\"a\":1}");
    }

    #[test]
    fn split_reads_still_parse() {
        let raw = "POST /api/runs HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\n\r\nabcd";
        let request = read_request(&mut OneByteReader(raw.as_bytes())).unwrap();
        assert_eq!(request.body, b"abcd");
        assert_eq!(request.header("host"), Some("x"));
    }

    #[test]
    fn head_over_the_cap_is_rejected() {
        let raw = format!(
            "GET / HTTP/1.1\r\nX-Big: {}\r\n\r\n",
            "a".repeat(MAX_HEAD_BYTES + 100)
        );
        assert!(matches!(parse(&raw), Err(HttpError::HeadTooLarge)));
        let endless = vec![b'a'; MAX_HEAD_BYTES * 2];
        assert!(matches!(
            read_request(&mut endless.as_slice()),
            Err(HttpError::HeadTooLarge)
        ));
    }

    #[test]
    fn declared_body_over_the_cap_is_rejected_without_reading_it() {
        let raw = format!(
            "POST /api/runs HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        let result = read_request(&mut PanicAfter {
            data: raw.as_bytes(),
        });
        assert!(matches!(result, Err(HttpError::BodyTooLarge)));
    }

    #[test]
    fn post_without_a_valid_content_length_is_malformed() {
        assert!(matches!(
            parse("POST /x HTTP/1.1\r\nHost: x\r\n\r\n"),
            Err(HttpError::Malformed)
        ));
        assert!(matches!(
            parse("POST /x HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n"),
            Err(HttpError::Malformed)
        ));
    }

    #[test]
    fn chunked_transfer_is_unsupported() {
        assert!(matches!(
            parse("POST /x HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"),
            Err(HttpError::Unsupported)
        ));
    }

    #[test]
    fn malformed_request_lines_are_rejected() {
        for raw in [
            "GARBAGE\r\n\r\n",
            "GET /a\r\n\r\n",
            "GET a HTTP/1.1\r\n\r\n",
            "get /a HTTP/1.1\r\n\r\n",
            "GET /a SPDY/3\r\n\r\n",
            "GET /a HTTP/1.1 extra\r\n\r\n",
            "GET /a HTTP/1.1\r\nno colon here\r\n\r\n",
        ] {
            assert!(matches!(parse(raw), Err(HttpError::Malformed)), "{raw:?}");
        }
    }

    #[test]
    fn truncated_stream_is_malformed() {
        assert!(matches!(
            parse("GET / HTTP/1.1\r\nHost: x"),
            Err(HttpError::Malformed)
        ));
        assert!(matches!(
            parse("POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc"),
            Err(HttpError::Malformed)
        ));
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let request = parse("GET / HTTP/1.1\r\nHoSt: Example\r\n\r\n").unwrap();
        assert_eq!(request.header("HOST"), Some("Example"));
        assert_eq!(request.header("host"), Some("Example"));
    }

    #[test]
    fn error_statuses_map_as_documented() {
        assert_eq!(HttpError::Malformed.status(), 400);
        assert_eq!(HttpError::HeadTooLarge.status(), 431);
        assert_eq!(HttpError::BodyTooLarge.status(), 413);
        assert_eq!(HttpError::Unsupported.status(), 501);
        assert_eq!(HttpError::Io.status(), 400);
    }

    #[test]
    fn response_bytes_carry_length_and_connection_close() {
        let response = Response::text(200, "text/plain", "hello".into()).with_header("X-A", "b");
        let text = String::from_utf8(response.to_bytes()).unwrap();
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Length: 5\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert!(text.contains("X-Content-Type-Options: nosniff\r\n"));
        assert!(text.contains("X-A: b\r\n"));
        assert!(text.ends_with("\r\n\r\nhello"));
        let error = Response::error(404, "nope");
        assert_eq!(error.body, br#"{"error":"nope"}"#);
    }

    #[test]
    fn guards_cover_every_outcome() {
        let ok = request("GET", &[("Host", "127.0.0.1:7880")]);
        assert!(check_guards(&ok, 7880).is_ok());
        let localhost = request("GET", &[("Host", "localhost:7880")]);
        assert!(check_guards(&localhost, 7880).is_ok());

        let bad_host = request("GET", &[("Host", "evil.example")]);
        assert_eq!(check_guards(&bad_host, 7880).unwrap_err().status, 403);
        let no_host = request("GET", &[]);
        assert_eq!(check_guards(&no_host, 7880).unwrap_err().status, 403);
        let wrong_port = request("GET", &[("Host", "127.0.0.1:1")]);
        assert_eq!(check_guards(&wrong_port, 7880).unwrap_err().status, 403);

        let bad_origin = request(
            "GET",
            &[
                ("Host", "127.0.0.1:7880"),
                ("Origin", "http://evil.example"),
            ],
        );
        assert_eq!(check_guards(&bad_origin, 7880).unwrap_err().status, 403);
        let good_origin = request(
            "GET",
            &[
                ("Host", "127.0.0.1:7880"),
                ("Origin", "http://127.0.0.1:7880"),
            ],
        );
        assert!(check_guards(&good_origin, 7880).is_ok());

        let text_post = request(
            "POST",
            &[("Host", "127.0.0.1:7880"), ("Content-Type", "text/plain")],
        );
        assert_eq!(check_guards(&text_post, 7880).unwrap_err().status, 415);
        let json_post = request(
            "POST",
            &[
                ("Host", "127.0.0.1:7880"),
                ("Content-Type", "application/json; charset=utf-8"),
            ],
        );
        assert!(check_guards(&json_post, 7880).is_ok());
    }
}
