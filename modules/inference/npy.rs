//! Test-only float32 `.npy` reader (`descr` in `<f4`/`|f4`/`=f4`, C-order, any rank).

use std::io::Read;
use std::path::Path;

pub fn read_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {}", path.display(), e))
        .read_to_end(&mut buf)
        .unwrap();
    assert_eq!(
        &buf[0..6],
        b"\x93NUMPY",
        "{} not an npy file",
        path.display()
    );
    let (header_start, header_len) = match buf[6] {
        1 => (10, u16::from_le_bytes([buf[8], buf[9]]) as usize),
        2 | 3 => (
            12,
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
        ),
        v => panic!("unsupported npy major version {v}"),
    };
    let header = std::str::from_utf8(&buf[header_start..header_start + header_len]).unwrap();
    let descr = extract_str(header, "'descr'");
    assert!(
        matches!(descr, "<f4" | "|f4" | "=f4"),
        "only f32 npy supported, got descr={descr:?} in {}",
        path.display()
    );
    let fortran = extract_str(header, "'fortran_order'");
    assert_eq!(fortran, "False", "fortran-order npy not supported");
    let shape_str = extract_tuple(header, "'shape'");
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let n: usize = shape.iter().product();
    let data_start = header_start + header_len;
    let raw = &buf[data_start..data_start + n * 4];
    let out: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, out)
}

fn extract_str<'a>(header: &'a str, key: &str) -> &'a str {
    let i = header
        .find(key)
        .unwrap_or_else(|| panic!("npy header missing {key}: {header}"));
    let after = &header[i + key.len()..];
    let colon = after.find(':').unwrap();
    let rest = after[colon + 1..].trim_start();
    if let Some(quote) = rest.chars().next().filter(|c| *c == '\'' || *c == '"') {
        let s = &rest[1..];
        let end = s.find(quote).unwrap();
        &s[..end]
    } else {
        let end = rest.find([',', '}']).unwrap();
        rest[..end].trim()
    }
}

fn extract_tuple<'a>(header: &'a str, key: &str) -> &'a str {
    let i = header.find(key).unwrap();
    let after = &header[i + key.len()..];
    let colon = after.find(':').unwrap();
    let rest = &after[colon + 1..];
    let open = rest.find('(').unwrap();
    let close = rest.find(')').unwrap();
    &rest[open + 1..close]
}
