use anyhow::{anyhow, Context, Result};
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

/// A source of YUYV 4:2:2 frames. Abstracted so the pipeline can be tested
/// without a camera.
pub trait Source {
    fn next_frame(&mut self) -> Result<&[u8]>;
    fn dimensions(&self) -> (u32, u32);
}

pub struct V4lSource {
    stream: MmapStream<'static>,
    width: u32,
    height: u32,
    buffer: Vec<u8>,
}

impl V4lSource {
    pub fn open(path: &str, width: u32, height: u32) -> Result<V4lSource> {
        let dev = Device::with_path(path).with_context(|| format!("opening {path}"))?;

        let mut fmt = Capture::format(&dev).context("querying capture format")?;
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = FourCC::new(b"YUYV");
        let fmt = Capture::set_format(&dev, &fmt).context("setting capture format")?;

        if fmt.width != width || fmt.height != height || fmt.fourcc != FourCC::new(b"YUYV") {
            return Err(anyhow!(
                "{path} would not accept YUYV {width}x{height}; it offered {} {}x{}",
                fmt.fourcc,
                fmt.width,
                fmt.height
            ));
        }

        // The stream borrows the device and both live for the process lifetime;
        // leaking avoids a self-referential struct for no practical cost.
        let dev: &'static Device = Box::leak(Box::new(dev));
        let stream = MmapStream::with_buffers(dev, Type::VideoCapture, 4)
            .context("allocating capture buffers")?;

        Ok(V4lSource {
            stream,
            width,
            height,
            buffer: vec![0u8; (width * height * 2) as usize],
        })
    }
}

impl Source for V4lSource {
    fn next_frame(&mut self) -> Result<&[u8]> {
        let (buf, _meta) = CaptureStream::next(&mut self.stream).context("capturing frame")?;
        let n = self.buffer.len().min(buf.len());
        self.buffer[..n].copy_from_slice(&buf[..n]);
        Ok(&self.buffer)
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Deterministic in-memory source for tests. Cycles through its frames.
pub struct TestSource {
    width: u32,
    height: u32,
    frames: Vec<Vec<u8>>,
    index: usize,
}

impl TestSource {
    pub fn new(width: u32, height: u32, frames: Vec<Vec<u8>>) -> TestSource {
        TestSource {
            width,
            height,
            frames,
            index: 0,
        }
    }
}

impl Source for TestSource {
    fn next_frame(&mut self) -> Result<&[u8]> {
        if self.frames.is_empty() {
            return Err(anyhow!("TestSource has no frames"));
        }
        let i = self.index % self.frames.len();
        self.index += 1;
        Ok(&self.frames[i])
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_cycles_frames() {
        let a = vec![1u8; 8];
        let b = vec![2u8; 8];
        let mut src = TestSource::new(2, 2, vec![a.clone(), b.clone()]);
        assert_eq!(src.dimensions(), (2, 2));
        assert_eq!(src.next_frame().unwrap(), &a[..]);
        assert_eq!(src.next_frame().unwrap(), &b[..]);
        assert_eq!(src.next_frame().unwrap(), &a[..], "should wrap around");
    }

    #[test]
    fn opening_a_missing_device_is_an_error() {
        assert!(V4lSource::open("/dev/definitely-not-a-camera", 1024, 576).is_err());
    }
}
